import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from app.agent import (
    AgentState,
    agent_node,
    check_input_scope,
    run_agent,
    should_continue,
)


# ===========================================================================
# 1. Unit tests — check_input_scope
# ===========================================================================

class TestCheckInputScope:
    """Tests for the pre-graph healthcare scope guardrail."""

    def test_valid_healthcare_question_is_allowed(self):
        result = check_input_scope("Which doctors treat hypertension?")
        assert result is None

    def test_valid_appointment_question_is_allowed(self):
        result = check_input_scope("What is the status of appointment AP101?")
        assert result is None

    def test_valid_bmi_question_is_allowed(self):
        result = check_input_scope(
            "My weight is 70 kg and height is 175 cm. What is my BMI?"
        )
        assert result is None

    def test_short_message_is_rejected(self):
        result = check_input_scope("Hi")
        assert result is not None
        assert "too short" in result.lower()

    def test_single_character_is_rejected(self):
        result = check_input_scope("a")
        assert result is not None

    def test_diagnosis_request_is_rejected(self):
        result = check_input_scope("Can you diagnose me?")
        assert result is not None
        assert "cannot diagnose" in result.lower()

    def test_prescription_request_is_rejected(self):
        result = check_input_scope("What medication should I take?")
        assert result is not None
        assert "medications" in result.lower()

    def test_treatment_request_is_rejected(self):
        result = check_input_scope("Can you treat me?")
        assert result is not None
        assert "cannot diagnose" in result.lower()

    def test_off_topic_code_request_is_rejected(self):
        result = check_input_scope("Write code for a Python application")
        assert result is not None
        assert "outside that scope" in result.lower()

    def test_off_topic_weather_request_is_rejected(self):
        result = check_input_scope("What is the weather today?")
        assert result is not None
        assert "healthcare navigation assistant" in result.lower()

    def test_whitespace_only_message_is_rejected(self):
        result = check_input_scope("   ")
        assert result is not None


# ===========================================================================
# 2. Unit tests — should_continue
# ===========================================================================

class TestShouldContinue:
    """Tests for LangGraph conditional routing."""

    def _make_state(
        self,
        messages,
        total_turns=0,
        error=None,
    ):
        return {
            "messages": messages,
            "total_turns": total_turns,
            "request_id": "test1234",
            "error": error,
        }

    def test_plain_ai_response_routes_to_end(self):
        message = AIMessage(content="Hypertension is high blood pressure.")

        state = self._make_state([message])

        result = should_continue(state)

        assert result == "__end__"

    def test_ai_tool_call_routes_to_tools(self):
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculate_bmi",
                    "args": {
                        "weight_kg": 70,
                        "height_cm": 175,
                    },
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )

        state = self._make_state([message])

        result = should_continue(state)

        assert result == "tools"

    def test_llm_error_routes_to_end(self):
        message = AIMessage(content="Temporary failure.")

        state = self._make_state(
            [message],
            error="LLM call failed: RuntimeError",
        )

        result = should_continue(state)

        assert result == "__end__"

    def test_max_turns_routes_to_end(self):
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculate_bmi",
                    "args": {
                        "weight_kg": 70,
                        "height_cm": 175,
                    },
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )

        # Current settings default is MAX_AGENT_TURNS=10.
        # Use the actual configured value rather than hardcoding it.
        from app.config import settings

        state = self._make_state(
            [message],
            total_turns=settings.max_agent_turns,
        )

        result = should_continue(state)

        assert result == "__end__"


# ===========================================================================
# 3. Unit tests — agent_node
# ===========================================================================

class TestAgentNode:
    """Tests for the individual LLM node."""

    def _make_state(self, messages=None):
        return {
            "messages": messages or [
                HumanMessage(content="What is hypertension?")
            ],
            "total_turns": 0,
            "request_id": "test1234",
            "error": None,
        }

    @patch("app.agent._build_llm")
    def test_agent_node_returns_llm_response(self, mock_build_llm):
        mock_llm = MagicMock()

        mock_response = AIMessage(
            content="Hypertension is high blood pressure."
        )

        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        mock_build_llm.return_value = mock_llm

        state = self._make_state()

        result = agent_node(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == (
            "Hypertension is high blood pressure."
        )

    @patch("app.agent._build_llm")
    def test_agent_node_handles_llm_exception(self, mock_build_llm):
        mock_llm = MagicMock()

        mock_llm.bind_tools.return_value.invoke.side_effect = RuntimeError(
            "GROQ unavailable"
        )

        mock_build_llm.return_value = mock_llm

        state = self._make_state()

        result = agent_node(state)

        assert "messages" in result
        assert "error" in result

        assert "LLM call failed" in result["error"]
        assert "RuntimeError" in result["error"]

        # Safe fallback should be returned rather than the raw exception.
        assert "temporary service issue" in result["messages"][0].content


# ===========================================================================
# 4. run_agent — guardrail tests
# ===========================================================================

class TestRunAgentGuardrail:
    """Tests that rejected requests never invoke the LLM graph."""

    @patch("app.agent._graph")
    def test_guardrail_rejection_does_not_call_graph(self, mock_graph):
        result = run_agent("Can you diagnose me?")

        assert "cannot diagnose" in result["answer"].lower()
        assert result["total_turns"] == 0

        mock_graph.invoke.assert_not_called()

    @patch("app.agent._graph")
    def test_off_topic_request_does_not_call_graph(self, mock_graph):
        result = run_agent("Write code for a Python application")

        assert "outside that scope" in result["answer"].lower()
        assert result["total_turns"] == 0

        mock_graph.invoke.assert_not_called()

    @patch("app.agent._graph")
    def test_short_request_does_not_call_graph(self, mock_graph):
        result = run_agent("Hi")

        assert "too short" in result["answer"].lower()
        assert result["total_turns"] == 0

        mock_graph.invoke.assert_not_called()


# ===========================================================================
# 5. run_agent — graph execution tests
# ===========================================================================

class TestRunAgent:
    """Tests for the public run_agent() entry point."""

    @patch("app.agent._graph")
    def test_direct_answer_without_tools(self, mock_graph):
        """
        LLM returns a normal final answer without using any tools.
        """

        final_state = {
            "messages": [
                HumanMessage(content="What is hypertension?"),
                AIMessage(
                    content="Hypertension is high blood pressure."
                ),
            ],
            "total_turns": 0,
            "request_id": "test1234",
            "error": None,
        }

        mock_graph.invoke.return_value = final_state

        result = run_agent("What is hypertension?")

        assert result["answer"] == "Hypertension is high blood pressure."
        assert result["total_turns"] == 0

        mock_graph.invoke.assert_called_once()

    @patch("app.agent._graph")
    def test_graph_result_with_one_tool_turn(self, mock_graph):
        """
        Graph completes after one tool round-trip.
        """

        final_state = {
            "messages": [
                HumanMessage(
                    content="I weigh 70 kg and am 175 cm tall. What is my BMI?"
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculate_bmi",
                            "args": {
                                "weight_kg": 70,
                                "height_cm": 175,
                            },
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="Your BMI is 22.86, which is Normal weight."
                ),
            ],
            "total_turns": 1,
            "request_id": "test1234",
            "error": None,
        }

        mock_graph.invoke.return_value = final_state

        result = run_agent(
            "I weigh 70 kg and am 175 cm tall. What is my BMI?"
        )

        assert result["answer"] == (
            "Your BMI is 22.86, which is Normal weight."
        )
        assert result["total_turns"] == 1

    @patch("app.agent._graph")
    def test_final_answer_is_last_ai_message_with_content(self, mock_graph):
        """
        _extract_final_answer() should return the latest AI message
        containing content, even when previous AI messages contain
        tool calls.
        """

        final_state = {
            "messages": [
                HumanMessage(content="Find a doctor for hypertension."),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "find_doctor",
                            "args": {
                                "condition": "Hypertension"
                            },
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="Dr. Sharma specializes in Cardiology."
                ),
            ],
            "total_turns": 1,
            "request_id": "test1234",
            "error": None,
        }

        mock_graph.invoke.return_value = final_state

        result = run_agent(
            "Find a doctor for hypertension."
        )

        assert result["answer"] == (
            "Dr. Sharma specializes in Cardiology."
        )

    @patch("app.agent._graph")
    def test_run_agent_returns_safe_fallback_when_no_ai_content(
        self,
        mock_graph,
    ):
        """
        If the graph finishes without a usable AI response,
        run_agent should return the fallback message.
        """

        final_state = {
            "messages": [
                HumanMessage(content="Some healthcare question")
            ],
            "total_turns": 0,
            "request_id": "test1234",
            "error": None,
        }

        mock_graph.invoke.return_value = final_state

        result = run_agent("Some healthcare question")

        assert "unable to generate a response" in result["answer"].lower()

    @patch("app.agent._graph")
    def test_run_agent_returns_error_fallback_when_graph_has_error(
        self,
        mock_graph,
    ):
        """
        If the graph state contains an error and no AI content,
        the final response should be the safe error fallback.
        """

        final_state = {
            "messages": [
                HumanMessage(content="What is hypertension?")
            ],
            "total_turns": 0,
            "request_id": "test1234",
            "error": "LLM call failed: RuntimeError",
        }

        mock_graph.invoke.return_value = final_state

        result = run_agent("What is hypertension?")

        assert "encountered an issue" in result["answer"].lower()
        assert result["total_turns"] == 0


# ===========================================================================
# 6. Basic state / architecture tests
# ===========================================================================

class TestAgentArchitecture:
    """Small tests verifying important agent configuration."""

    def test_registered_tools_exist(self):
        from app.agent import REGISTERED_TOOLS

        tool_names = {tool.name for tool in REGISTERED_TOOLS}

        assert "find_doctor" in tool_names
        assert "get_appointment" in tool_names
        assert "calculate_bmi" in tool_names

    def test_agent_state_has_expected_fields(self):
        from app.agent import AgentState

        annotations = AgentState.__annotations__

        assert "messages" in annotations
        assert "total_turns" in annotations
        assert "request_id" in annotations
        assert "error" in annotations

    def test_system_prompt_contains_navigation_scope(self):
        from app.agent import SYSTEM_PROMPT

        prompt = SYSTEM_PROMPT.lower()

        assert "healthcare navigation assistant" in prompt
        assert "find_doctor" in prompt
        assert "get_appointment" in prompt
        assert "calculate_bmi" in prompt

    def test_system_prompt_restricts_diagnosis(self):
        from app.agent import SYSTEM_PROMPT

        assert "diagnose" in SYSTEM_PROMPT.lower()
        assert "medications" in SYSTEM_PROMPT.lower()