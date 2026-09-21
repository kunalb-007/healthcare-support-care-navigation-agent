"""
tests/test_agent.py
--------------------
Tests for the LangGraph agent orchestration.

All LLM calls are mocked — no real API calls are made.
Tests verify:
  - Direct responses (no tool calls)
  - Tool routing and tracking
  - Maximum iteration limit
  - LLM failure handling
  - should_continue routing logic
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent import (
    AgentState,
    _extract_final_answer,
    run_agent,
    should_continue,
)
from app.config import settings


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_state(
        messages=None,
        tools_used=None,
        total_turns=0,
        request_id="test-001",
        error=None,
) -> AgentState:
    """Build a minimal AgentState for routing tests."""
    return AgentState(
        messages=messages or [HumanMessage(content="Hello")],
        tools_used=tools_used or [],
        total_turns=total_turns,
        request_id=request_id,
        error=error,
    )


def _ai_message_with_tool_call(tool_name="find_doctor") -> AIMessage:
    """Build an AIMessage that contains a tool call."""
    msg = AIMessage(content="")
    msg.tool_calls = [
        {
            "name": tool_name,
            "args": {"condition": "Hypertension"},
            "id": "call_001",
            "type": "tool_call",
        }
    ]
    return msg


def _ai_message_plain(content="Here is your answer.") -> AIMessage:
    """Build a plain AIMessage with no tool calls."""
    msg = AIMessage(content=content)
    msg.tool_calls = []
    return msg


# ------------------------------------------------------------------
# should_continue routing tests
# ------------------------------------------------------------------

class TestShouldContinue:
    """Unit tests for the conditional routing function."""

    def test_routes_to_tools_when_tool_call_present(self):
        """If last message has tool_calls, route to 'tools'."""
        state = _make_state(
            messages=[_ai_message_with_tool_call()],
            total_turns=0,
        )
        result = should_continue(state)
        assert result == "tools"

    def test_routes_to_end_when_no_tool_calls(self):
        """If last message has no tool_calls, route to END."""
        from langgraph.graph import END
        state = _make_state(
            messages=[_ai_message_plain()],
            total_turns=0,
        )
        result = should_continue(state)
        assert result == END

    def test_routes_to_end_when_max_turns_reached(self):
        """If total_turns >= MAX_AGENT_TURNS, stop even if tool_calls present."""
        from langgraph.graph import END
        state = _make_state(
            messages=[_ai_message_with_tool_call()],
            total_turns=settings.max_agent_turns,  # at the limit
        )
        result = should_continue(state)
        assert result == END

    def test_routes_to_end_on_error(self):
        """If state.error is set, always route to END."""
        from langgraph.graph import END
        state = _make_state(
            messages=[_ai_message_with_tool_call()],
            error="LLM call failed",
        )
        result = should_continue(state)
        assert result == END

    def test_routes_to_tools_just_under_max_turns(self):
        """One turn below MAX_AGENT_TURNS should still route to tools."""
        state = _make_state(
            messages=[_ai_message_with_tool_call()],
            total_turns=settings.max_agent_turns - 1,
        )
        result = should_continue(state)
        assert result == "tools"


# ------------------------------------------------------------------
# _extract_final_answer tests
# ------------------------------------------------------------------

class TestExtractFinalAnswer:
    """Tests for the helper that extracts the last AI response."""

    def test_extracts_last_ai_message(self):
        state = _make_state(
            messages=[
                HumanMessage(content="Hello"),
                _ai_message_plain("First response"),
                ToolMessage(content="tool result", tool_call_id="x"),
                _ai_message_plain("Final answer"),
            ]
        )
        answer = _extract_final_answer(state)
        assert answer == "Final answer"

    def test_returns_fallback_when_no_ai_message(self):
        state = _make_state(messages=[HumanMessage(content="Hello")])
        answer = _extract_final_answer(state)
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_returns_error_message_on_graph_error(self):
        state = _make_state(
            messages=[HumanMessage(content="Hello")],
            error="Something failed",
        )
        answer = _extract_final_answer(state)
        assert isinstance(answer, str)
        assert len(answer) > 0


# ------------------------------------------------------------------
# run_agent integration tests (mocked LLM)
# ------------------------------------------------------------------

class TestRunAgentDirectResponse:
    """Tests for queries answered directly without tool calls."""

    def test_direct_response_no_tools(self):
        """LLM answers directly → tools_used is empty, total_turns is 0."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = _ai_message_plain("Hypertension is high blood pressure.")

        with patch("app.agent._build_llm", return_value=mock_llm):
            result = run_agent("What is hypertension?")

        assert result["answer"] == "Hypertension is high blood pressure."
        assert result["tools_used"] == []
        assert result["total_turns"] == 0

    def test_result_has_required_keys(self):
        """run_agent always returns answer, tools_used, total_turns."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = _ai_message_plain("A direct answer.")

        with patch("app.agent._build_llm", return_value=mock_llm):
            result = run_agent("Hello")

        assert "answer" in result
        assert "tools_used" in result
        assert "total_turns" in result


class TestRunAgentMaxIterations:
    """Tests for the maximum iteration guard."""

    def test_stops_at_max_turns(self):
        """
        When the LLM always returns tool calls, the agent must stop
        at MAX_AGENT_TURNS and return a response — not loop forever.
        """
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        # Always return a tool call
        mock_llm.invoke.return_value = _ai_message_with_tool_call()

        # Mock the tool execution to return a simple result
        mock_tool_node_result = {
            "messages": [ToolMessage(content='{"doctors": []}', tool_call_id="call_001")],
        }

        with patch("app.agent._build_llm", return_value=mock_llm):
            with patch("app.agent.tool_node") as mock_tool_node:
                mock_tool_node.invoke.return_value = mock_tool_node_result
                result = run_agent("Find me a doctor")

        # Must terminate and return something
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0
        # Must not exceed max turns
        assert result["total_turns"] <= settings.max_agent_turns


class TestRunAgentLLMFailure:
    """Tests for LLM failure scenarios."""

    def test_llm_exception_returns_safe_message(self):
        """If the LLM call raises an exception, return a safe message."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.side_effect = Exception("Connection refused to OpenRouter")

        with patch("app.agent._build_llm", return_value=mock_llm):
            result = run_agent("What is my BMI?")

        # Must return a string, not raise an exception
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0
        # Must NOT leak connection details to the user
        assert "OpenRouter" not in result["answer"]
        assert "Connection refused" not in result["answer"]

    def test_llm_exception_does_not_propagate(self):
        """run_agent must never raise — always return a dict."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.side_effect = RuntimeError("timeout")

        with patch("app.agent._build_llm", return_value=mock_llm):
            # This must NOT raise
            result = run_agent("Hello")

        assert isinstance(result, dict)