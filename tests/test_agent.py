"""
Unit tests for the Healthcare Agent.

Run with:  pytest tests/ -v

These tests are designed to be explainable in 30 seconds each during an interview.
They test the core logic WITHOUT making real API/DB calls (using mocks).
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from app.agent import (
    validate_tool_args,
    dispatch_tool,
    run_agent,
)
from app.tools.calculate_bmi import calculate_bmi


# ===========================================================================
# 1. Unit tests — calculate_bmi  (pure function, no mocking needed)
# ===========================================================================

class TestCalculateBMI:

    def test_normal_weight(self):
        result = calculate_bmi(70, 175)
        assert result["bmi"] == 22.86
        assert result["category"] == "Normal weight"

    def test_underweight(self):
        result = calculate_bmi(45, 175)
        assert result["category"] == "Underweight"

    def test_overweight(self):
        result = calculate_bmi(85, 175)
        assert result["category"] == "Overweight"

    def test_obese(self):
        result = calculate_bmi(110, 175)
        assert result["category"] == "Obese"

    def test_zero_weight_returns_error(self):
        result = calculate_bmi(0, 175)
        assert "error" in result

    def test_negative_height_returns_error(self):
        result = calculate_bmi(70, -5)
        assert "error" in result


# ===========================================================================
# 2. Unit tests — validate_tool_args
# ===========================================================================

class TestValidateToolArgs:

    def test_valid_bmi_args(self):
        assert validate_tool_args("calculate_bmi", {"weight_kg": 70, "height_cm": 175}) is None

    def test_missing_height(self):
        error = validate_tool_args("calculate_bmi", {"weight_kg": 70})
        assert error is not None
        assert "height_cm" in error

    def test_negative_weight(self):
        error = validate_tool_args("calculate_bmi", {"weight_kg": -10, "height_cm": 170})
        assert error is not None

    def test_valid_appointment_id(self):
        assert validate_tool_args("get_appointment", {"appointment_id": "AP101"}) is None

    def test_empty_appointment_id(self):
        error = validate_tool_args("get_appointment", {"appointment_id": ""})
        assert error is not None

    def test_find_doctor_no_args(self):
        error = validate_tool_args("find_doctor", {})
        assert error is not None

    def test_find_doctor_with_condition(self):
        assert validate_tool_args("find_doctor", {"condition": "Hypertension"}) is None

    def test_find_doctor_with_specialization(self):
        assert validate_tool_args("find_doctor", {"specialization": "Cardiology"}) is None


# ===========================================================================
# 3. Unit tests — dispatch_tool
# ===========================================================================

class TestDispatchTool:

    def test_unknown_tool_returns_error_json(self):
        result = json.loads(dispatch_tool("nonexistent_tool", {}))
        assert "error" in result

    def test_validation_error_returned_as_json(self):
        # dispatch with bad args — should NOT raise, returns JSON error
        result = json.loads(dispatch_tool("calculate_bmi", {"weight_kg": -5, "height_cm": 170}))
        assert "error" in result

    def test_dispatch_calculate_bmi_success(self):
        result = json.loads(dispatch_tool("calculate_bmi", {"weight_kg": 70, "height_cm": 175}))
        assert "bmi" in result
        assert result["bmi"] == 22.86

    @patch("app.agent.TOOL_MAP")
    def test_tool_retry_on_exception(self, mock_tool_map):
        """If the tool raises an exception, dispatch retries MAX_RETRIES times."""
        mock_fn = MagicMock(side_effect=RuntimeError("DB timeout"))
        mock_tool_map.__contains__ = MagicMock(return_value=True)
        mock_tool_map.__getitem__  = MagicMock(return_value=mock_fn)

        with patch("app.agent.validate_tool_args", return_value=None), \
             patch("app.agent.time.sleep"):          # skip real sleep in tests
            result = json.loads(dispatch_tool("find_doctor", {"condition": "Hypertension"}))

        assert "error" in result
        assert "failed after" in result["error"]


# ===========================================================================
# 4. Integration-style test — run_agent (LLM mocked)
# ===========================================================================

class TestRunAgent:

    def _make_mock_response(self, content: str):
        """Helper to create a mock LLM response with a plain text answer."""
        mock_msg = MagicMock()
        mock_msg.tool_calls = None
        mock_msg.content    = content

        mock_choice = MagicMock()
        mock_choice.message        = mock_msg
        mock_choice.finish_reason  = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    def _make_mock_tool_call_response(self, tool_name: str, args: dict, call_id="call_1"):
        """Helper to create a mock LLM response that requests a tool call."""
        mock_tool_call           = MagicMock()
        mock_tool_call.id        = call_id
        mock_tool_call.function.name      = tool_name
        mock_tool_call.function.arguments = json.dumps(args)

        mock_msg            = MagicMock()
        mock_msg.tool_calls = [mock_tool_call]
        mock_msg.content    = None

        mock_choice = MagicMock()
        mock_choice.message       = mock_msg
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    @patch("app.agent.call_llm")
    def test_direct_answer_no_tools(self, mock_llm):
        """Agent returns answer directly when LLM doesn't call any tools."""
        mock_llm.return_value = self._make_mock_response("Hypertension is high blood pressure.")

        result = run_agent("What is hypertension?")

        assert result["answer"] == "Hypertension is high blood pressure."
        assert result["tools_used"] == []
        assert result["total_turns"] == 0

    @patch("app.agent.call_llm")
    def test_agent_calls_bmi_tool(self, mock_llm):
        """Agent calls calculate_bmi when user provides weight + height."""
        # Turn 1: LLM requests tool call
        # Turn 2: LLM gives final answer after seeing tool result
        mock_llm.side_effect = [
            self._make_mock_tool_call_response(
                "calculate_bmi", {"weight_kg": 70, "height_cm": 175}
            ),
            self._make_mock_response("Your BMI is 22.86, which is Normal weight.")
        ]

        result = run_agent("I weigh 70 kg and am 175 cm tall. What is my BMI?")

        assert result["total_turns"] == 1
        assert len(result["tools_used"]) == 1
        assert result["tools_used"][0]["tool"] == "calculate_bmi"
        assert "BMI" in result["answer"] or "bmi" in result["answer"].lower()

    @patch("app.agent.call_llm")
    def test_agent_returns_fallback_on_max_turns(self, mock_llm):
        """Agent returns graceful fallback when MAX_TURNS is hit."""
        # Always return a tool call — forces infinite loop until MAX_TURNS
        mock_llm.return_value = self._make_mock_tool_call_response(
            "calculate_bmi", {"weight_kg": 70, "height_cm": 175}
        )

        result = run_agent("Loop forever")

        assert "maximum" in result["answer"].lower()
        assert result["total_turns"] == 10    # MAX_TURNS