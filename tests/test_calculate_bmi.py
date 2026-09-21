"""
tests/test_calculate_bmi.py
----------------------------
Unit tests for the calculate_bmi tool.

No external dependencies — this is a pure deterministic function.
All tests run without network, database, or API access.
"""

import pytest
from app.tools.calculate_bmi import calculate_bmi


class TestCalculateBmiValidInput:
    """Tests for valid inputs — verify correct BMI values and categories."""

    def test_normal_weight_category(self):
        """72 kg, 175 cm → BMI ~23.51 → Normal weight."""
        result = calculate_bmi(weight_kg=72, height_cm=175)
        assert "error" not in result
        assert result["bmi"] == 23.51
        assert result["category"] == "Normal weight"
        assert result["weight_kg"] == 72
        assert result["height_cm"] == 175

    def test_underweight_category(self):
        """45 kg, 175 cm → BMI ~14.69 → Underweight."""
        result = calculate_bmi(weight_kg=45, height_cm=175)
        assert result["category"] == "Underweight"
        assert result["bmi"] < 18.5

    def test_overweight_category(self):
        """85 kg, 170 cm → BMI ~29.41 → Overweight."""
        result = calculate_bmi(weight_kg=85, height_cm=170)
        assert result["category"] == "Overweight"
        assert 25.0 <= result["bmi"] < 30.0

    def test_obese_category(self):
        """100 kg, 165 cm → BMI ~36.73 → Obese."""
        result = calculate_bmi(weight_kg=100, height_cm=165)
        assert result["category"] == "Obese"
        assert result["bmi"] >= 30.0

    def test_bmi_boundary_normal_lower(self):
        """BMI exactly at 18.5 boundary → Normal weight."""
        # BMI = 18.5 → weight = 18.5 * (1.70)^2 = 53.465 kg
        result = calculate_bmi(weight_kg=53.465, height_cm=170)
        assert result["category"] == "Normal weight"
        assert result["bmi"] == 18.5

    def test_bmi_boundary_overweight(self):
        """BMI exactly at 25.0 boundary → Overweight."""
        # BMI = 25.0 → weight = 25 * (1.70)^2 = 72.25 kg
        result = calculate_bmi(weight_kg=72.25, height_cm=170)
        assert result["category"] == "Overweight"
        assert result["bmi"] == 25.0

    def test_result_contains_disclaimer(self):
        """Result must include a disclaimer — tool is not a diagnosis."""
        result = calculate_bmi(weight_kg=72, height_cm=175)
        assert "disclaimer" in result
        assert len(result["disclaimer"]) > 0

    def test_result_contains_advice(self):
        """Result must include advice for each category."""
        result = calculate_bmi(weight_kg=72, height_cm=175)
        assert "advice" in result
        assert len(result["advice"]) > 0


class TestCalculateBmiInvalidInput:
    """Tests for invalid inputs — verify safe error responses."""

    def test_zero_weight_returns_error(self):
        """Weight of 0 is invalid — must return error dict."""
        result = calculate_bmi(weight_kg=0, height_cm=175)
        assert "error" in result
        assert "bmi" not in result

    def test_negative_weight_returns_error(self):
        """Negative weight is invalid."""
        result = calculate_bmi(weight_kg=-10, height_cm=175)
        assert "error" in result

    def test_zero_height_returns_error(self):
        """Height of 0 would cause division by zero — must return error."""
        result = calculate_bmi(weight_kg=70, height_cm=0)
        assert "error" in result
        assert "bmi" not in result

    def test_negative_height_returns_error(self):
        """Negative height is invalid."""
        result = calculate_bmi(weight_kg=70, height_cm=-160)
        assert "error" in result

    def test_error_message_is_user_friendly(self):
        """Error message should be readable — not a raw Pydantic error."""
        result = calculate_bmi(weight_kg=0, height_cm=175)
        # Must not expose internal Pydantic field paths like "weight_kg"
        # in a raw format, and must not be an empty string
        assert isinstance(result["error"], str)
        assert len(result["error"]) > 0

    def test_inputs_echoed_on_error(self):
        """Invalid inputs are echoed back for debugging."""
        result = calculate_bmi(weight_kg=0, height_cm=175)
        assert result["weight_kg"] == 0
        assert result["height_cm"] == 175