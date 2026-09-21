"""
Deterministic BMI calculation tool.

Validates positive height and weight inputs and returns
a BMI classification. This is not a medical diagnostic tool.
"""

from pydantic import BaseModel, Field, field_validator


# ------------------------------------------------------------------
# Input validation model
# ------------------------------------------------------------------

class BMIInput(BaseModel):
    """
    Validates the arguments the LLM passes to calculate_bmi.

    Pydantic enforces types and value constraints before the
    calculation runs, preventing silent errors from bad LLM outputs.
    """
    weight_kg: float = Field(
        ...,
        gt=0,
        description="Body weight in kilograms. Must be a positive number.",
    )
    height_cm: float = Field(
        ...,
        gt=0,
        description="Height in centimetres. Must be a positive number.",
    )

    @field_validator("weight_kg", "height_cm", mode="before")
    @classmethod
    def must_be_finite(cls, v: float) -> float:
        """Reject infinity and NaN, which pass the gt=0 check."""
        import math
        if not math.isfinite(float(v)):
            raise ValueError("Value must be a finite number.")
        return v


# ------------------------------------------------------------------
# Tool function
# ------------------------------------------------------------------

def calculate_bmi(weight_kg: float, height_cm: float) -> dict:
    """
    Calculate BMI and return the WHO weight classification.

    Formula: BMI = weight_kg / (height_m ** 2)

    Args:
        weight_kg: Body weight in kilograms (must be > 0).
        height_cm: Height in centimetres (must be > 0).

    Returns:
        dict with keys:
            bmi        — rounded to 2 decimal places
            category   — WHO classification string
            advice     — brief guidance (not a medical diagnosis)
            weight_kg  — echoed input
            height_cm  — echoed input
        On validation failure, returns:
            error      — human-readable error message
    """
    # Validate inputs using the Pydantic model
    try:
        validated = BMIInput(weight_kg=weight_kg, height_cm=height_cm)
    except Exception as e:
        # Return a structured error dict; the LLM surfaces this to the user.
        # We do NOT expose the raw Pydantic error object.
        return {
            "error": f"Invalid inputs: {_extract_validation_message(e)}",
            "weight_kg": weight_kg,
            "height_cm": height_cm,
        }

    height_m = validated.height_cm / 100.0
    bmi = round(validated.weight_kg / (height_m ** 2), 2)

    if bmi < 18.5:
        category = "Underweight"
        advice = (
            "Your BMI suggests you may be underweight. "
            "Consider consulting a General Medicine doctor about healthy weight gain."
        )
    elif bmi < 25.0:
        category = "Normal weight"
        advice = (
            "Your BMI is within the healthy range. "
            "Maintain a balanced diet and regular physical activity."
        )
    elif bmi < 30.0:
        category = "Overweight"
        advice = (
            "Your BMI suggests you may be overweight. "
            "Consider consulting a General Medicine doctor about weight management."
        )
    else:
        category = "Obese"
        advice = (
            "Your BMI suggests obesity. "
            "It is recommended to consult a doctor for personalised guidance."
        )

    return {
        "bmi": bmi,
        "category": category,
        "advice": advice,
        "weight_kg": validated.weight_kg,
        "height_cm": validated.height_cm,
        "disclaimer": (
            "This BMI result is for general informational purposes only "
            "and does not constitute medical advice or diagnosis."
        ),
    }


def _extract_validation_message(exc: Exception) -> str:
    """
    Extract a clean user-facing message from a Pydantic ValidationError.
    Avoids exposing raw internal error representations.
    """
    try:
        errors = exc.errors()
        messages = [e.get("msg", "Invalid value") for e in errors]
        return "; ".join(messages)
    except Exception:
        return "weight and height must be positive numbers."