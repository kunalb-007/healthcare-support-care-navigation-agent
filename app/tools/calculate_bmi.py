def calculate_bmi(weight_kg: float, height_cm: float) -> dict:
    """
    Calculate BMI given weight in kilograms and height in centimetres.
    Formula : BMI = weight_kg / (height_m ** 2)
    Returns BMI value rounded to 2 decimal places and the WHO category.
    """
    if weight_kg <= 0 or height_cm <= 0:
        return {
            "error": "Weight and height must be positive values.",
            "weight_kg": weight_kg,
            "height_cm": height_cm
        }

    try:
        height_m = height_cm / 100.0
        bmi = round(weight_kg / (height_m ** 2), 2)

        if bmi < 18.5:
            category = "Underweight"
            advice = "Consider consulting a General Medicine doctor about healthy weight gain."
        elif bmi < 25.0:
            category = "Normal weight"
            advice = "You are within the healthy weight range. Maintain a balanced diet and regular exercise."
        elif bmi < 30.0:
            category = "Overweight"
            advice = "Consider consulting a General Medicine doctor about a healthy weight management plan."
        else:
            category = "Obese"
            advice = "It is recommended to consult a doctor for personalised weight management guidance."

        return {
            "bmi": bmi,
            "category": category,
            "advice": advice,
            "weight_kg": weight_kg,
            "height_cm": height_cm
        }

    except Exception as e:
        return {"error": f"BMI calculation failed: {str(e)}"}