"""
app/tools/find_doctor.py
------------------------
Neo4j-based doctor discovery tool.

Searches the healthcare Knowledge Graph for doctors by:
  - Medical condition  (e.g. "Hypertension", "Migraine")
  - Specialization     (e.g. "Cardiology", "Neurology")

When searching by condition, also returns related conditions
so the LLM can provide richer context to the user.

If both condition and specialization are provided, condition
takes precedence (simpler graph traversal, fewer hops).

Error handling:
    Neo4j driver errors are caught here. The raw error message
    is logged but NOT forwarded to the LLM — only a safe message
    is returned. This prevents internal connection details, stack
    traces, or file paths from leaking into responses.

Limitation:
    Case-insensitive matching via toLower() prevents standard
    Neo4j property index use. Acceptable for a prototype with
    small data volume.
"""

from pydantic import BaseModel, Field, model_validator

from app.graph.neo4j_client import get_neo4j_client
from app.logger import get_logger

log = get_logger(__name__)


# ------------------------------------------------------------------
# Input validation model
# ------------------------------------------------------------------

class FindDoctorInput(BaseModel):
    """
    Validates the arguments the LLM passes to find_doctor.

    At least one of condition or specialization must be provided.
    Both may be provided; condition will take precedence.
    """
    condition: str | None = Field(
        default=None,
        description=(
            "A medical condition or diagnosis, e.g. 'Hypertension', "
            "'Migraine', 'Diabetes'. Use this when the user describes a symptom."
        ),
    )
    specialization: str | None = Field(
        default=None,
        description=(
            "A medical specialization, e.g. 'Cardiology', 'Neurology'. "
            "Use this when the user asks for a specific type of doctor."
        ),
    )

    @model_validator(mode="after")
    def at_least_one_field(self) -> "FindDoctorInput":
        if not self.condition and not self.specialization:
            raise ValueError(
                "Provide either a medical condition or a specialization name."
            )
        return self


# ------------------------------------------------------------------
# Tool function
# ------------------------------------------------------------------

def find_doctor(
        condition: str | None = None,
        specialization: str | None = None,
) -> dict:
    """
    Search the Neo4j Knowledge Graph for doctors.

    Args:
        condition:      Medical condition, e.g. 'Hypertension'.
        specialization: Specialization, e.g. 'Cardiology'.

    Returns:
        dict with keys:
            doctors          — list of matching doctor records
            count            — number of doctors found
            query_type       — "condition" or "specialization"
            query_value      — the value that was searched
            related_conditions — (condition queries only) related conditions
            message          — (when no results) human-readable explanation
        On error, returns:
            error            — sanitized error message
    """
    # Validate inputs
    try:
        validated = FindDoctorInput(condition=condition, specialization=specialization)
    except Exception as e:
        return {"error": _extract_validation_message(e)}

    # Condition takes precedence when both are provided
    effective_condition = validated.condition
    effective_spec = validated.specialization if not effective_condition else None

    client = get_neo4j_client()

    try:
        if effective_condition:
            return _search_by_condition(client, effective_condition)
        else:
            return _search_by_specialization(client, effective_spec)

    except Exception as e:
        # Log full error internally; return safe message externally.
        log.error(
            "Neo4j query failed in find_doctor.",
            extra={"condition": effective_condition, "specialization": effective_spec, "error": str(e)},
        )
        return {
            "error": (
                "Doctor search is temporarily unavailable due to a database issue. "
                "Please try again in a moment."
            )
        }


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _search_by_condition(client, condition: str) -> dict:
    """Execute a condition-based doctor search with related conditions."""
    log.info("Searching doctors by condition.", extra={"condition": condition})

    doctors = client.find_doctors_by_condition(condition)
    related = client.find_related_conditions(condition)
    related_names = [r["related_condition"] for r in related]

    if not doctors:
        return {
            "doctors": [],
            "count": 0,
            "query_type": "condition",
            "query_value": condition,
            "related_conditions": related_names,
            "message": (
                f"No doctors found for condition '{condition}' in the Knowledge Graph. "
                "The condition may not yet be represented in the graph."
            ),
        }

    log.info(
        "Doctor search by condition complete.",
        extra={"condition": condition, "count": len(doctors)},
    )
    return {
        "doctors": doctors,
        "count": len(doctors),
        "query_type": "condition",
        "query_value": condition,
        "related_conditions": related_names,
    }


def _search_by_specialization(client, specialization: str) -> dict:
    """Execute a specialization-based doctor search."""
    log.info("Searching doctors by specialization.", extra={"specialization": specialization})

    doctors = client.find_doctors_by_specialization(specialization)

    if not doctors:
        return {
            "doctors": [],
            "count": 0,
            "query_type": "specialization",
            "query_value": specialization,
            "message": (
                f"No doctors found for specialization '{specialization}' in the Knowledge Graph."
            ),
        }

    log.info(
        "Doctor search by specialization complete.",
        extra={"specialization": specialization, "count": len(doctors)},
    )
    return {
        "doctors": doctors,
        "count": len(doctors),
        "query_type": "specialization",
        "query_value": specialization,
    }


def _extract_validation_message(exc: Exception) -> str:
    """Extract a clean user-facing message from a Pydantic ValidationError."""
    try:
        errors = exc.errors()
        return "; ".join(e.get("msg", "Invalid value") for e in errors)
    except Exception:
        return "Invalid arguments: provide either a condition or a specialization."