from app.graph.neo4j_client import Neo4jClient

# Module-level client — one connection reused across all requests.
neo4j_client = Neo4jClient()


def find_doctor(condition: str = None, specialization: str = None) -> dict:
    """
    Search the Neo4j Knowledge Graph for doctors.

    Use `condition` when the user describes a medical problem
                         (e.g. 'Hypertension', 'Migraine').
    Use `specialization` when the user asks for a type of specialist
                         (e.g. 'Cardiology', 'Neurology').

    It also fetches related conditions when searching by condition,
    giving the LLM additional context to include in its answer.
    """
    # Guard: at least one parameter must be provided
    if not condition and not specialization:
        return {
            "error": "Provide either a medical condition or a specialization name."
        }

    # Guard: both parameters provided — prefer condition
    if condition and specialization:
        specialization = None  # condition takes precedence; simpler graph traversal

    try:
        if condition:
            doctors = neo4j_client.find_doctors_by_condition(condition)

            # Bonus: also fetch related conditions for richer LLM context
            related = neo4j_client.find_related_conditions(condition)
            related_names = [r["related_condition"] for r in related]

            if not doctors:
                return {
                    "doctors": [],
                    "count": 0,
                    "query_type": "condition",
                    "query_value": condition,
                    "related_conditions": related_names,
                    "message": f"No doctors found for condition '{condition}'. "
                               "It may not be in the Knowledge Graph yet."
                }

            return {
                "doctors": doctors,
                "count": len(doctors),
                "query_type": "condition",
                "query_value": condition,
                "related_conditions": related_names
            }

        else:
            doctors = neo4j_client.find_doctors_by_specialization(specialization)

            if not doctors:
                return {
                    "doctors": [],
                    "count": 0,
                    "query_type": "specialization",
                    "query_value": specialization,
                    "message": f"No doctors found for specialization '{specialization}'."
                }

            return {
                "doctors": doctors,
                "count": len(doctors),
                "query_type": "specialization",
                "query_value": specialization
            }

    except Exception as e:
        return {"error": f"Knowledge Graph query failed: {str(e)}"}