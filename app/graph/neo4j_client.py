"""
app/graph/neo4j_client.py
--------------------------
Neo4j driver wrapper providing the four Cypher queries used by find_doctor.

Design decisions:
    - The driver is created lazily (on first use) rather than at module
      import time. This prevents import-time failures when Neo4j is
      not yet available (e.g. during unit tests or CI).
    - Each query method opens and closes a session explicitly.
    - All queries use parameterised Cypher to prevent injection.
    - toLower() is used for case-insensitive matching.

Performance note on toLower():
    Using toLower($param) on a property prevents Neo4j from using a
    standard index on that property. For a prototype with small data
    this is acceptable. A production system would use a full-text index
    (db.index.fulltext) or store a normalised lowercase copy of the
    property alongside the original.

Error handling:
    All exceptions from the Neo4j driver bubble up to the caller
    (find_doctor), which catches them and returns a safe error dict.
    Raw driver error messages are NOT forwarded to the LLM.
"""

from app.config import settings
from app.logger import get_logger

log = get_logger(__name__)


class Neo4jClient:
    """
    Thin wrapper around the Neo4j Python driver.

    Intended as a module-level singleton via get_neo4j_client().
    Not thread-safe to re-initialise concurrently; safe for concurrent
    reads through the driver's built-in connection pool.
    """

    def __init__(self) -> None:
        from neo4j import GraphDatabase

        log.info(
            "Connecting to Neo4j.",
            extra={"uri": settings.neo4j_uri, "user": settings.neo4j_user},
        )
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self) -> None:
        """Close the driver and release all connections."""
        self._driver.close()
        log.info("Neo4j driver closed.")

    def verify_connectivity(self) -> bool:
        """
        Ping Neo4j to confirm the connection is alive.
        Returns True on success, False on failure.
        Used by the /health endpoint.
        """
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as e:
            log.warning("Neo4j connectivity check failed.", extra={"error": str(e)})
            return False

    # ------------------------------------------------------------------
    # Query 1 — find doctors directly by specialization (single hop)
    # Doctor -[:SPECIALIZES_IN]-> Specialization
    # Doctor -[:WORKS_AT]-> Hospital
    # ------------------------------------------------------------------
    def find_doctors_by_specialization(self, specialization: str) -> list[dict]:
        query = """
        MATCH (d:Doctor)-[:SPECIALIZES_IN]->(s:Specialization),
              (d)-[:WORKS_AT]->(h:Hospital)
        WHERE toLower(s.name) = toLower($specialization)
        RETURN d.name              AS doctor,
               s.name              AS specialization,
               h.name              AS hospital,
               h.location          AS location,
               d.experience_years  AS experience_years
        ORDER BY d.experience_years DESC
        """
        with self._driver.session() as session:
            result = session.run(query, specialization=specialization)
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Query 2 — find doctors by medical condition (multi-hop traversal)
    # Condition -[:TREATED_BY]-> Specialization <-[:SPECIALIZES_IN]- Doctor
    # Doctor -[:WORKS_AT]-> Hospital
    # ------------------------------------------------------------------
    def find_doctors_by_condition(self, condition: str) -> list[dict]:
        query = """
        MATCH (c:Condition)-[:TREATED_BY]->(s:Specialization)
              <-[:SPECIALIZES_IN]-(d:Doctor),
              (d)-[:WORKS_AT]->(h:Hospital)
        WHERE toLower(c.name) = toLower($condition)
        RETURN d.name              AS doctor,
               s.name              AS specialization,
               h.name              AS hospital,
               h.location          AS location,
               d.experience_years  AS experience_years
        ORDER BY d.experience_years DESC
        """
        with self._driver.session() as session:
            result = session.run(query, condition=condition)
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Query 3 — find conditions related to a given condition (one hop)
    # Condition -[:RELATED_TO]-> Condition
    # ------------------------------------------------------------------
    def find_related_conditions(self, condition: str) -> list[dict]:
        query = """
        MATCH (c:Condition)-[:RELATED_TO]->(related:Condition)
        WHERE toLower(c.name) = toLower($condition)
        RETURN related.name AS related_condition
        """
        with self._driver.session() as session:
            result = session.run(query, condition=condition)
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Query 4 — get all doctors at a specific hospital (reverse lookup)
    # Doctor -[:WORKS_AT]-> Hospital
    # Doctor -[:SPECIALIZES_IN]-> Specialization
    # ------------------------------------------------------------------
    def find_doctors_at_hospital(self, hospital: str) -> list[dict]:
        query = """
        MATCH (d:Doctor)-[:WORKS_AT]->(h:Hospital),
              (d)-[:SPECIALIZES_IN]->(s:Specialization)
        WHERE toLower(h.name) = toLower($hospital)
        RETURN d.name              AS doctor,
               s.name              AS specialization,
               d.experience_years  AS experience_years
        ORDER BY d.experience_years DESC
        """
        with self._driver.session() as session:
            result = session.run(query, hospital=hospital)
            return [dict(record) for record in result]


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------
_neo4j_client: Neo4jClient | None = None


def get_neo4j_client() -> Neo4jClient:
    """
    Return the shared Neo4jClient instance, creating it on first call.

    Lazy initialisation ensures that import-time failures do not
    occur when Neo4j is unavailable (e.g. during unit tests that
    mock this function).
    """
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    return _neo4j_client


def close_neo4j_client() -> None:
    """Close and release the shared client. Called during app shutdown."""
    global _neo4j_client
    if _neo4j_client is not None:
        _neo4j_client.close()
        _neo4j_client = None