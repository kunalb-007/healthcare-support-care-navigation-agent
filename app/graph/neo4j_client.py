import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


class Neo4jClient:
    """
    Singleton-style client for Neo4j.
    Provides four Cypher query methods used by the find_doctor tool.
    """

    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # ------------------------------------------------------------------
    # Query 1 — find doctors directly by specialization name (single hop)
    # ------------------------------------------------------------------
    def find_doctors_by_specialization(self, specialization: str) -> list[dict]:
        query = """
        MATCH (d:Doctor)-[:SPECIALIZES_IN]->(s:Specialization),
              (d)-[:WORKS_AT]->(h:Hospital)
        WHERE toLower(s.name) = toLower($specialization)
        RETURN d.name AS doctor,
               s.name AS specialization,
               h.name AS hospital,
               h.location AS location,
               d.experience_years AS experience_years
        ORDER BY d.experience_years DESC
        """
        with self.driver.session() as session:
            result = session.run(query, specialization=specialization)
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Query 2 — find doctors by medical condition (multi-hop traversal)
    # Condition -[:TREATED_BY]-> Specialization <-[:SPECIALIZES_IN]- Doctor
    #                                                Doctor -[:WORKS_AT]-> Hospital
    # ------------------------------------------------------------------
    def find_doctors_by_condition(self, condition: str) -> list[dict]:
        query = """
        MATCH (c:Condition)-[:TREATED_BY]->(s:Specialization)<-[:SPECIALIZES_IN]-(d:Doctor),
              (d)-[:WORKS_AT]->(h:Hospital)
        WHERE toLower(c.name) = toLower($condition)
        RETURN d.name            AS doctor,
               s.name            AS specialization,
               h.name            AS hospital,
               h.location        AS location,
               d.experience_years AS experience_years
        ORDER BY d.experience_years DESC
        """
        with self.driver.session() as session:
            result = session.run(query, condition=condition)
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Query 3 — find conditions related to a given condition
    # ------------------------------------------------------------------
    def find_related_conditions(self, condition: str) -> list[dict]:
        query = """
        MATCH (c:Condition)-[:RELATED_TO]->(related:Condition)
        WHERE toLower(c.name) = toLower($condition)
        RETURN related.name AS related_condition
        """
        with self.driver.session() as session:
            result = session.run(query, condition=condition)
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Query 4 — get all doctors at a specific hospital (reverse lookup)
    # ------------------------------------------------------------------
    def find_doctors_at_hospital(self, hospital: str) -> list[dict]:
        query = """
        MATCH (d:Doctor)-[:WORKS_AT]->(h:Hospital),
              (d)-[:SPECIALIZES_IN]->(s:Specialization)
        WHERE toLower(h.name) = toLower($hospital)
        RETURN d.name            AS doctor,
               s.name            AS specialization,
               d.experience_years AS experience_years
        ORDER BY d.experience_years DESC
        """
        with self.driver.session() as session:
            result = session.run(query, hospital=hospital)
            return [dict(record) for record in result]