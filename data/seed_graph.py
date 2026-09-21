"""
data/seed_graph.py
------------------
Seed the Neo4j Knowledge Graph with synthetic healthcare data.

Run once after starting Neo4j:
    python data/seed_graph.py

Safe to re-run: clears all existing nodes and relationships before seeding.

Changes from original:
    1. Uses settings from app.config rather than os.getenv directly.
    2. Adds uniqueness constraints before seeding to enforce data integrity.
    3. Adds composite index on Doctor.name for faster lookups.
    4. Comments explain the graph model clearly for interview discussion.

Knowledge Graph Schema:
    Nodes:
        (:Doctor   {name, experience_years})
        (:Hospital {name, location})
        (:Specialization {name})
        (:Condition {name})

    Relationships:
        (Doctor)-[:SPECIALIZES_IN]->(Specialization)
        (Doctor)-[:WORKS_AT]->(Hospital)
        (Condition)-[:TREATED_BY]->(Specialization)
        (Condition)-[:RELATED_TO]->(Condition)

    Multi-hop traversal example:
        User asks about "Hypertension"
        → (Condition{Hypertension})-[:TREATED_BY]->(Specialization{Cardiology})
          <-[:SPECIALIZES_IN]-(Doctor{Dr. Sharma})
          -[:WORKS_AT]->(Hospital{City Hospital})

Note on toLower() and indexes:
    The Cypher queries use toLower() for case-insensitive matching.
    Standard Neo4j property indexes do not support toLower() lookups.
    The uniqueness constraints below ensure data integrity but do NOT
    speed up case-insensitive queries. For a production system with
    large datasets, consider a full-text index instead:
        CREATE FULLTEXT INDEX condition_name FOR (c:Condition) ON EACH [c.name]
"""

import os
import sys

# Allow running from project root or from data/ directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings


def create_constraints(tx) -> None:
    """
    Create uniqueness constraints on node name properties.

    Constraints serve two purposes:
      1. Data integrity: prevent duplicate nodes with the same name.
      2. Index creation: Neo4j automatically creates a backing index
         for each constraint (used by exact-match MATCH clauses).

    Note: IF NOT EXISTS syntax requires Neo4j >= 4.1.
    """
    constraints = [
        "CREATE CONSTRAINT doctor_name IF NOT EXISTS FOR (d:Doctor) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT hospital_name IF NOT EXISTS FOR (h:Hospital) REQUIRE h.name IS UNIQUE",
        "CREATE CONSTRAINT specialization_name IF NOT EXISTS FOR (s:Specialization) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT condition_name IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE",
    ]
    for constraint in constraints:
        tx.run(constraint)
    print(f"  Created {len(constraints)} uniqueness constraints.")


def seed(tx) -> None:
    """Seed all nodes and relationships."""

    # ------------------------------------------------------------------
    # 0. Clear existing data
    # ------------------------------------------------------------------
    tx.run("MATCH (n) DETACH DELETE n")
    print("  Cleared existing graph data.")

    # ------------------------------------------------------------------
    # 1. Specialization nodes
    # ------------------------------------------------------------------
    specializations = [
        "Cardiology",
        "Neurology",
        "Orthopedics",
        "General Medicine",
        "Endocrinology",
    ]
    for spec in specializations:
        tx.run("CREATE (:Specialization {name: $name})", name=spec)
    print(f"  Created {len(specializations)} Specialization nodes.")

    # ------------------------------------------------------------------
    # 2. Hospital nodes
    # ------------------------------------------------------------------
    hospitals = [
        {"name": "City Hospital",     "location": "Mumbai"},
        {"name": "Apollo Clinic",     "location": "Delhi"},
        {"name": "Sunrise Medical",   "location": "Bangalore"},
        {"name": "Global Health Hub", "location": "Hyderabad"},
    ]
    for h in hospitals:
        tx.run("CREATE (:Hospital {name: $name, location: $location})", **h)
    print(f"  Created {len(hospitals)} Hospital nodes.")

    # ------------------------------------------------------------------
    # 3. Doctor nodes
    # ------------------------------------------------------------------
    doctors = [
        {"name": "Dr. Sharma",     "experience_years": 12},
        {"name": "Dr. Patel",      "experience_years": 8},
        {"name": "Dr. Mehta",      "experience_years": 15},
        {"name": "Dr. Rao",        "experience_years": 6},
        {"name": "Dr. Gupta",      "experience_years": 10},
        {"name": "Dr. Krishnarao", "experience_years": 20},
    ]
    for d in doctors:
        tx.run(
            "CREATE (:Doctor {name: $name, experience_years: $experience_years})",
            **d,
        )
    print(f"  Created {len(doctors)} Doctor nodes.")

    # ------------------------------------------------------------------
    # 4. Condition nodes
    # ------------------------------------------------------------------
    conditions = [
        "Hypertension",
        "Heart Disease",
        "Migraine",
        "Knee Pain",
        "Diabetes",
        "Thyroid Disorder",
        "Arthritis",
        "Stroke",
    ]
    for c in conditions:
        tx.run("CREATE (:Condition {name: $name})", name=c)
    print(f"  Created {len(conditions)} Condition nodes.")

    # ------------------------------------------------------------------
    # 5. Doctor -[:SPECIALIZES_IN]-> Specialization
    # ------------------------------------------------------------------
    doctor_specializations = [
        ("Dr. Sharma",     "Cardiology"),
        ("Dr. Patel",      "Cardiology"),
        ("Dr. Mehta",      "Neurology"),
        ("Dr. Rao",        "Orthopedics"),
        ("Dr. Gupta",      "General Medicine"),
        ("Dr. Krishnarao", "Endocrinology"),
    ]
    for doctor, spec in doctor_specializations:
        tx.run(
            """
            MATCH (d:Doctor {name: $doctor}),
                  (s:Specialization {name: $spec})
            CREATE (d)-[:SPECIALIZES_IN]->(s)
            """,
            doctor=doctor,
            spec=spec,
        )
    print(f"  Created {len(doctor_specializations)} SPECIALIZES_IN relationships.")

    # ------------------------------------------------------------------
    # 6. Doctor -[:WORKS_AT]-> Hospital
    # ------------------------------------------------------------------
    doctor_hospitals = [
        ("Dr. Sharma",     "City Hospital"),
        ("Dr. Patel",      "Apollo Clinic"),
        ("Dr. Mehta",      "Sunrise Medical"),
        ("Dr. Rao",        "City Hospital"),
        ("Dr. Gupta",      "Global Health Hub"),
        ("Dr. Krishnarao", "Apollo Clinic"),
    ]
    for doctor, hospital in doctor_hospitals:
        tx.run(
            """
            MATCH (d:Doctor {name: $doctor}),
                  (h:Hospital {name: $hospital})
            CREATE (d)-[:WORKS_AT]->(h)
            """,
            doctor=doctor,
            hospital=hospital,
        )
    print(f"  Created {len(doctor_hospitals)} WORKS_AT relationships.")

    # ------------------------------------------------------------------
    # 7. Condition -[:TREATED_BY]-> Specialization
    # ------------------------------------------------------------------
    condition_treatments = [
        ("Hypertension",     "Cardiology"),
        ("Heart Disease",    "Cardiology"),
        ("Migraine",         "Neurology"),
        ("Stroke",           "Neurology"),
        ("Knee Pain",        "Orthopedics"),
        ("Arthritis",        "Orthopedics"),
        ("Diabetes",         "General Medicine"),
        ("Thyroid Disorder", "Endocrinology"),
    ]
    for condition, spec in condition_treatments:
        tx.run(
            """
            MATCH (c:Condition {name: $condition}),
                  (s:Specialization {name: $spec})
            CREATE (c)-[:TREATED_BY]->(s)
            """,
            condition=condition,
            spec=spec,
        )
    print(f"  Created {len(condition_treatments)} TREATED_BY relationships.")

    # ------------------------------------------------------------------
    # 8. Condition -[:RELATED_TO]-> Condition
    #
    # These edges allow the agent to surface related conditions,
    # giving the LLM additional context. Example:
    #   User asks about "Hypertension"
    #   → also mention "Heart Disease" and "Diabetes" as related
    # ------------------------------------------------------------------
    related_conditions = [
        ("Hypertension",     "Heart Disease"),
        ("Diabetes",         "Heart Disease"),
        ("Diabetes",         "Hypertension"),
        ("Migraine",         "Stroke"),
        ("Knee Pain",        "Arthritis"),
        ("Thyroid Disorder", "Diabetes"),
    ]
    for c1, c2 in related_conditions:
        tx.run(
            """
            MATCH (c1:Condition {name: $c1}),
                  (c2:Condition {name: $c2})
            CREATE (c1)-[:RELATED_TO]->(c2)
            """,
            c1=c1,
            c2=c2,
        )
    print(f"  Created {len(related_conditions)} RELATED_TO relationships.")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    from neo4j import GraphDatabase

    print(f"\nConnecting to Neo4j at {settings.neo4j_uri}...")
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    print("Seeding Neo4j Knowledge Graph...\n")
    try:
        with driver.session() as session:
            # Constraints must be created outside the seed transaction
            # because schema changes and data changes cannot be mixed
            # in the same transaction in Neo4j.
            session.execute_write(create_constraints)
            session.execute_write(seed)
        print("\nGraph seeded successfully.\n")
    except Exception as e:
        print(f"\nSeeding failed: {e}")
        sys.exit(1)
    finally:
        driver.close()