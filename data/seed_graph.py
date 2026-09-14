"""
Seed the Neo4j Knowledge Graph with synthetic healthcare data.

Run once after starting Neo4j:
    python data/seed_graph.py

Clears all existing nodes and relationships before re-seeding,
so it is safe to run multiple times.
"""

import os
import sys
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Allow running from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER",     "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


def seed(tx):
    # ------------------------------------------------------------------
    # 0. Clear all existing data
    # ------------------------------------------------------------------
    tx.run("MATCH (n) DETACH DELETE n")
    print("  Cleared existing graph data.")

    # ------------------------------------------------------------------
    # 1. Specializations
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
    # 2. Hospitals
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
    # 3. Doctors
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
        tx.run("CREATE (:Doctor {name: $name, experience_years: $experience_years})", **d)
    print(f"  Created {len(doctors)} Doctor nodes.")

    # ------------------------------------------------------------------
    # 4. Conditions
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
    # 5. Doctor → Specialization  [:SPECIALIZES_IN]
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
        tx.run("""
            MATCH (d:Doctor {name: $doctor}),
                  (s:Specialization {name: $spec})
            CREATE (d)-[:SPECIALIZES_IN]->(s)
        """, doctor=doctor, spec=spec)
    print(f"  Created {len(doctor_specializations)} SPECIALIZES_IN relationships.")

    # ------------------------------------------------------------------
    # 6. Doctor → Hospital  [:WORKS_AT]
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
        tx.run("""
            MATCH (d:Doctor {name: $doctor}),
                  (h:Hospital {name: $hospital})
            CREATE (d)-[:WORKS_AT]->(h)
        """, doctor=doctor, hospital=hospital)
    print(f"  Created {len(doctor_hospitals)} WORKS_AT relationships.")

    # ------------------------------------------------------------------
    # 7. Condition → Specialization  [:TREATED_BY]
    # ------------------------------------------------------------------
    condition_treatments = [
        ("Hypertension",    "Cardiology"),
        ("Heart Disease",   "Cardiology"),
        ("Migraine",        "Neurology"),
        ("Stroke",          "Neurology"),
        ("Knee Pain",       "Orthopedics"),
        ("Arthritis",       "Orthopedics"),
        ("Diabetes",        "General Medicine"),
        ("Thyroid Disorder","Endocrinology"),
    ]
    for condition, spec in condition_treatments:
        tx.run("""
            MATCH (c:Condition {name: $condition}),
                  (s:Specialization {name: $spec})
            CREATE (c)-[:TREATED_BY]->(s)
        """, condition=condition, spec=spec)
    print(f"  Created {len(condition_treatments)} TREATED_BY relationships.")

    # ------------------------------------------------------------------
    # 8. Condition → Condition  [:RELATED_TO]
    # ------------------------------------------------------------------
    related_conditions = [
        ("Hypertension", "Heart Disease"),
        ("Diabetes",     "Heart Disease"),
        ("Diabetes",     "Hypertension"),
        ("Migraine",     "Stroke"),
        ("Knee Pain",    "Arthritis"),
        ("Thyroid Disorder", "Diabetes"),
    ]
    for c1, c2 in related_conditions:
        tx.run("""
            MATCH (c1:Condition {name: $c1}),
                  (c2:Condition {name: $c2})
            CREATE (c1)-[:RELATED_TO]->(c2)
        """, c1=c1, c2=c2)
    print(f"  Created {len(related_conditions)} RELATED_TO relationships.")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\nSeeding Neo4j Knowledge Graph...")
    try:
        with driver.session() as session:
            session.execute_write(seed)
        print("\nGraph seeded successfully.\n")
    except Exception as e:
        print(f"\nSeeding failed: {e}")
    finally:
        driver.close()