"""
SQLite-backed appointment lookup tool.

Returns structured success or error responses and sanitizes
internal database errors before returning them to the caller.
"""

import re
import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from app.logger import get_logger

log = get_logger(__name__)

DB_PATH = "appointments.db"

# ------------------------------------------------------------------
# Appointment ID validation: must match AP + digits, e.g. AP101
# ------------------------------------------------------------------
_APPOINTMENT_ID_PATTERN = re.compile(r"^AP\d+$", re.IGNORECASE)


class AppointmentInput(BaseModel):
    """Validates the appointment_id argument before querying the DB."""

    appointment_id: str = Field(
        ...,
        min_length=2,
        max_length=20,
        description="Appointment ID in the format AP followed by digits, e.g. AP101.",
    )

    @field_validator("appointment_id", mode="before")
    @classmethod
    def normalise_and_validate(cls, v: str) -> str:
        """Strip whitespace, uppercase, and validate format."""
        if not isinstance(v, str):
            raise ValueError("Appointment ID must be a string.")
        v = v.strip().upper()
        if not _APPOINTMENT_ID_PATTERN.match(v):
            raise ValueError(
                f"'{v}' is not a valid appointment ID. "
                "Expected format: AP followed by digits, e.g. AP101."
            )
        return v


# ------------------------------------------------------------------
# Database initialisation
# ------------------------------------------------------------------

def init_appointments_db() -> None:
    """
    Create the appointments table and seed it with sample data
    if it does not already exist.

    Called once at FastAPI startup via the lifespan context manager.
    Safe to call multiple times — INSERT OR IGNORE prevents duplicates.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS appointments (
                                                               appointment_id TEXT PRIMARY KEY,
                                                               patient_name   TEXT NOT NULL,
                                                               doctor_name    TEXT NOT NULL,
                                                               specialization TEXT NOT NULL,
                                                               date           TEXT NOT NULL,
                                                               time           TEXT NOT NULL,
                                                               status         TEXT NOT NULL,
                                                               notes          TEXT
                   )
                   """)

    sample_appointments = [
        ("AP101", "Rahul Mehta",  "Dr. Sharma", "Cardiology",    "2026-09-20", "10:00 AM", "confirmed", "Follow-up for blood pressure check"),
        ("AP102", "Priya Singh",  "Dr. Patel",  "Cardiology",    "2026-09-22", "11:30 AM", "pending",   "Initial consultation"),
        ("AP103", "Arun Kumar",   "Dr. Mehta",  "Neurology",     "2026-09-18", "09:00 AM", "cancelled", "Cancelled by patient"),
        ("AP104", "Sunita Verma", "Dr. Rao",    "Orthopedics",   "2026-09-25", "02:00 PM", "confirmed", "Knee pain evaluation"),
        ("AP105", "Vikram Nair",  "Dr. Sharma", "Cardiology",    "2026-09-28", "03:30 PM", "confirmed", "Routine cardiac checkup"),
        ("AP106", "Anjali Desai", "Dr. Mehta",  "Neurology",     "2026-09-30", "10:00 AM", "pending",   "Migraine assessment"),
    ]

    cursor.executemany(
        "INSERT OR IGNORE INTO appointments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        sample_appointments,
    )

    conn.commit()
    conn.close()
    log.info("Appointments database initialised.", extra={"db_path": DB_PATH})


# ------------------------------------------------------------------
# Tool function
# ------------------------------------------------------------------

def get_appointment(appointment_id: str) -> dict:
    """
    Look up an appointment by its unique ID.

    Args:
        appointment_id: Appointment ID string, e.g. 'AP101'.
                        Case-insensitive; leading/trailing whitespace stripped.

    Returns:
        dict with full appointment details on success, or
        dict with 'found: False' when the record does not exist, or
        dict with 'error' key when a database error occurs.
    """
    # Validate the input before touching the database
    try:
        validated = AppointmentInput(appointment_id=appointment_id)
    except Exception as e:
        return {
            "error": _extract_validation_message(e),
            "appointment_id": appointment_id,
        }

    normalised_id = validated.appointment_id  # already uppercased

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM appointments WHERE appointment_id = ?",
            (normalised_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            log.info(
                "Appointment not found.",
                extra={"appointment_id": normalised_id},
            )
            return {
                "found": False,
                "appointment_id": normalised_id,
                "message": (
                    f"No appointment found with ID '{normalised_id}'. "
                    "Please check the ID and try again."
                ),
            }

        log.info(
            "Appointment retrieved.",
            extra={"appointment_id": normalised_id, "status": row["status"]},
        )
        return {
            "found": True,
            "appointment_id": row["appointment_id"],
            "patient_name":   row["patient_name"],
            "doctor_name":    row["doctor_name"],
            "specialization": row["specialization"],
            "date":           row["date"],
            "time":           row["time"],
            "status":         row["status"],
            "notes":          row["notes"],
        }

    except sqlite3.Error as e:
        # Log the full error for debugging; return a safe message to caller.
        log.error(
            "SQLite error during appointment lookup.",
            extra={"appointment_id": normalised_id, "error": str(e)},
        )
        return {
            "error": "Appointment lookup is temporarily unavailable. Please try again.",
            "appointment_id": normalised_id,
        }


def _extract_validation_message(exc: Exception) -> str:
    """Extract a clean message from a Pydantic ValidationError."""
    try:
        errors = exc.errors()
        return "; ".join(e.get("msg", "Invalid value") for e in errors)
    except Exception:
        return "Invalid appointment ID format."