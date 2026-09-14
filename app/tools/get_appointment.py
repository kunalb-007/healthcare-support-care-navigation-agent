import sqlite3

DB_PATH = "appointments.db"


def init_appointments_db() -> None:
    """
    Create the appointments table and seed it with sample data
    if it does not already exist.
    Called once at FastAPI startup.
    """
    conn   = sqlite3.connect(DB_PATH)
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

    # Sample appointments — INSERT OR IGNORE so re-running startup is safe
    sample_appointments = [
        ("AP101", "Rahul Mehta",   "Dr. Sharma", "Cardiology",      "2026-09-20", "10:00 AM", "confirmed",  "Follow-up for blood pressure check"),
        ("AP102", "Priya Singh",   "Dr. Patel",  "Cardiology",      "2026-09-22", "11:30 AM", "pending",    "Initial consultation"),
        ("AP103", "Arun Kumar",    "Dr. Mehta",  "Neurology",       "2026-09-18", "09:00 AM", "cancelled",  "Cancelled by patient"),
        ("AP104", "Sunita Verma",  "Dr. Rao",    "Orthopedics",     "2026-09-25", "02:00 PM", "confirmed",  "Knee pain evaluation"),
        ("AP105", "Vikram Nair",   "Dr. Sharma", "Cardiology",      "2026-09-28", "03:30 PM", "confirmed",  "Routine cardiac checkup"),
        ("AP106", "Anjali Desai",  "Dr. Mehta",  "Neurology",       "2026-09-30", "10:00 AM", "pending",    "Migraine assessment"),
    ]

    cursor.executemany(
        "INSERT OR IGNORE INTO appointments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        sample_appointments
    )

    conn.commit()
    conn.close()


def get_appointment(appointment_id: str) -> dict:
    """
    Look up an appointment by its ID.

    Returns full appointment details: patient name, doctor, specialization,
    date, time, status, and any notes.

    Returns a 'found: False' payload when the ID does not exist rather than
    raising an exception, so the LLM can report 'not found' gracefully.
    """
    try:
        conn             = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row          # access columns by name
        cursor           = conn.cursor()

        cursor.execute(
            "SELECT * FROM appointments WHERE appointment_id = ?",
            (appointment_id.upper().strip(),)  # normalise casing
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return {
                "found": False,
                "appointment_id": appointment_id,
                "message":  f"No appointment found with ID '{appointment_id}'. "
                            "Please check the ID and try again."
            }

        return {
            "found":          True,
            "appointment_id": row["appointment_id"],
            "patient_name":   row["patient_name"],
            "doctor_name":    row["doctor_name"],
            "specialization": row["specialization"],
            "date":           row["date"],
            "time":           row["time"],
            "status":         row["status"],
            "notes":          row["notes"]
        }

    except Exception as e:
        return {"error": f"Appointment lookup failed: {str(e)}"}