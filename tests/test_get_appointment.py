"""
tests/test_get_appointment.py
------------------------------
Unit tests for the get_appointment tool.

Uses a temporary SQLite database via monkeypatching the DB_PATH.
"""

import os
import sqlite3
import tempfile

import pytest

import app.tools.get_appointment as appt_module
from app.tools.get_appointment import get_appointment, init_appointments_db


@pytest.fixture(autouse=True)
def use_temp_db(tmp_path, monkeypatch):
    """
    Redirect all DB operations to a temporary file for each test.
    The temp file is deleted automatically after each test.
    """
    db_file = str(tmp_path / "test_appointments.db")
    monkeypatch.setattr(appt_module, "DB_PATH", db_file)
    init_appointments_db()
    yield db_file


class TestGetAppointmentSuccess:
    """Tests for valid appointment IDs that exist in the database."""

    def test_found_appointment_has_expected_keys(self):
        """A successful lookup returns all expected fields."""
        result = get_appointment("AP101")
        expected_keys = {
            "found", "appointment_id", "patient_name",
            "doctor_name", "specialization", "date", "time", "status", "notes",
        }
        assert set(result.keys()) == expected_keys

    def test_found_is_true(self):
        result = get_appointment("AP101")
        assert result["found"] is True

    def test_correct_appointment_data(self):
        result = get_appointment("AP101")
        assert result["appointment_id"] == "AP101"
        assert result["patient_name"] == "Rahul Mehta"
        assert result["doctor_name"] == "Dr. Sharma"
        assert result["specialization"] == "Cardiology"
        assert result["status"] == "confirmed"

    def test_case_insensitive_lookup(self):
        """Lowercase appointment ID should be normalised and found."""
        result = get_appointment("ap101")
        assert result["found"] is True
        assert result["appointment_id"] == "AP101"

    def test_mixed_case_lookup(self):
        result = get_appointment("Ap101")
        assert result["found"] is True

    def test_cancelled_appointment_returned(self):
        """Cancelled appointments are returned, not filtered out."""
        result = get_appointment("AP103")
        assert result["found"] is True
        assert result["status"] == "cancelled"

    def test_pending_appointment_returned(self):
        result = get_appointment("AP102")
        assert result["found"] is True
        assert result["status"] == "pending"

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace in ID is stripped."""
        result = get_appointment("  AP101  ")
        assert result["found"] is True


class TestGetAppointmentNotFound:
    """Tests for IDs that do not exist in the database."""

    def test_unknown_id_returns_found_false(self):
        result = get_appointment("AP999")
        assert result["found"] is False

    def test_unknown_id_contains_message(self):
        result = get_appointment("AP999")
        assert "message" in result
        assert "AP999" in result["message"]

    def test_unknown_id_no_exception_raised(self):
        """Missing records must not raise exceptions."""
        # This should not throw
        result = get_appointment("AP000")
        assert "error" not in result or result.get("found") is False


class TestGetAppointmentValidation:
    """Tests for invalid appointment ID formats."""

    def test_invalid_format_returns_error(self):
        """IDs not matching AP+digits format return a validation error."""
        result = get_appointment("INVALID")
        assert "error" in result

    def test_empty_string_returns_error(self):
        result = get_appointment("")
        assert "error" in result

    def test_numeric_only_returns_error(self):
        """'101' without 'AP' prefix is invalid."""
        result = get_appointment("101")
        assert "error" in result

    def test_error_does_not_expose_internals(self):
        """Error message must not contain SQLite internals or file paths."""
        result = get_appointment("INVALID_FORMAT")
        if "error" in result:
            error_msg = result["error"].lower()
            assert "sqlite" not in error_msg
            assert "/tmp" not in error_msg
            assert "traceback" not in error_msg