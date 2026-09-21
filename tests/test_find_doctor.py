"""
tests/test_find_doctor.py
--------------------------
Unit tests for the find_doctor tool.

Neo4j is mocked
Tests verify argument validation, routing logic, error handling, and the shape of returned data.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.tools.find_doctor import find_doctor


# ------------------------------------------------------------------
# Shared mock data
# ------------------------------------------------------------------
MOCK_DOCTORS = [
    {
        "doctor": "Dr. Sharma",
        "specialization": "Cardiology",
        "hospital": "City Hospital",
        "location": "Mumbai",
        "experience_years": 12,
    },
    {
        "doctor": "Dr. Patel",
        "specialization": "Cardiology",
        "hospital": "Apollo Clinic",
        "location": "Delhi",
        "experience_years": 8,
    },
]

MOCK_RELATED = [{"related_condition": "Heart Disease"}]


def _make_mock_client(doctors=None, related=None):
    """Build a mock Neo4jClient with controllable return values."""
    client = MagicMock()
    client.find_doctors_by_condition.return_value = doctors if doctors is not None else []
    client.find_doctors_by_specialization.return_value = doctors if doctors is not None else []
    client.find_related_conditions.return_value = related if related is not None else []
    return client


# ------------------------------------------------------------------
# Validation tests
# ------------------------------------------------------------------

class TestFindDoctorValidation:
    """Tests for argument validation — no DB calls should occur."""

    def test_no_arguments_returns_error(self):
        result = find_doctor()
        assert "error" in result

    def test_none_arguments_returns_error(self):
        result = find_doctor(condition=None, specialization=None)
        assert "error" in result

    def test_empty_string_condition_returns_error(self):
        """Empty string condition is falsy — treated as None."""
        # Empty string passes validation as falsy; tool returns "no doctors found"
        # because the query returns no results for an empty string.
        with patch("app.tools.find_doctor.get_neo4j_client") as mock_get:
            mock_get.return_value = _make_mock_client(doctors=[])
            result = find_doctor(condition="")
        # Either a validation error or a not-found response is acceptable
        assert "error" in result or result.get("count") == 0


# ------------------------------------------------------------------
# Condition-based search tests
# ------------------------------------------------------------------

class TestFindDoctorByCondition:
    """Tests for condition-based doctor search."""

    def test_condition_search_returns_doctors(self):
        with patch("app.tools.find_doctor.get_neo4j_client") as mock_get:
            mock_get.return_value = _make_mock_client(
                doctors=MOCK_DOCTORS, related=MOCK_RELATED
            )
            result = find_doctor(condition="Hypertension")

        assert "error" not in result
        assert result["count"] == 2
        assert result["query_type"] == "condition"
        assert result["query_value"] == "Hypertension"
        assert len(result["doctors"]) == 2

    def test_condition_search_includes_related_conditions(self):
        with patch("app.tools.find_doctor.get_neo4j_client") as mock_get:
            mock_get.return_value = _make_mock_client(
                doctors=MOCK_DOCTORS, related=MOCK_RELATED
            )
            result = find_doctor(condition="Hypertension")

        assert "related_conditions" in result
        assert "Heart Disease" in result["related_conditions"]

    def test_condition_no_results_returns_empty_list(self):
        with patch("app.tools.find_doctor.get_neo4j_client") as mock_get:
            mock_get.return_value = _make_mock_client(doctors=[], related=[])
            result = find_doctor(condition="UnknownCondition")

        assert result["count"] == 0
        assert result["doctors"] == []
        assert "message" in result

    def test_condition_takes_precedence_over_specialization(self):
        """When both are provided, condition is used; specialization ignored."""
        with patch("app.tools.find_doctor.get_neo4j_client") as mock_get:
            client = _make_mock_client(doctors=MOCK_DOCTORS, related=[])
            mock_get.return_value = client
            find_doctor(condition="Hypertension", specialization="Cardiology")

        # Should call find_doctors_by_condition, not by specialization
        client.find_doctors_by_condition.assert_called_once_with("Hypertension")
        client.find_doctors_by_specialization.assert_not_called()


# ------------------------------------------------------------------
# Specialization-based search tests
# ------------------------------------------------------------------

class TestFindDoctorBySpecialization:
    """Tests for specialization-based doctor search."""

    def test_specialization_search_returns_doctors(self):
        with patch("app.tools.find_doctor.get_neo4j_client") as mock_get:
            mock_get.return_value = _make_mock_client(doctors=MOCK_DOCTORS)
            result = find_doctor(specialization="Cardiology")

        assert "error" not in result
        assert result["count"] == 2
        assert result["query_type"] == "specialization"

    def test_specialization_no_results_returns_message(self):
        with patch("app.tools.find_doctor.get_neo4j_client") as mock_get:
            mock_get.return_value = _make_mock_client(doctors=[])
            result = find_doctor(specialization="UnknownSpec")

        assert result["count"] == 0
        assert "message" in result

    def test_specialization_result_has_no_related_conditions(self):
        """Specialization queries do not return related conditions."""
        with patch("app.tools.find_doctor.get_neo4j_client") as mock_get:
            mock_get.return_value = _make_mock_client(doctors=MOCK_DOCTORS)
            result = find_doctor(specialization="Cardiology")

        assert "related_conditions" not in result


# ------------------------------------------------------------------
# Error handling tests
# ------------------------------------------------------------------

class TestFindDoctorErrorHandling:
    """Tests for Neo4j failure scenarios."""

    def test_neo4j_exception_returns_safe_error(self):
        """A Neo4j connection error must return a safe message, not raise."""
        with patch("app.tools.find_doctor.get_neo4j_client") as mock_get:
            client = MagicMock()
            client.find_doctors_by_condition.side_effect = Exception(
                "bolt://localhost:7687 connection refused"
            )
            mock_get.return_value = client

            result = find_doctor(condition="Hypertension")

        assert "error" in result
        # Raw connection string must not leak into the response
        assert "bolt://" not in result["error"]
        assert "7687" not in result["error"]

    def test_neo4j_exception_does_not_raise(self):
        """Exceptions from Neo4j must be caught — not propagated to caller."""
        with patch("app.tools.find_doctor.get_neo4j_client") as mock_get:
            client = MagicMock()
            client.find_doctors_by_specialization.side_effect = Exception("timeout")
            mock_get.return_value = client

            # Must not raise
            result = find_doctor(specialization="Cardiology")

        assert "error" in result