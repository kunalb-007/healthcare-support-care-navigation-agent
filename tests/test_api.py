"""
tests/test_api.py
------------------
Integration tests for the FastAPI /chat and /health endpoints.

The agent (run_agent) is mocked so no real LLM or database calls
are made. Tests verify the HTTP contract: request validation,
response schema, and error handling.
"""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Must be set before any app module is imported so the Settings singleton
# picks them up. Using os.environ directly (not setdefault) so they
# override anything already present in the environment.
os.environ["OPENROUTER_API_KEY"] = "test-key-for-unit-tests"
os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000"

from app.main import app  # noqa: E402  (import after env setup)


@pytest.fixture
def client():
    """
    TestClient that runs the FastAPI lifespan with two side-effects mocked:
      - settings.validate() — would raise without a real API key
      - init_appointments_db() — would create a real SQLite file

    The lifespan itself still runs so the startup/shutdown path is exercised.
    """
    with (
        patch("app.main.settings.validate"),          # skip config check
        patch("app.main.init_appointments_db"),        # skip DB init
        TestClient(app, raise_server_exceptions=False) as c,
    ):
        yield c


def _mock_agent_result(
        answer="Test answer.",
        tools_used=None,
        total_turns=0,
):
    return {
        "answer": answer,
        "tools_used": tools_used or [],
        "total_turns": total_turns,
    }


# ------------------------------------------------------------------
# /chat endpoint tests
# ------------------------------------------------------------------

class TestChatEndpoint:

    def test_valid_request_returns_200(self, client):
        with patch("app.main.run_agent", return_value=_mock_agent_result()):
            response = client.post("/chat", json={"message": "What is hypertension?"})
        assert response.status_code == 200

    def test_response_schema_has_required_fields(self, client):
        with patch("app.main.run_agent", return_value=_mock_agent_result()):
            response = client.post("/chat", json={"message": "Hello"})
        data = response.json()
        assert "answer" in data
        assert "tools_used" in data
        assert "total_turns" in data

    def test_answer_content_matches_agent_output(self, client):
        with patch("app.main.run_agent", return_value=_mock_agent_result(answer="42")):
            response = client.post("/chat", json={"message": "Test"})
        assert response.json()["answer"] == "42"

    def test_tools_used_returned_correctly(self, client):
        tools = [{"tool": "calculate_bmi", "args": {"weight_kg": 70, "height_cm": 175}}]
        with patch("app.main.run_agent", return_value=_mock_agent_result(tools_used=tools, total_turns=1)):
            response = client.post("/chat", json={"message": "My BMI?"})
        data = response.json()
        assert len(data["tools_used"]) == 1
        assert data["tools_used"][0]["tool"] == "calculate_bmi"
        assert data["total_turns"] == 1

    def test_empty_message_returns_422(self, client):
        """Empty message violates min_length=1 → 422 Unprocessable Entity."""
        response = client.post("/chat", json={"message": ""})
        assert response.status_code == 422

    def test_missing_message_field_returns_422(self, client):
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_message_too_long_returns_422(self, client):
        """Message exceeding max_length=1000 → 422."""
        response = client.post("/chat", json={"message": "x" * 1001})
        assert response.status_code == 422

    def test_agent_exception_returns_500(self, client):
        with patch("app.main.run_agent", side_effect=Exception("LLM down")):
            response = client.post("/chat", json={"message": "Hello"})
        assert response.status_code == 500

    def test_500_response_does_not_expose_internals(self, client):
        """Raw exception messages must not leak in 500 responses."""
        with patch("app.main.run_agent", side_effect=Exception("secret internal path /data/model")):
            response = client.post("/chat", json={"message": "Hello"})
        body = response.json()
        assert "/data/model" not in str(body)
        assert "secret" not in str(body)


# ------------------------------------------------------------------
# Health endpoint tests
# ------------------------------------------------------------------

class TestHealthEndpoints:

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "2.0.0"

    def test_health_endpoint_ok_when_neo4j_up(self, client):
        with patch("app.main._check_neo4j", return_value=True):
            response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["components"]["neo4j"] == "ok"

    def test_health_endpoint_degraded_when_neo4j_down(self, client):
        with patch("app.main._check_neo4j", return_value=False):
            response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["neo4j"] == "unavailable"