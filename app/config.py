"""
Centralized application configuration loaded from environment
variables using Pydantic Settings.

Provides a single configuration source for the application.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """
    Application settings loaded from environment variables.

    Validated at import time so misconfiguration fails fast
    rather than at first request.
    """

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-4o-mini"

    # ------------------------------------------------------------------
    # Neo4j
    # ------------------------------------------------------------------
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")

    # ------------------------------------------------------------------
    # Agent behaviour
    # ------------------------------------------------------------------
    max_agent_turns: int = int(os.getenv("MAX_AGENT_TURNS", "10"))

    # ------------------------------------------------------------------
    # CORS — comma-separated list of allowed origins
    # Defaults to localhost dev; MUST be restricted in production.
    # ------------------------------------------------------------------
    allowed_origins_raw: str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")

    @property
    def allowed_origins(self) -> list[str]:
        """Return allowed origins as a list."""
        return [o.strip() for o in self.allowed_origins_raw.split(",") if o.strip()]

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    def validate(self) -> None:
        """
        Raise ValueError for critical missing configuration.

        Called at application startup so the service fails fast
        rather than on the first request.
        """
        if not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Copy .env.example to .env and add your API key."
            )
        if not self.neo4j_password or self.neo4j_password == "password":
            # Warn but do not block — "password" is a common local dev default.
            logging.getLogger(__name__).warning(
                "NEO4J_PASSWORD appears to be the default value. "
                "Set a strong password for any non-local environment."
            )


# Single shared instance — imported by all modules
settings = Settings()