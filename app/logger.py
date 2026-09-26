"""
app/logger.py
-------------
Application-wide logging setup.

Provides consistent log formatting across all modules.
Each module calls get_logger(__name__) to get a named logger.

What we log:
  - Request start/end with request_id and total latency
  - LLM call latency per turn
  - Tool name being executed
  - Errors and exceptions (sanitized — no secrets or internal paths)

What we do NOT log:
  - API keys or passwords
  - Patient names or health data from requests
  - Raw exception tracebacks sent to users
"""

import logging
import sys

from app.config import settings

_CONFIGURED = False


def _configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    numeric_level = getattr(logging, settings.log_level, logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%dT%H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(numeric_level)
    if not root.handlers:
        root.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _CONFIGURED = True


_configure_logging()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)