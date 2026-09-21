"""
Configures application-wide logging with a consistent format,
including optional contextual fields through the extra parameter.
"""

import logging
import sys
from app.config import settings


_CONFIGURED = False


def _configure_logging() -> None:
    """Configure the root logger once at import time."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    numeric_level = getattr(logging, settings.log_level, logging.INFO)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(numeric_level)
    # Avoid adding duplicate handlers if the function is called more than once
    if not root.handlers:
        root.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _CONFIGURED = True


_configure_logging()


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        logging.Logger configured by the root setup above.
    """
    return logging.getLogger(name)