"""
Structured logging setup for the backend.
- Set LOG_JSON=true for JSON lines (one object per line) for production.
- Otherwise uses a human-readable format.
"""
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            log_obj["request_id"] = getattr(record, "request_id", None)
        return json.dumps(log_obj, default=str)


def configure_logging(
    log_level: str = "INFO",
    log_json: bool = False,
) -> None:
    """
    Configure root logger and common handlers.
    Call once at app startup (e.g. in main.py).
    """
    import os
    level = getattr(logging, log_level.upper(), logging.INFO)
    use_json = log_json or (os.getenv("LOG_JSON", "false").lower() == "true")

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers when reloading
    if root.handlers:
        for h in root.handlers[:]:
            root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if use_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
        )
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""
    return logging.getLogger(name)
