"""Structured logging setup for C. elegans emulator.

In production (LOG_LEVEL != DEBUG) logs are emitted as JSON to a file handler.
Raw exception tracebacks are never written to stdout; use logger.exception()
which sends structured traces to the log file only.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class _JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str, level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Return a named logger with JSON handler (file) and simple handler (stdout).

    Args:
        name: Logger name (typically __name__ of caller).
        level: Log level string, e.g. "INFO", "DEBUG".
        log_file: Optional path for the JSON file handler.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured — avoid adding duplicate handlers.
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Stdout handler — plain text, no raw tracebacks.
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(numeric_level)
    stdout_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(stdout_handler)

    # File handler — structured JSON.
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(_JSONFormatter())
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid double-logging.
    logger.propagate = False
    return logger
