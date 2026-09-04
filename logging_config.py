"""
Structured JSON logging configuration for the card scanner service.
Provides correlation ID support for request tracing.
"""

import logging
import json
import sys
from datetime import datetime, timezone
from contextvars import ContextVar
from typing import Optional

# Context variable for correlation ID (thread-safe)
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from context."""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: Optional[str]) -> None:
    """Set the correlation ID in context."""
    correlation_id_var.set(correlation_id)


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    Outputs JSON lines with timestamp, level, logger, message, and correlation_id.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add correlation ID if present
        correlation_id = get_correlation_id()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry)


class StandardFormatter(logging.Formatter):
    """
    Standard text formatter with correlation ID support.
    Format: timestamp - level - logger - [correlation_id] - message
    """

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        correlation_id = get_correlation_id()

        if correlation_id:
            prefix = f"{timestamp} - {record.levelname} - {record.name} - [{correlation_id}]"
        else:
            prefix = f"{timestamp} - {record.levelname} - {record.name}"

        message = f"{prefix} - {record.getMessage()}"

        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

        return message


def setup_logging(level: str = "INFO", json_format: bool = True) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: If True, use JSON formatting; otherwise use standard text format
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Set formatter based on preference
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(StandardFormatter())

    root_logger.addHandler(console_handler)

    # Reduce noise from external libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("git").setLevel(logging.WARNING)


class CorrelationLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes correlation ID in log messages.
    """

    def process(self, msg, kwargs):
        correlation_id = get_correlation_id()
        if correlation_id:
            return f"[{correlation_id}] {msg}", kwargs
        return msg, kwargs


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
