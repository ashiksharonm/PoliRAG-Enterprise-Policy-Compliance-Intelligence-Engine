"""Logging configuration for PoliRAG."""

import sys
from typing import Any, Dict

from loguru import logger

from src.config import get_settings


def setup_logging() -> None:
    """Configure logging with loguru."""
    settings = get_settings()

    # Remove default handler
    logger.remove()

    # Determine log format
    if settings.log_format == "json":
        log_format = (
            "{\"time\": \"{time:YYYY-MM-DD HH:mm:ss.SSS}\", "
            '"level": "{level}", '
            '"module": "{module}", '
            '"function": "{function}", '
            '"line": {line}, '
            '"message": "{message}"'
            "}"
        )
    else:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add handler with configured level
    logger.add(
        sys.stdout,
        format=log_format,
        level=settings.log_level,
        colorize=settings.log_format != "json",
    )

    # Add file handler for errors
    logger.add(
        "logs/error.log",
        format=log_format,
        level="ERROR",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
    )

    logger.info(f"Logging initialized at {settings.log_level} level")


def log_with_context(level: str, message: str, **context: Any) -> None:
    """Log message with additional context."""
    logger.bind(**context).log(level, message)


def get_logger(name: str) -> Any:
    """Get a logger instance for a module."""
    return logger.bind(module=name)