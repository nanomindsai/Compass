"""Logging configuration for Compass."""

import logging
import sys
from typing import Optional, Dict, Any, Union

import structlog


def setup_logging(
    level: Union[int, str] = logging.INFO,
    format_as_json: bool = False,
) -> None:
    """
    Set up structured logging for Compass.
    
    Args:
        level: The logging level (e.g., logging.INFO, logging.DEBUG).
        format_as_json: Whether to format logs as JSON (useful for production).
    """
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if format_as_json:
        # JSON formatter for production
        formatter = structlog.processors.JSONRenderer()
    else:
        # Console formatter for development
        formatter = structlog.dev.ConsoleRenderer(colors=True)
    
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    handler = logging.StreamHandler(sys.stdout)
    
    # Use ProcessorFormatter to format logs from both structlog and standard library
    processor_formatter = structlog.stdlib.ProcessorFormatter(
        processor=formatter,
        foreign_pre_chain=shared_processors,
    )
    
    handler.setFormatter(processor_formatter)
    
    # Set up the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    
    # Set level for all loggers
    logging.getLogger("compass").setLevel(level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger with the given name.
    
    Args:
        name: Name of the logger.
        
    Returns:
        structlog.stdlib.BoundLogger: A structured logger.
    """
    return structlog.get_logger(name)