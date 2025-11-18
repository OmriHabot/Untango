"""
Centralized logging configuration for the RAG application.
Configured for optimal Docker container logging with structured output.
"""
import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    Makes logs easily parseable by log aggregation tools.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data
        
        return json.dumps(log_data)


class SimpleFormatter(logging.Formatter):
    """
    Simple, clean formatter for production and development.
    Format: LEVEL [timestamp]: message
    """
    
    def format(self, record: logging.LogRecord) -> str:
        # Format timestamp as ISO format without milliseconds
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%dT%H:%M:%S')
        
        # Get the message
        message = record.getMessage()
        
        # Build the log line
        log_line = f"{record.levelname} [{timestamp}]: {message}"
        
        # Add exception info if present
        if record.exc_info:
            log_line += "\n" + self.formatException(record.exc_info)
        
        return log_line


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for development with human-readable output.
    """
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        # Format timestamp as ISO format without milliseconds
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%dT%H:%M:%S')
        
        # Get the message
        message = record.getMessage()
        
        # Add color to level
        color = self.COLORS.get(record.levelname, self.RESET)
        colored_level = f"{color}{record.levelname}{self.RESET}"
        
        # Build the log line
        log_line = f"{colored_level} [{timestamp}]: {message}"
        
        # Add exception info if present
        if record.exc_info:
            log_line += "\n" + self.formatException(record.exc_info)
        
        return log_line


def setup_logging(log_level: str = "INFO", json_logs: bool = False) -> None:
    """
    Configure application-wide logging.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: If True, use JSON formatting (for log aggregation tools).
                   If False (default), use simple clean format: LEVEL [timestamp]: message
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create console handler (stdout for Docker)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Choose formatter based on configuration
    if json_logs:
        formatter = JSONFormatter()
    else:
        formatter = SimpleFormatter()
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured - Level: {log_level.upper()}, Format: {'JSON' if json_logs else 'Simple'}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name of the logger (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_with_extra(logger: logging.Logger, level: str, message: str, **kwargs) -> None:
    """
    Log a message with extra structured data.
    
    Args:
        logger: Logger instance
        level: Log level (debug, info, warning, error, critical)
        message: Log message
        **kwargs: Extra data to include in structured logs
    """
    log_func = getattr(logger, level.lower())
    extra = {"extra_data": kwargs} if kwargs else {}
    log_func(message, extra=extra)

