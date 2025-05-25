"""
Logging Configuration for Universal MCP Server
Provides structured logging with multiple handlers and formatters
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

def setup_logger(
    name: str = "universal_mcp_server",
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_size_mb: int = 100,
    backup_count: int = 5,
    enable_console: bool = True,
    enable_color: bool = True,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up a comprehensive logger with file rotation and console output.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        max_size_mb: Maximum size of log file in MB before rotation
        backup_count: Number of backup files to keep
        enable_console: Whether to log to console
        enable_color: Whether to use colored console output
        log_format: Custom log format string
    
    Returns:
        Configured logger instance
    """
    
    logger = logging.getLogger(name)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Default log format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        if enable_color and HAS_COLORLOG:
            # Colored console formatter
            color_formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message_log_color)s%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                },
                secondary_log_colors={
                    'message': {
                        'ERROR': 'red',
                        'CRITICAL': 'red,bg_white'
                    }
                }
            )
            console_handler.setFormatter(color_formatter)
        else:
            # Standard console formatter
            console_formatter = logging.Formatter(
                log_format,
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
        
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,  # Convert MB to bytes
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        
        # File formatter (no colors)
        file_formatter = logging.Formatter(
            log_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
    
    # Prevent log messages from being handled by parent loggers
    logger.propagate = False
    
    return logger

def setup_request_logger(name: str = "requests") -> logging.Logger:
    """Set up a specialized logger for HTTP requests."""
    logger = logging.getLogger(f"universal_mcp_server.{name}")
    
    if not logger.handlers:
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Request log handler
        handler = logging.handlers.RotatingFileHandler(
            log_dir / "requests.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        
        formatter = logging.Formatter(
            "%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    
    return logger

def setup_security_logger(name: str = "security") -> logging.Logger:
    """Set up a specialized logger for security events."""
    logger = logging.getLogger(f"universal_mcp_server.{name}")
    
    if not logger.handlers:
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Security log handler
        handler = logging.handlers.RotatingFileHandler(
            log_dir / "security.log",
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=5
        )
        
        formatter = logging.Formatter(
            "%(asctime)s - SECURITY - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)  # Only log warnings and above for security
        logger.propagate = False
    
    return logger

def setup_audit_logger(name: str = "audit") -> logging.Logger:
    """Set up a specialized logger for audit trail."""
    logger = logging.getLogger(f"universal_mcp_server.{name}")
    
    if not logger.handlers:
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Audit log handler (no rotation for compliance)
        handler = logging.FileHandler(
            log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        
        formatter = logging.Formatter(
            "%(asctime)s - AUDIT - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    
    return logger

def log_request(
    method: str,
    path: str,
    status_code: int,
    response_time_ms: float,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    **kwargs
):
    """Log an HTTP request with structured data."""
    request_logger = setup_request_logger()
    
    log_data = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "response_time_ms": response_time_ms,
        "timestamp": datetime.now().isoformat()
    }
    
    if user_id:
        log_data["user_id"] = user_id
    if ip_address:
        log_data["ip_address"] = ip_address
    if user_agent:
        log_data["user_agent"] = user_agent
    
    log_data.update(kwargs)
    
    # Format as JSON-like string for easy parsing
    log_message = " ".join([f"{k}={v}" for k, v in log_data.items()])
    request_logger.info(log_message)

def log_security_event(
    event_type: str,
    severity: str,
    description: str,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    **kwargs
):
    """Log a security event."""
    security_logger = setup_security_logger()
    
    log_data = {
        "event_type": event_type,
        "severity": severity,
        "description": description,
        "timestamp": datetime.now().isoformat()
    }
    
    if user_id:
        log_data["user_id"] = user_id
    if ip_address:
        log_data["ip_address"] = ip_address
    
    log_data.update(kwargs)
    
    log_message = " ".join([f"{k}={v}" for k, v in log_data.items()])
    
    # Map severity to log level
    level_map = {
        "low": logging.INFO,
        "medium": logging.WARNING,
        "high": logging.ERROR,
        "critical": logging.CRITICAL
    }
    
    level = level_map.get(severity.lower(), logging.WARNING)
    security_logger.log(level, log_message)

def log_audit_event(
    action: str,
    resource: str,
    user_id: Optional[str] = None,
    result: str = "success",
    details: Optional[Dict[str, Any]] = None
):
    """Log an audit event for compliance."""
    audit_logger = setup_audit_logger()
    
    log_data = {
        "action": action,
        "resource": resource,
        "result": result,
        "timestamp": datetime.now().isoformat()
    }
    
    if user_id:
        log_data["user_id"] = user_id
    if details:
        log_data.update(details)
    
    log_message = " ".join([f"{k}={v}" for k, v in log_data.items()])
    audit_logger.info(log_message)

class StructuredLogger:
    """A structured logger that provides consistent logging across the application."""
    
    def __init__(self, name: str, context: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(name)
        self.context = context or {}
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log a message with context."""
        log_data = self.context.copy()
        log_data.update(kwargs)
        
        if log_data:
            context_str = " ".join([f"{k}={v}" for k, v in log_data.items()])
            full_message = f"{message} | {context_str}"
        else:
            full_message = message
        
        self.logger.log(level, full_message)
    
    def debug(self, message: str, **kwargs):
        """Log a debug message."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log an info message."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log a warning message."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log an error message."""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log a critical message."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def with_context(self, **context) -> 'StructuredLogger':
        """Create a new logger with additional context."""
        new_context = self.context.copy()
        new_context.update(context)
        return StructuredLogger(self.logger.name, new_context)

def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name, context)

# Set up default loggers when module is imported
default_logger = setup_logger()
request_logger = setup_request_logger()
security_logger = setup_security_logger()
audit_logger = setup_audit_logger()

# Export commonly used functions
__all__ = [
    'setup_logger',
    'setup_request_logger',
    'setup_security_logger', 
    'setup_audit_logger',
    'log_request',
    'log_security_event',
    'log_audit_event',
    'StructuredLogger',
    'get_logger'
]
