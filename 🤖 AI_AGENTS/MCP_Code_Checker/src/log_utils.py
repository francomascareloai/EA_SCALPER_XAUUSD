"""Logging utilities for the MCP Code Checker server."""

import json
import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, cast

import structlog
from pythonjsonlogger import jsonlogger  # type: ignore[import-untyped]

# Type variable for function return types
T = TypeVar("T")

# Create standard logger
stdlogger = logging.getLogger(__name__)


def setup_logging(log_level: str, log_file: Optional[str] = None) -> None:
    """Configure logging - if log_file specified, logs only to file; otherwise to console."""
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.setLevel(numeric_level)

    # Set up logging based on whether log_file is specified
    if log_file:
        # FILE LOGGING ONLY - no console output
        # Create directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

        # Configure JSON file handler
        json_handler = logging.FileHandler(log_file)

        # This formatter ensures timestamp and level are included as separate fields in JSON
        json_formatter = jsonlogger.JsonFormatter(  # type: ignore[attr-defined]
            fmt="%(timestamp)s %(level)s %(name)s %(message)s %(module)s %(funcName)s %(lineno)d",
            timestamp=True,
        )
        json_handler.setFormatter(json_formatter)
        root_logger.addHandler(json_handler)

        # Configure structlog processors
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Log initialization message to file only
        stdlogger.info(f"Logging initialized: file={log_file}, level={log_level}")
    else:
        # CONSOLE LOGGING ONLY (fallback when no file specified)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        stdlogger.info(f"Logging initialized: console={log_level}")


def log_function_call(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to log function calls with parameters, timing, and results."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        func_name = func.__name__
        module_name = func.__module__
        line_no = func.__code__.co_firstlineno

        # Prepare parameters for logging
        log_params = {}

        # Handle method calls (skip self/cls)
        if (
            args
            and hasattr(args[0], "__class__")
            and args[0].__class__.__module__ != "builtins"
        ):
            log_params.update(
                {
                    k: v
                    for k, v in zip(func.__code__.co_varnames[1 : len(args)], args[1:])
                }
            )
        else:
            log_params.update(
                {k: v for k, v in zip(func.__code__.co_varnames[: len(args)], args)}
            )

        # Add keyword arguments
        log_params.update(kwargs)

        # Convert Path objects to strings and handle other non-serializable types
        serializable_params = {}
        for k, v in log_params.items():
            if isinstance(v, Path):
                serializable_params[k] = str(v)
            else:
                try:
                    # Test if it's JSON serializable
                    json.dumps(v)
                    serializable_params[k] = v
                except (TypeError, OverflowError):
                    # If not serializable, convert to string
                    serializable_params[k] = str(v)

        # Check if structured logging is enabled
        has_structured = any(
            isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers
        )

        # Log function call - always provide 'event' as the first parameter
        if has_structured:
            structlogger = structlog.get_logger(module_name)
            structlogger.debug(
                f"Calling function {func_name}",  # This is the 'event' parameter
                function=func_name,
                parameters=serializable_params,
                module=module_name,
                lineno=line_no,
            )

        stdlogger.debug(
            f"Calling {func_name} with parameters: {json.dumps(serializable_params, default=str)}"
        )

        # Execute function and measure time
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed_ms = round((time.time() - start_time) * 1000, 2)

            # Handle large results
            result_for_log: Any = result
            if isinstance(result, (list, dict)) and len(str(result)) > 1000:
                result_for_log = f"<Large result of type {type(result).__name__}, length: {len(str(result))}>"

            # Attempt to make result JSON serializable for structured logging
            serializable_result: Optional[Any] = None
            try:
                if result is not None:
                    json.dumps(result)  # Test if result is JSON serializable
                    serializable_result = result
            except (TypeError, OverflowError):
                serializable_result = str(result)

            # Log completion
            if has_structured:
                structlogger.debug(
                    f"Function {func_name} completed",  # This is the 'event' parameter
                    function=func_name,
                    execution_time_ms=elapsed_ms,
                    status="success",
                    result=serializable_result,
                    module=module_name,
                    lineno=line_no,
                )

            stdlogger.debug(
                f"{func_name} completed in {elapsed_ms}ms with result: {result_for_log}"
            )
            return result

        except Exception as e:
            # Log exceptions
            elapsed_ms = round((time.time() - start_time) * 1000, 2)

            if has_structured:
                structlogger.error(
                    f"Function {func_name} failed",  # This is the 'event' parameter
                    function=func_name,
                    execution_time_ms=elapsed_ms,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    module=module_name,
                    lineno=line_no,
                    exc_info=True,
                )

            stdlogger.error(
                f"{func_name} failed after {elapsed_ms}ms with error: {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise

    return cast(Callable[..., T], wrapper)
