"""
Code checker package that runs pytest tests and analyzes the results.

This package provides functionality to run pytest tests on a given project
and process the test results.
"""

# Re-export public models individually
from .models import (
    Collector,
    CollectorResult,
    Crash,
    Log,
    LogRecord,
    PytestReport,
    StageInfo,
    Summary,
    Test,
    TracebackEntry,
    Warning,
)
from .parsers import parse_pytest_report
from .reporting import create_prompt_for_failed_tests, get_test_summary

# Re-export runner functionality
# Re-export main functionality that should be part of the public API
from .runners import check_code_with_pytest, run_tests

# Define the public API explicitly
__all__ = [
    # Models
    "Crash",
    "TracebackEntry",
    "LogRecord",
    "Log",
    "StageInfo",
    "Test",
    "CollectorResult",
    "Collector",
    "Summary",
    "Warning",
    "PytestReport",
    # Main functionality
    "run_tests",
    "parse_pytest_report",
    "check_code_with_pytest",
    "create_prompt_for_failed_tests",
    "get_test_summary",
]
