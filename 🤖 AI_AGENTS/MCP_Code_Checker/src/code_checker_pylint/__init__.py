"""
Code checker package that runs pylint analysis and generates smart prompts for LLMs.

This package provides functionality to run pylint on a given project
and process the analysis results.
"""

# Re-export public models
from .models import (
    DEFAULT_CATEGORIES,
    PylintCategory,
    PylintMessage,
    PylintMessageType,
    PylintResult,
)

# Re-export main functionality
from .reporting import get_direct_instruction_for_pylint_code, get_pylint_prompt
from .runners import get_pylint_results, run_pylint_check

# Re-export utilities
from .utils import filter_pylint_codes_by_category, normalize_path

# Define the public API explicitly
__all__ = [
    # Models
    "DEFAULT_CATEGORIES",
    "PylintCategory",
    "PylintMessage",
    "PylintMessageType",
    "PylintResult",
    # Main functionality
    "get_pylint_results",
    "run_pylint_check",
    "get_pylint_prompt",
    "get_direct_instruction_for_pylint_code",
    # Utilities
    "filter_pylint_codes_by_category",
    "normalize_path",
]
