"""
Utility functions for pylint code checking.
"""

import os
from typing import Set

from .models import PylintMessageType


def normalize_path(path: str, base_dir: str) -> str:
    """
    Normalize a path relative to the base directory.

    Args:
        path: The path to normalize
        base_dir: The base directory to make the path relative to

    Returns:
        Normalized path
    """
    # Normalize both paths to use the platform-specific separator
    normalized_path = path.replace("\\", os.path.sep).replace("/", os.path.sep)
    normalized_base = base_dir.replace("\\", os.path.sep).replace("/", os.path.sep)

    # Make path relative to base_dir if it starts with base_dir
    if normalized_path.startswith(normalized_base):
        prefix = normalized_base
        if not prefix.endswith(os.path.sep):
            prefix += os.path.sep
        normalized_path = normalized_path.replace(prefix, "", 1)

    return normalized_path


def filter_pylint_codes_by_category(
    pylint_codes: Set[str],
    categories: Set[PylintMessageType],
) -> Set[str]:
    """
    Filters Pylint codes based on the specified categories.

    Args:
        pylint_codes: A set of Pylint codes (e.g., {"C0301", "R0201", "W0613", "E0602", "F0001"}).
        categories: A set of PylintMessageType enums to filter by (e.g., {PylintMessageType.ERROR, PylintMessageType.FATAL}).

    Returns:
        A set of Pylint codes that match the specified categories.
    """
    category_prefixes = {
        PylintMessageType.CONVENTION: "C",
        PylintMessageType.REFACTOR: "R",
        PylintMessageType.WARNING: "W",
        PylintMessageType.ERROR: "E",
        PylintMessageType.FATAL: "F",
    }
    filtered_codes: Set[str] = set()
    for code in pylint_codes:
        for category in categories:
            if code.startswith(category_prefixes[category]):
                filtered_codes.add(code)
                break
    return filtered_codes
