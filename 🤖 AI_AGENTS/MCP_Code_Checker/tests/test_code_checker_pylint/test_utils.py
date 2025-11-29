"""Unit tests for pylint utils module."""

import os
from typing import Set

from src.code_checker_pylint.models import PylintMessageType
from src.code_checker_pylint.utils import (
    filter_pylint_codes_by_category,
    normalize_path,
)


class TestNormalizePath:
    """Test cases for normalize_path function."""

    def test_normalize_path_with_base_dir_prefix(self) -> None:
        """Test normalizing a path that starts with the base directory."""
        base_dir = os.path.join("home", "user", "project")
        path = os.path.join("home", "user", "project", "src", "module.py")
        result = normalize_path(path, base_dir)
        assert result == os.path.join("src", "module.py")

    def test_normalize_path_without_base_dir_prefix(self) -> None:
        """Test normalizing a path that doesn't start with the base directory."""
        base_dir = os.path.join("home", "user", "project")
        path = os.path.join("other", "path", "module.py")
        result = normalize_path(path, base_dir)
        assert result == os.path.join("other", "path", "module.py")

    def test_normalize_path_with_backslashes(self) -> None:
        """Test normalizing a path with backslashes."""
        base_dir = "home/user/project"
        path = "home\\user\\project\\src\\module.py"
        result = normalize_path(path, base_dir)
        expected = os.path.join("src", "module.py")
        assert result == expected

    def test_normalize_path_with_forward_slashes(self) -> None:
        """Test normalizing a path with forward slashes."""
        base_dir = "home\\user\\project"
        path = "home/user/project/src/module.py"
        result = normalize_path(path, base_dir)
        expected = os.path.join("src", "module.py")
        assert result == expected

    def test_normalize_path_base_dir_without_trailing_sep(self) -> None:
        """Test normalizing when base_dir doesn't have trailing separator."""
        base_dir = os.path.join("home", "user", "project")
        path = os.path.join("home", "user", "project", "src", "module.py")
        result = normalize_path(path, base_dir)
        assert result == os.path.join("src", "module.py")

    def test_normalize_path_base_dir_with_trailing_sep(self) -> None:
        """Test normalizing when base_dir has trailing separator."""
        base_dir = os.path.join("home", "user", "project") + os.path.sep
        path = os.path.join("home", "user", "project", "src", "module.py")
        result = normalize_path(path, base_dir)
        assert result == os.path.join("src", "module.py")


class TestFilterPylintCodesByCategory:
    """Test cases for filter_pylint_codes_by_category function."""

    def test_filter_by_single_category(self) -> None:
        """Test filtering by a single category."""
        pylint_codes: Set[str] = {"C0301", "R0201", "W0613", "E0602", "F0001"}
        categories: Set[PylintMessageType] = {PylintMessageType.ERROR}
        expected_codes: Set[str] = {"E0602"}
        assert (
            filter_pylint_codes_by_category(pylint_codes, categories) == expected_codes
        )

    def test_filter_by_multiple_categories(self) -> None:
        """Test filtering by multiple categories."""
        pylint_codes: Set[str] = {"C0301", "R0201", "W0613", "E0602", "F0001"}
        categories: Set[PylintMessageType] = {
            PylintMessageType.ERROR,
            PylintMessageType.FATAL,
        }
        expected_codes: Set[str] = {"E0602", "F0001"}
        assert (
            filter_pylint_codes_by_category(pylint_codes, categories) == expected_codes
        )

    def test_filter_with_convention_category(self) -> None:
        """Test filtering with convention category."""
        pylint_codes: Set[str] = {"C0301", "C0114", "R0201", "W0613"}
        categories: Set[PylintMessageType] = {PylintMessageType.CONVENTION}
        expected_codes: Set[str] = {"C0301", "C0114"}
        assert (
            filter_pylint_codes_by_category(pylint_codes, categories) == expected_codes
        )

    def test_filter_with_refactor_category(self) -> None:
        """Test filtering with refactor category."""
        pylint_codes: Set[str] = {"R0902", "R0911", "E0602", "W0613"}
        categories: Set[PylintMessageType] = {PylintMessageType.REFACTOR}
        expected_codes: Set[str] = {"R0902", "R0911"}
        assert (
            filter_pylint_codes_by_category(pylint_codes, categories) == expected_codes
        )

    def test_filter_with_warning_category(self) -> None:
        """Test filtering with warning category."""
        pylint_codes: Set[str] = {"W0612", "W0613", "E0602", "C0301"}
        categories: Set[PylintMessageType] = {PylintMessageType.WARNING}
        expected_codes: Set[str] = {"W0612", "W0613"}
        assert (
            filter_pylint_codes_by_category(pylint_codes, categories) == expected_codes
        )

    def test_filter_with_empty_pylint_codes(self) -> None:
        """Test filtering with empty pylint codes."""
        pylint_codes: Set[str] = set()
        categories: Set[PylintMessageType] = {
            PylintMessageType.ERROR,
            PylintMessageType.FATAL,
        }
        expected_codes: Set[str] = set()
        assert (
            filter_pylint_codes_by_category(pylint_codes, categories) == expected_codes
        )

    def test_filter_with_empty_categories(self) -> None:
        """Test filtering with empty categories."""
        pylint_codes: Set[str] = {"C0301", "R0201", "W0613", "E0602", "F0001"}
        categories: Set[PylintMessageType] = set()
        expected_codes: Set[str] = set()
        assert (
            filter_pylint_codes_by_category(pylint_codes, categories) == expected_codes
        )

    def test_filter_with_all_categories(self) -> None:
        """Test filtering with all categories."""
        pylint_codes: Set[str] = {"C0301", "R0201", "W0613", "E0602", "F0001"}
        categories: Set[PylintMessageType] = {
            PylintMessageType.CONVENTION,
            PylintMessageType.REFACTOR,
            PylintMessageType.WARNING,
            PylintMessageType.ERROR,
            PylintMessageType.FATAL,
        }
        expected_codes: Set[str] = {"C0301", "R0201", "W0613", "E0602", "F0001"}
        assert (
            filter_pylint_codes_by_category(pylint_codes, categories) == expected_codes
        )

    def test_filter_codes_with_lowercase_prefix(self) -> None:
        """Test that filtering is case-sensitive (codes should have uppercase prefixes)."""
        pylint_codes: Set[str] = {"c0301", "r0201", "w0613", "e0602", "f0001"}
        categories: Set[PylintMessageType] = {PylintMessageType.ERROR}
        expected_codes: Set[str] = set()  # No matches because prefixes are lowercase
        assert (
            filter_pylint_codes_by_category(pylint_codes, categories) == expected_codes
        )
