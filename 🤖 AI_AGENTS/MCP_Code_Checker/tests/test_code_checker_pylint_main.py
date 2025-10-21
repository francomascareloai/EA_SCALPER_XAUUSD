import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Set

import pytest

from src.code_checker_pylint import (
    DEFAULT_CATEGORIES,
    PylintCategory,
    filter_pylint_codes_by_category,
    get_pylint_results,
    run_pylint_check,
)


# Helper functions needed for tests
def write_file(file_path: str, content: str) -> None:
    """Write content to a file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.write(content)


def read_file(file_path: str) -> str:
    """Read content from a file."""
    with open(file_path, "r") as f:
        return f.read()


def create_default_project(
    project_dir: str, provide_config_module: bool = False
) -> None:
    """Create a basic Python project structure."""
    os.makedirs(os.path.join(project_dir, "src"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "tests"), exist_ok=True)
    write_file(os.path.join(project_dir, "src", "__init__.py"), "")
    write_file(os.path.join(project_dir, "tests", "__init__.py"), "")
    if provide_config_module:
        write_file(os.path.join(project_dir, "src", "config.py"), "DEBUG = True\n")


@pytest.fixture  # type: ignore[misc]  # confusion with fixture
def temp_project_dir() -> Generator[Path, None, None]:
    """Creates a temporary project directory for testing, cleaning it up after the test."""
    temp_dir = Path(tempfile.mkdtemp())
    create_default_project(str(temp_dir), provide_config_module=False)

    yield temp_dir

    shutil.rmtree(temp_dir)


def test_get_pylint_results_no_issues(temp_project_dir: Path) -> None:
    """Tests get_pylint_results with a project that has no Pylint issues."""
    write_file(
        os.path.join(temp_project_dir, "src", "test_module.py"),
        "def hello():\n    print('hello')\n",
    )

    result = get_pylint_results(
        str(temp_project_dir), disable_codes=["C0114", "C0116"], python_executable=None
    )
    assert result.return_code == 0
    assert not result.messages
    assert result.error is None


def test_get_pylint_results_with_issues(temp_project_dir: Path) -> None:
    """Tests get_pylint_results with a project that has Pylint issues."""
    write_file(
        os.path.join(temp_project_dir, "src", "test_module.py"),
        "def hello():\n    print('hello')\n",
    )
    result = get_pylint_results(str(temp_project_dir), python_executable=None)
    # assert result.return_code == 0
    assert len(result.messages) > 0
    assert result.error is None
    assert any(msg.symbol == "missing-function-docstring" for msg in result.messages)


def test_get_pylint_results_invalid_project_dir() -> None:
    """Tests get_pylint_results with an invalid project directory."""
    with pytest.raises(FileNotFoundError):
        get_pylint_results("invalid_dir")


def test_get_pylint_results_pylint_error(temp_project_dir: Path) -> None:
    """Tests get_pylint_results with a project that causes Pylint to error out."""
    write_file(
        os.path.join(temp_project_dir, "src", "test_module.py"),
        "def hello()\n    print('hello')\n",
    )  # missing colon
    result = get_pylint_results(str(temp_project_dir), python_executable=None)

    assert result.return_code != 0
    # assert result.messages == []
    # assert result.error is not None


def test_get_pylint_results_empty_file(temp_project_dir: Path) -> None:
    """Tests get_pylint_results with an empty python file"""
    write_file(os.path.join(temp_project_dir, "src", "empty_file.py"), "")

    result = get_pylint_results(str(temp_project_dir), python_executable=None)
    assert result.return_code == 0
    assert len(result.messages) == 0
    assert result.error is None


def test_run_pylint_check(temp_project_dir: Path) -> None:
    """Tests the new run_pylint_check function."""
    write_file(
        os.path.join(temp_project_dir, "src", "test_module.py"),
        "def hello():\n    print('hello')\n",
    )

    # Test with default parameters
    result = run_pylint_check(str(temp_project_dir))
    assert isinstance(result.return_code, int)
    assert isinstance(result.messages, list)

    # Test with categories parameter
    result = run_pylint_check(
        str(temp_project_dir), categories={PylintCategory.ERROR, PylintCategory.FATAL}
    )
    assert isinstance(result.return_code, int)

    # Test with disable_codes parameter
    result = run_pylint_check(str(temp_project_dir), disable_codes=["C0114", "C0116"])
    assert isinstance(result.return_code, int)


def test_default_categories_from_init() -> None:
    """Tests that DEFAULT_CATEGORIES is correctly exposed via __init__."""
    assert DEFAULT_CATEGORIES is not None
    assert isinstance(DEFAULT_CATEGORIES, set)
    assert PylintCategory.ERROR in DEFAULT_CATEGORIES
    assert PylintCategory.FATAL in DEFAULT_CATEGORIES


class TestFilterPylintCodesByCategory:
    def test_filter_by_single_category(self) -> None:
        pylint_codes: Set[str] = {"C0301", "R0201", "W0613", "E0602", "F0001"}
        categories: Set[PylintCategory] = {PylintCategory.ERROR}
        expected_codes: Set[str] = {"E0602"}
        assert (
            filter_pylint_codes_by_category(pylint_codes, categories) == expected_codes
        )

    def test_filter_by_multiple_categories(self) -> None:
        pylint_codes: Set[str] = {"C0301", "R0201", "W0613", "E0602", "F0001"}
        categories: Set[PylintCategory] = {PylintCategory.ERROR, PylintCategory.FATAL}
        expected_codes: Set[str] = {"E0602", "F0001"}
        assert (
            filter_pylint_codes_by_category(pylint_codes, categories) == expected_codes
        )

    def test_filter_with_no_matching_category(self) -> None:
        pylint_codes: Set[str] = {"C0301", "R0201", "W0613", "E0602", "F0001"}
        categories: Set[PylintCategory] = {PylintCategory.CONVENTION}
        expected_codes: Set[str] = {"C0301"}
        assert (
            filter_pylint_codes_by_category(pylint_codes, categories) == expected_codes
        )

    def test_filter_with_empty_pylint_codes(self) -> None:
        pylint_codes: Set[str] = set()
        categories: Set[PylintCategory] = {PylintCategory.ERROR, PylintCategory.FATAL}
        expected_codes: Set[str] = set()
        assert (
            filter_pylint_codes_by_category(pylint_codes, categories) == expected_codes
        )

    def test_filter_with_empty_categories(self) -> None:
        pylint_codes: Set[str] = {"C0301", "R0201", "W0613", "E0602", "F0001"}
        categories: Set[PylintCategory] = set()
        expected_codes: Set[str] = set()
        assert (
            filter_pylint_codes_by_category(pylint_codes, categories) == expected_codes
        )

    def test_filter_with_all_categories(self) -> None:
        pylint_codes: Set[str] = {"C0301", "R0201", "W0613", "E0602", "F0001"}
        categories: Set[PylintCategory] = {
            PylintCategory.CONVENTION,
            PylintCategory.REFACTOR,
            PylintCategory.WARNING,
            PylintCategory.ERROR,
            PylintCategory.FATAL,
        }
        expected_codes: Set[str] = {"C0301", "R0201", "W0613", "E0602", "F0001"}
        assert (
            filter_pylint_codes_by_category(pylint_codes, categories) == expected_codes
        )
