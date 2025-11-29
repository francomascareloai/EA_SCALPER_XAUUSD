"""
Tests for the server functionality with updated parameter exposure.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture  # type: ignore[misc]
def mock_project_dir() -> Path:
    """Return a mock project directory path."""
    return Path("/fake/project/dir")


@pytest.mark.asyncio
@patch("src.code_checker_pytest.runners.check_code_with_pytest")
@patch("mcp.server.fastmcp.FastMCP")
async def test_run_pytest_check_parameters(
    mock_fastmcp: MagicMock, mock_check_pytest: MagicMock, mock_project_dir: Path
) -> None:
    """Test that run_pytest_check properly uses server parameters and passes parameters correctly."""
    from src.server import CodeCheckerServer

    # Setup mocks
    mock_tool = MagicMock()
    mock_fastmcp.return_value.tool.return_value = mock_tool

    # Setup mock result
    mock_check_pytest.return_value = {
        "success": True,
        "summary": {"passed": 5, "failed": 0, "error": 0},
        "test_results": MagicMock(),
    }

    # Create server with the static parameters
    CodeCheckerServer(
        mock_project_dir, test_folder="custom_tests", keep_temp_files=True
    )

    # Get the run_pytest_check function (it's decorated by mock_tool)
    run_pytest_check = mock_tool.call_args_list[1][0][0]

    # Call with only the dynamic parameters (without test_folder and keep_temp_files)
    run_pytest_check(
        markers=["slow", "integration"],
        verbosity=3,
        extra_args=["--no-header"],
        env_vars={"TEST_ENV": "value"},
    )

    # Verify check_code_with_pytest was called with correct parameters
    # test_folder and keep_temp_files should come from the server instance
    mock_check_pytest.assert_called_once_with(
        project_dir=str(mock_project_dir),
        test_folder="custom_tests",  # From server constructor
        python_executable=None,
        markers=["slow", "integration"],
        verbosity=3,
        extra_args=["--no-header"],
        env_vars={"TEST_ENV": "value"},
        venv_path=None,
        keep_temp_files=True,  # From server constructor
    )


@pytest.mark.asyncio
@patch("mcp.server.fastmcp.FastMCP")
async def test_run_all_checks_parameters(
    mock_fastmcp: MagicMock,
    mock_project_dir: Path,
) -> None:
    """Test that run_all_checks properly uses server parameters and passes parameters correctly."""
    from src.server import CodeCheckerServer

    # Setup mocks
    mock_tool = MagicMock()
    mock_fastmcp.return_value.tool.return_value = mock_tool

    # Create server with the static parameters
    with (
        patch("src.server.get_pylint_prompt") as mock_pylint,
        patch("src.server.check_code_with_pytest") as mock_check_pytest,
    ):
        # Setup mock results
        mock_pylint.return_value = None
        mock_check_pytest.return_value = {
            "success": True,
            "summary": {"passed": 5, "failed": 0, "error": 0},
            "test_results": MagicMock(),
        }

        server = CodeCheckerServer(
            mock_project_dir, test_folder="custom_tests", keep_temp_files=True
        )

        # Get the run_all_checks function (it's decorated by mock_tool)
        run_all_checks = mock_tool.call_args_list[2][0][0]

        # Call with only the dynamic parameters (without test_folder and keep_temp_files)
        # The function needs to be invoked to trigger the actual checks
        result = run_all_checks(
            markers=["slow", "integration"],
            verbosity=3,
            extra_args=["--no-header"],
            env_vars={"TEST_ENV": "value"},
            categories={"ERROR"},  # Updated from pylint_categories to categories
        )

        # Verify check_code_with_pytest was called with correct parameters
        # test_folder and keep_temp_files should come from the server instance
        mock_check_pytest.assert_called_once_with(
            project_dir=str(mock_project_dir),
            test_folder="custom_tests",  # From server constructor
            python_executable=None,
            markers=["slow", "integration"],
            verbosity=3,
            extra_args=["--no-header"],
            env_vars={"TEST_ENV": "value"},
            venv_path=None,
            keep_temp_files=True,  # From server constructor
        )
