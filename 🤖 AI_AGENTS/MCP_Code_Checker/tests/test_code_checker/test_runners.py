"""
Tests for the code_checker_pytest runner functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.code_checker_pytest import PytestReport, check_code_with_pytest, run_tests
from tests.test_code_checker.test_code_checker_pytest_common import (
    _cleanup_test_project,
    _create_test_project,
)


def test_run_tests() -> None:
    """Integration test for run_tests function with a sample project."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        _create_test_project(test_dir)

        try:
            # Run the tests and parse the report
            result = run_tests(str(test_dir), "tests")
            assert isinstance(result, PytestReport)

            # Check the summary
            assert result.summary.total == 2
            assert result.summary.passed == 1
            assert result.summary.failed == 1
            assert result.summary.collected == 2

            # Make sure tests are not None before accessing
            assert result.tests is not None

            # Find the passing and failing tests
            passing_test = next(
                (t for t in result.tests if t.nodeid.endswith("::test_passing")), None
            )
            failing_test = next(
                (t for t in result.tests if t.nodeid.endswith("::test_failing")), None
            )

            # Assert the passing test
            assert passing_test is not None
            assert passing_test.outcome == "passed"
            assert passing_test.call is not None
            assert passing_test.call.outcome == "passed"

            # Assert the failing test
            assert failing_test is not None
            assert failing_test.outcome == "failed"
            assert failing_test.call is not None
            assert failing_test.call.outcome == "failed"
            assert failing_test.call.crash is not None
            assert "assert 1 == 2" in failing_test.call.crash.message
        finally:
            _cleanup_test_project(test_dir)


def test_run_tests_with_custom_parameters() -> None:
    """Test run_tests function with custom parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        _create_test_project(test_dir)

        # Create a test with a custom marker
        with open(test_dir / "tests" / "test_marked.py", "w") as f:
            f.write(
                """
import pytest

@pytest.mark.slow
def test_slow():
    assert True
"""
            )

        try:
            # Run tests with markers filter
            result = run_tests(
                str(test_dir),
                "tests",
                markers=["slow"],
                verbosity=3,
                keep_temp_files=True,
            )

            assert isinstance(result, PytestReport)

            # Only the marked test should be run
            assert result.summary.total == 1
            assert result.summary.deselected == 2
            assert result.summary.total == 1
            assert result.summary.passed == 1

            # Check environment context was captured
            assert result.environment_context is not None
        finally:
            _cleanup_test_project(test_dir)


def test_run_tests_no_tests_found() -> None:
    """Test the run_tests function when no tests are found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        (test_dir / "src").mkdir(parents=True, exist_ok=True)
        (test_dir / "tests").mkdir(parents=True, exist_ok=True)

        with open(test_dir / "src" / "__init__.py", "w") as f:
            f.write("")
        with open(test_dir / "tests" / "__init__.py", "w") as f:
            f.write("")

        try:
            with pytest.raises(
                Exception, match="No Tests Found: Pytest did not find any tests to run."
            ):
                run_tests(str(test_dir), "tests")
        finally:
            _cleanup_test_project(test_dir)


@patch("src.code_checker_pytest.runners.run_tests")
def test_check_code_with_pytest(mock_run_tests: MagicMock) -> None:
    """Test the full check_code_with_pytest function."""
    # Create a mock Summary instance for our mock report
    mock_summary = MagicMock()
    mock_summary.collected = 2
    mock_summary.total = 2
    mock_summary.passed = 2
    mock_summary.failed = 0
    mock_summary.error = 0
    mock_summary.skipped = 0
    mock_summary.xfailed = 0
    mock_summary.xpassed = 0
    mock_summary.deselected = 0

    # Mock the run_tests function to return a test report
    mock_report = PytestReport(
        created=1678972345.123,
        duration=0.5,
        exitcode=0,
        root="/path/to/project",
        environment={"Python": "3.9.0"},
        summary=mock_summary,
    )
    mock_run_tests.return_value = mock_report

    # Call the function
    result = check_code_with_pytest("/test/project")

    # Verify the results
    assert result["success"] is True
    assert "summary" in result
    assert "Collected 2 tests" in result["summary"]
    assert "✅ Passed: 2" in result["summary"]
    assert result["failed_tests_prompt"] is None
    assert result["test_results"] == mock_report


@patch("src.code_checker_pytest.runners.run_tests")
def test_check_code_with_pytest_with_custom_parameters(
    mock_run_tests: MagicMock,
) -> None:
    """Test check_code_with_pytest with custom parameters."""
    # Create a mock Summary with all necessary attributes
    mock_summary = MagicMock()
    mock_summary.collected = 1
    mock_summary.total = 1
    mock_summary.passed = 1
    mock_summary.failed = 0
    mock_summary.error = 0
    mock_summary.skipped = 0
    mock_summary.xfailed = 0
    mock_summary.xpassed = 0
    mock_summary.deselected = 0

    # Mock the run_tests function
    mock_report = PytestReport(
        created=1678972345.123,
        duration=0.5,
        exitcode=0,
        root="/path/to/project",
        environment={"Python": "3.9.0"},
        summary=mock_summary,
    )
    mock_run_tests.return_value = mock_report

    # Set up mock report properties for a success result
    mock_report.error_context = None
    mock_report.tests = []
    mock_run_tests.return_value = mock_report

    # Custom parameters to test
    custom_env = {"PYTHONPATH": "/custom/path"}
    extra_args = ["--no-header"]

    # Call the function with custom parameters
    result = check_code_with_pytest(
        project_dir="/test/project",
        test_folder="custom_tests",
        markers=["slow"],
        verbosity=3,
        extra_args=extra_args,
        env_vars=custom_env,
        keep_temp_files=True,
    )

    # Verify run_tests was called with the correct parameters
    mock_run_tests.assert_called_once_with(
        "/test/project",
        "custom_tests",
        None,  # python_executable
        ["slow"],  # markers
        3,  # verbosity
        extra_args,
        custom_env,
        None,  # venv_path
        True,  # keep_temp_files
    )

    # Verify result is correct
    assert result["success"] is True


@patch("src.code_checker_pytest.runners.run_tests")
def test_check_code_with_pytest_with_failed_tests(mock_run_tests: MagicMock) -> None:
    """Test check_code_with_pytest with failed tests."""
    # Create a mock Summary instance for our mock report
    mock_summary = MagicMock()
    mock_summary.collected = 4
    mock_summary.total = 4
    mock_summary.passed = 2
    mock_summary.failed = 2
    mock_summary.error = 0
    mock_summary.skipped = 0
    mock_summary.xfailed = 0
    mock_summary.xpassed = 0
    mock_summary.deselected = 0

    # Create a mock test stage with failed outcome
    mock_call = MagicMock()
    mock_call.outcome = "failed"
    mock_call.crash = MagicMock()
    mock_call.crash.message = "AssertionError: assert 1 == 2"
    mock_call.traceback = [MagicMock()]
    mock_call.stdout = "Test output"
    mock_call.stderr = "Test error"
    mock_call.longrepr = "Failure representation"

    # Create mock tests
    mock_test = MagicMock()
    mock_test.nodeid = "test_file.py::test_failing"
    mock_test.outcome = "failed"
    mock_test.call = mock_call

    # Set up the mock report
    mock_report = PytestReport(
        created=1678972345.123,
        duration=0.5,
        exitcode=1,
        root="/path/to/project",
        environment={"Python": "3.9.0"},
        summary=mock_summary,
        tests=[mock_test],
    )
    mock_run_tests.return_value = mock_report

    # Call the function
    result = check_code_with_pytest("/test/project")

    # Verify the results
    assert result["success"] is True
    assert "summary" in result
    assert "Collected 4 tests" in result["summary"]
    assert "✅ Passed: 2" in result["summary"]
    assert "❌ Failed: 2" in result["summary"]
    assert result["failed_tests_prompt"] is not None
    assert "test_file.py::test_failing" in result["failed_tests_prompt"]
    assert result["test_results"] == mock_report


@patch("src.code_checker_pytest.runners.run_tests")
def test_check_code_with_pytest_with_error(mock_run_tests: MagicMock) -> None:
    """Test check_code_with_pytest with an error during test execution."""
    # Make run_tests raise an exception
    mock_run_tests.side_effect = Exception("Test execution error")

    # Call the function
    result = check_code_with_pytest("/test/project")

    # Verify the results
    assert result["success"] is False
    assert "error" in result
    assert result["error"] == "Test execution error"
