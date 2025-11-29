"""MCP server implementation for code checking functionality."""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar

import structlog

# Import all code checking modules at the top
from src.code_checker_pylint import PylintMessageType, get_pylint_prompt
from src.code_checker_pytest.reporting import create_prompt_for_failed_tests
from src.code_checker_pytest.runners import check_code_with_pytest
from src.log_utils import log_function_call
from src.utils.subprocess_runner import execute_command

# Type definitions for FastMCP
T = TypeVar("T")


class ToolDecorator(Protocol):
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]: ...


class FastMCPProtocol(Protocol):
    def tool(self) -> ToolDecorator: ...
    def run(self) -> None: ...


# Initialize loggers
logger = logging.getLogger(__name__)
structured_logger = structlog.get_logger(__name__)


class CodeCheckerServer:
    """MCP server for code checking functionality."""

    def __init__(
        self,
        project_dir: Path,
        python_executable: Optional[str] = None,
        venv_path: Optional[str] = None,
        test_folder: str = "tests",
        keep_temp_files: bool = False,
    ) -> None:
        """
        Initialize the server with the project directory and Python configuration.

        Args:
            project_dir: Path to the project directory to check
            python_executable: Optional path to Python interpreter to use for running tests. If None, defaults to sys.executable.
            venv_path: Optional path to a virtual environment to activate for running tests. When specified, the Python executable from this venv will be used instead of python_executable.
            test_folder: Path to the test folder (relative to project_dir). Defaults to 'tests'.
            keep_temp_files: Whether to keep temporary files after test execution. Useful for debugging when tests fail.
        """
        self.project_dir = project_dir
        self.python_executable = python_executable
        self.venv_path = venv_path
        self.test_folder = test_folder
        self.keep_temp_files = keep_temp_files

        # Import FastMCP
        from mcp.server.fastmcp import FastMCP

        self.mcp: FastMCPProtocol = FastMCP("Code Checker Service")
        self._register_tools()

    def _format_pylint_result(self, pylint_prompt: Optional[str]) -> str:
        """Format pylint check result."""
        if pylint_prompt is None:
            return "Pylint check completed. No issues found that require attention."
        return f"Pylint found issues that need attention:\n\n{pylint_prompt}"

    def _format_pytest_result(self, test_results: dict[str, Any]) -> str:
        """Format pytest check result."""
        if not test_results["success"]:
            return f"Error running pytest: {test_results.get('error', 'Unknown error')}"

        summary = test_results["summary"]
        failed_count = summary.get("failed", 0)
        error_count = summary.get("error", 0)
        passed_count = summary.get("passed", 0)

        if (failed_count > 0 or error_count > 0) and test_results.get("test_results"):
            failed_tests_prompt = create_prompt_for_failed_tests(
                test_results["test_results"]
            )
            return f"Pytest found issues that need attention:\n\n{failed_tests_prompt}"

        return f"Pytest check completed. All {passed_count} tests passed successfully."

    def _parse_pylint_categories(
        self, categories: Optional[List[str]]
    ) -> Optional[set[PylintMessageType]]:
        """Parse string categories into PylintMessageType enum values."""
        if not categories:
            return None

        pylint_categories = set()
        for category in categories:
            try:
                pylint_categories.add(PylintMessageType(category.lower()))
            except ValueError:
                logger.warning(f"Unknown pylint category: {category}")

        return pylint_categories if pylint_categories else None

    def _register_tools(self) -> None:
        """Register all tools with the MCP server."""

        @self.mcp.tool()
        @log_function_call
        def run_pylint_check(
            categories: Optional[List[str]] = None,
            disable_codes: Optional[List[str]] = None,
            target_directories: Optional[List[str]] = None,
        ) -> str:
            """
            Run pylint on the project code and generate smart prompts for LLMs.

            Args:
                categories: Optional list of pylint message categories to include.
                    Available categories: 'convention', 'refactor', 'warning', 'error', 'fatal'
                    Defaults to ['error', 'fatal'] if not specified.
                    Examples:
                    - ['error', 'fatal'] - Only critical issues (default)
                    - ['error', 'fatal', 'warning'] - Include warnings
                    - ['convention', 'refactor'] - Only style and refactoring suggestions
                    - ['convention', 'refactor', 'warning', 'error', 'fatal'] - All categories
                disable_codes: Optional list of pylint error codes to disable during analysis.
                    Common codes to disable include:
                    - C0114: Missing module docstring
                    - C0116: Missing function docstring
                    - C0301: Line too long
                    - W0611: Unused import
                    - W1514: Using open without explicitly specifying an encoding
                target_directories: Optional list of directories to analyze relative to project_dir.
                    Defaults to ["src"] and conditionally "tests" if it exists.
                    Examples:
                    - ["src"] - Analyze only source code
                    - ["src", "tests"] - Analyze both source and tests (default)
                    - ["mypackage", "tests"] - Custom package structure
                    - ["."] - Analyze entire project (may be slow)

            Returns:
                A string containing either pylint results or a prompt for an LLM to interpret
            """
            try:
                logger.info(
                    f"Running pylint check on project directory: {self.project_dir}"
                )
                structured_logger.info(
                    "Starting pylint check",
                    project_dir=str(self.project_dir),
                    categories=categories,
                    disable_codes=disable_codes,
                    target_directories=target_directories,
                )

                # Convert categories and run pylint
                pylint_categories = self._parse_pylint_categories(categories)
                pylint_prompt = get_pylint_prompt(
                    str(self.project_dir),
                    categories=pylint_categories,
                    disable_codes=disable_codes,
                    python_executable=self.python_executable,
                    target_directories=target_directories,
                )

                result = self._format_pylint_result(pylint_prompt)

                structured_logger.info(
                    "Pylint check completed",
                    issues_found=pylint_prompt is not None,
                    result_length=len(result),
                )

                return result

            except Exception as e:
                logger.error(f"Error running pylint check: {str(e)}")
                structured_logger.error(
                    "Pylint check failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    project_dir=str(self.project_dir),
                )
                raise

        @self.mcp.tool()
        @log_function_call
        def run_pytest_check(
            markers: Optional[List[str]] = None,
            verbosity: int = 2,
            extra_args: Optional[List[str]] = None,
            env_vars: Optional[Dict[str, str]] = None,
        ) -> str:
            """
            Run pytest on the project code and generate smart prompts for LLMs.

            Args:
                markers: Optional list of pytest markers to filter tests. Examples: ['slow', 'integration']
                verbosity: Integer for pytest verbosity level (0-3), default 2. Higher values provide more detailed output.
                extra_args: Optional list of additional pytest arguments. Examples: ['-xvs', '--no-header']
                env_vars: Optional dictionary of environment variables for the subprocess. Example: {'DEBUG': '1', 'PYTHONPATH': '/custom/path'}

            Returns:
                A string containing either pytest results or a prompt for an LLM to interpret
            """
            try:
                logger.info(
                    f"Running pytest check on project directory: {self.project_dir}"
                )
                structured_logger.info(
                    "Starting pytest check",
                    project_dir=str(self.project_dir),
                    test_folder=self.test_folder,
                    markers=markers,
                    verbosity=verbosity,
                    extra_args=extra_args,
                )

                # Run pytest
                test_results = check_code_with_pytest(
                    project_dir=str(self.project_dir),
                    test_folder=self.test_folder,
                    python_executable=self.python_executable,
                    markers=markers,
                    verbosity=verbosity,
                    extra_args=extra_args,
                    env_vars=env_vars,
                    venv_path=self.venv_path,
                    keep_temp_files=self.keep_temp_files,
                )

                result = self._format_pytest_result(test_results)

                if test_results.get("success"):
                    summary = test_results["summary"]
                    structured_logger.info(
                        "Pytest execution completed",
                        passed=summary.get("passed", 0),
                        failed=summary.get("failed", 0),
                        errors=summary.get("error", 0),
                        duration=test_results.get("duration", 0),
                    )
                else:
                    structured_logger.error(
                        "Pytest execution failed",
                        error=test_results.get("error", "Unknown error"),
                    )

                return result

            except Exception as e:
                logger.error(f"Error running pytest check: {str(e)}")
                structured_logger.error(
                    "Pytest check failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    project_dir=str(self.project_dir),
                )
                raise

        @self.mcp.tool()
        @log_function_call
        def run_all_checks(
            markers: Optional[List[str]] = None,
            verbosity: int = 2,
            extra_args: Optional[List[str]] = None,
            env_vars: Optional[Dict[str, str]] = None,
            categories: Optional[List[str]] = None,
            target_directories: Optional[List[str]] = None,
        ) -> str:
            """
            Run all code checks (pylint and pytest) and generate combined results.

            Args:
                markers: Optional list of pytest markers to filter tests. Examples: ['slow', 'integration']
                verbosity: Integer for pytest verbosity level (0-3), default 2. Higher values provide more detailed output.
                extra_args: Optional list of additional pytest arguments. Examples: ['-xvs', '--no-header']
                env_vars: Optional dictionary of environment variables for the subprocess. Example: {'DEBUG': '1', 'PYTHONPATH': '/custom/path'}
                categories: Optional list of pylint message categories to include.
                    Available categories: 'convention', 'refactor', 'warning', 'error', 'fatal'
                    Defaults to ['error', 'fatal'] if not specified.
                target_directories: Optional list of directories to analyze relative to project_dir.
                    Defaults to ["src"] and conditionally "tests" if it exists.
                    Examples:
                    - ["src"] - Analyze only source code
                    - ["src", "tests"] - Analyze both source and tests (default)
                    - ["mypackage", "tests"] - Custom package structure
                    - ["."] - Analyze entire project (may be slow)

            Returns:
                A string containing results from all checks and/or LLM prompts
            """
            try:
                logger.info(
                    f"Running all code checks on project directory: {self.project_dir}"
                )
                structured_logger.info(
                    "Starting all code checks",
                    project_dir=str(self.project_dir),
                    test_folder=self.test_folder,
                )

                # Run pylint
                pylint_categories = self._parse_pylint_categories(categories)
                pylint_prompt = get_pylint_prompt(
                    str(self.project_dir),
                    categories=pylint_categories,
                    python_executable=self.python_executable,
                    target_directories=target_directories,
                )

                # Run pytest
                test_results = check_code_with_pytest(
                    project_dir=str(self.project_dir),
                    test_folder=self.test_folder,
                    python_executable=self.python_executable,
                    markers=markers,
                    verbosity=verbosity,
                    extra_args=extra_args,
                    env_vars=env_vars,
                    venv_path=self.venv_path,
                    keep_temp_files=self.keep_temp_files,
                )

                # Format results
                pylint_result = self._format_pylint_result(pylint_prompt)
                pytest_result = self._format_pytest_result(test_results)

                # Combine results
                result = "All code checks completed:\n\n"
                result += f"1. Pylint: {pylint_result}\n\n"
                result += f"2. Pytest: {pytest_result}"

                structured_logger.info(
                    "All code checks completed",
                    pylint_issues_found=pylint_prompt is not None,
                    pytest_success=test_results.get("success", False),
                )

                return result

            except Exception as e:
                logger.error(f"Error running all code checks: {str(e)}")
                structured_logger.error(
                    "All code checks failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    project_dir=str(self.project_dir),
                )
                raise

        @self.mcp.tool()
        @log_function_call
        def second_sleep(sleep_seconds: float = 5.0) -> str:
            """
            Sleep for specified seconds using Python script.

            Args:
                sleep_seconds: Number of seconds to sleep (default: 5.0, max: 300 for safety)

            Returns:
                A string indicating the sleep operation result
            """
            # Validate input
            if not 0 <= sleep_seconds <= 300:
                raise ValueError("Sleep seconds must be between 0 and 300")

            # Check if sleep script exists
            sleep_script = self.project_dir / "tools" / "sleep_script.py"
            if not sleep_script.exists():
                raise FileNotFoundError(f"Sleep script not found: {sleep_script}")

            # Build command
            python_exe = self.python_executable or "python"
            command = [python_exe, "-u", str(sleep_script), str(sleep_seconds)]

            # Execute with timeout buffer
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            result = execute_command(
                command,
                cwd=str(self.project_dir),
                timeout_seconds=int(sleep_seconds) + 30,
                env=env,
            )

            if result.return_code == 0:
                return (
                    result.stdout.strip()
                    or f"Successfully slept for {sleep_seconds} seconds"
                )
            else:
                return f"Sleep failed (code {result.return_code}): {result.stderr}"

    @log_function_call
    def run(self) -> None:
        """Run the MCP server."""
        logger.info("Starting MCP server")
        structured_logger.info("Starting MCP server")
        self.mcp.run()


@log_function_call
def create_server(
    project_dir: Path,
    python_executable: Optional[str] = None,
    venv_path: Optional[str] = None,
    test_folder: str = "tests",
    keep_temp_files: bool = False,
) -> CodeCheckerServer:
    """
    Create a new CodeCheckerServer instance.

    Args:
        project_dir: Path to the project directory to check
        python_executable: Optional path to Python interpreter to use for running tests. If None, defaults to sys.executable.
        venv_path: Optional path to a virtual environment to activate for running tests. When specified, the Python executable from this venv will be used instead of python_executable.
        test_folder: Path to the test folder (relative to project_dir). Defaults to 'tests'.
        keep_temp_files: Whether to keep temporary files after test execution. Useful for debugging when tests fail.

    Returns:
        A new CodeCheckerServer instance
    """
    return CodeCheckerServer(
        project_dir,
        python_executable=python_executable,
        venv_path=venv_path,
        test_folder=test_folder,
        keep_temp_files=keep_temp_files,
    )
