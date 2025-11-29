"""
Functions for running pytest tests and processing results.
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

import structlog

from src.log_utils import log_function_call

from .models import PytestReport
from .parsers import parse_pytest_report
from .reporting import create_prompt_for_failed_tests, get_test_summary
from .utils import collect_environment_info, create_error_context, read_file

logger = logging.getLogger(__name__)
structured_logger = structlog.get_logger(__name__)


@log_function_call
def run_tests(
    project_dir: str,
    test_folder: str,
    python_executable: Optional[str] = None,
    markers: Optional[List[str]] = None,
    verbosity: int = 2,
    extra_args: Optional[List[str]] = None,
    env_vars: Optional[Dict[str, str]] = None,
    venv_path: Optional[str] = None,
    keep_temp_files: bool = False,
) -> PytestReport:
    """
    Run pytest tests in the specified project directory and test folder and returns the results.

    Args:
        project_dir: The path to the project directory
        test_folder: The path to the folder containing the tests relative to the project directory
        python_executable: Optional path to Python interpreter to use. Defaults to sys.executable if not provided
        markers: Optional list of pytest markers to filter tests. Examples: ['slow', 'integration', 'unit']
        verbosity: Integer for pytest verbosity level (0-3). Default is 2. Higher values provide more detailed output
        extra_args: Optional list of additional pytest arguments. Examples: ['-xvs', '--no-header', '--durations=10']
        env_vars: Optional dictionary of environment variables to set for the subprocess. Example: {'DEBUG': '1'}
        venv_path: Optional path to a virtual environment to activate. When provided, this venv's Python will be used
        keep_temp_files: Whether to keep temporary files after execution (useful for debugging failures)


    Returns:
        PytestReport: An object containing the results of the test session with the following attributes:
        - summary: Summary statistics of the test run (passed, failed, skipped counts)
        - test_results: List of individual test results with detailed information
        - environment_context: Information about the test environment
        - error_context: Information about any errors that occurred during execution

    Raises:
        Exception: If pytest is not installed or if an error occurs during test execution
    """

    # Create a temporary directory for output files
    temp_dir = tempfile.mkdtemp(prefix="pytest_runner_")
    temp_report_file = os.path.join(temp_dir, "pytest_result.json")

    structured_logger.info(
        "Starting pytest execution",
        project_dir=project_dir,
        test_folder=test_folder,
        markers=markers,
        verbosity=verbosity,
        venv_path=venv_path,
    )

    try:
        # Determine Python executable
        py_executable = python_executable

        # Handle virtual environment activation
        if venv_path:
            if not os.path.exists(venv_path):
                raise Exception(f"Virtual environment path does not exist: {venv_path}")

            # Locate the Python executable in the virtual environment
            if os.name == "nt":  # Windows
                venv_python = os.path.join(venv_path, "Scripts", "python.exe")
            else:  # Unix-like systems
                venv_python = os.path.join(venv_path, "bin", "python")

            if not os.path.exists(venv_python):
                raise Exception(
                    f"Python executable not found in virtual environment: {venv_python}"
                )

            py_executable = venv_python

        # If no executable is specified (either directly or via venv), use the current one
        if not py_executable:
            py_executable = sys.executable

        # Construct the pytest command
        command = [
            py_executable,
            "-m",
            "pytest",
        ]

        # Add verbosity flags based on level
        if verbosity > 0:
            verbosity_flag = "-" + "v" * min(verbosity, 3)  # -v, -vv, or -vvv
            command.append(verbosity_flag)

        # Add markers if provided
        if markers and len(markers) > 0:
            if len(markers) == 1:
                command.extend(["-m", markers[0]])
            else:
                # Combine multiple markers with "and"
                command.extend(["-m", " and ".join(markers)])

        # Add rootdir and json-report options
        command.extend(
            [
                "--rootdir",
                project_dir,
                "--json-report",
                f"--json-report-file={temp_report_file}",
            ]
        )

        # Add any extra arguments
        if extra_args:
            command.extend(extra_args)

        # Add the test folder path
        command.append(os.path.join(project_dir, test_folder))

        logger.debug(f"Running command: {' '.join(command)}")

        # Prepare environment variables
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        # If using a virtual environment, adjust PATH to prioritize it
        if venv_path:
            if os.name == "nt":  # Windows
                venv_bin = os.path.join(venv_path, "Scripts")
            else:  # Unix-like systems
                venv_bin = os.path.join(venv_path, "bin")

            env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"

        # Collect environment info before running the tests
        environment_context = collect_environment_info(command)

        try:
            # Print command for debugging
            print(f"Running command: {' '.join(command)}")

            # Run with timeout to prevent hanging
            try:
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    cwd=project_dir,
                    env=env,
                    timeout=30,  # 30 second timeout
                )
                print(f"Command completed with return code: {process.returncode}")
            except subprocess.TimeoutExpired as e:
                print(f"Command timed out after 30 seconds: {' '.join(command)}")
                raise Exception(f"Subprocess timed out: {' '.join(command)}") from e

            output = process.stdout
            error_output = process.stderr
            combined_output = f"{output}\n{error_output}"
            logger.debug(output)

            # Check if plugin is missing
            if (
                "no plugin named 'json-report'" in combined_output.lower()
                or "no module named 'pytest_json_report'" in combined_output.lower()
            ):
                print(
                    "pytest-json-report plugin not found, attempting to install it..."
                )
                try:
                    subprocess.run(
                        [py_executable, "-m", "pip", "install", "pytest-json-report"],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=60,  # Give it time to install
                    )
                    print("Installed pytest-json-report, retrying...")
                    # Retry the command
                    process = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        check=False,
                        cwd=project_dir,
                        env=env,
                        timeout=30,
                    )
                    output = process.stdout
                    error_output = process.stderr
                    combined_output = f"{output}\n{error_output}"
                except subprocess.CalledProcessError as install_error:
                    print(f"Failed to install pytest-json-report: {install_error}")
                    raise Exception(
                        "Failed to install the required pytest-json-report plugin"
                    )
                except subprocess.TimeoutExpired:
                    print("Installation or retry timed out")
                    raise Exception(
                        "Timed out while installing pytest-json-report or retrying the test"
                    )

            # Check specifically for 'no tests found' case
            if "collected 0 items" in combined_output or process.returncode == 5:
                print("No tests found, raising specific exception")
                raise Exception("No Tests Found: Pytest did not find any tests to run.")

            # Create error context if needed
            error_context = None
            if process.returncode != 0:
                error_context = create_error_context(
                    process.returncode, combined_output
                )

            # Always continue on collection errors but log warnings
            report_exists = os.path.isfile(temp_report_file)
            if (process.returncode in [1, 2, 5]) and not report_exists:
                error_details = (
                    error_context.error_message if error_context else combined_output
                )
                # Log warning but continue execution
                logger.warning(
                    f"Test collection error occurred (code {process.returncode}), "
                    f"but continuing execution: {error_details}"
                )

            # Handle other error cases
            elif process.returncode == 3:
                print(combined_output)
                raise Exception(
                    f"Internal Error: {error_context.exit_code_meaning if error_context else 'Pytest encountered an internal error'}. "
                    f"Suggestion: {error_context.suggestion if error_context else 'Check pytest version compatibility'}"
                )
            elif process.returncode == 4:
                print(combined_output)
                raise Exception(
                    f"Usage Error: {error_context.exit_code_meaning if error_context else 'Pytest was used incorrectly'}. "
                    f"Suggestion: {error_context.suggestion if error_context else 'Verify command-line arguments'}"
                )
            elif process.returncode == 5 and report_exists:
                # Continue if we have a report file but no tests were found
                logger.warning(
                    "No tests were found, but report file was generated. Continuing with processing."
                )
            elif process.returncode > 5:
                # Handle plugin-specific exit codes
                print(combined_output)
                raise Exception(
                    f"Plugin Error: {error_context.exit_code_meaning if error_context else f'Pytest plugin returned exit code {process.returncode}'}. "
                    f"Suggestion: {error_context.suggestion if error_context else 'Check plugin documentation'}"
                )

            # Final check to ensure we have a report file
            if not report_exists:
                print(combined_output)
                if "collected 0 items" in combined_output:
                    raise Exception(
                        "No Tests Found: Pytest did not find any tests to run."
                    )
                else:
                    raise Exception(
                        "Test execution completed but no report file was generated. "
                        "Check for configuration errors in pytest.ini or pytest plugins."
                    )

            file_contents = read_file(temp_report_file)
            parsed_results = parse_pytest_report(file_contents)

            # Add environment and error context to the results
            parsed_results.environment_context = environment_context
            parsed_results.error_context = error_context

            structured_logger.info(
                "Pytest execution completed successfully",
                passed=parsed_results.summary.passed,
                failed=parsed_results.summary.failed,
                errors=parsed_results.summary.error,
                skipped=parsed_results.summary.skipped,
                duration=parsed_results.duration,
            )

            return parsed_results

        except Exception as e:
            command_line = " ".join(command)
            structured_logger.error(
                "Pytest execution failed",
                error=str(e),
                error_type=type(e).__name__,
                project_dir=project_dir,
                command=command_line,
            )
            print(
                f"""Error during pytest execution:
- folder {project_dir}
- {command_line}"""
            )
            raise e

    except Exception as e:
        raise e
    finally:
        # Clean up temporary files unless keep_temp_files is True
        if not keep_temp_files and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to clean up temporary directory: {cleanup_error}"
                )


@log_function_call
def check_code_with_pytest(
    project_dir: str,
    test_folder: str = "tests",
    python_executable: Optional[str] = None,
    markers: Optional[List[str]] = None,
    verbosity: int = 2,
    extra_args: Optional[List[str]] = None,
    env_vars: Optional[Dict[str, str]] = None,
    venv_path: Optional[str] = None,
    keep_temp_files: bool = False,
) -> Dict[str, Any]:
    """
    Run pytest on the specified project and return results.

    Args:
        project_dir: Path to the project directory
        test_folder: Path to the test folder (relative to project_dir). Defaults to 'tests'
        python_executable: Optional path to Python interpreter to use for running tests. If None, defaults to sys.executable
        markers: Optional list of pytest markers to filter tests. Examples: ['slow', 'integration', 'unit']
        verbosity: Integer for pytest verbosity level (0-3), default 2. Higher values provide more detailed output
        extra_args: Optional list of additional pytest arguments. Examples: ['-xvs', '--no-header']
        env_vars: Optional dictionary of environment variables for the subprocess. Example: {'DEBUG': '1', 'PYTHONPATH': '/custom/path'}
        venv_path: Optional path to a virtual environment to activate for running tests. When specified, the Python executable from this venv will be used instead of python_executable
        keep_temp_files: Whether to keep temporary files after test execution. Useful for debugging when tests fail


    Returns:
        Dictionary with test results containing the following keys:
        - success: Boolean indicating if the test execution was successful
        - summary: Summary of test results as a formatted string
        - failed_tests_prompt: Formatted prompt for failed tests (if any)
        - test_results: Complete PytestReport object with detailed test information
        - environment_info: Information about the test environment (Python version, pytest version, etc.)
        - error_info: Details about any errors that occurred during test execution
    """
    structured_logger.info(
        "Starting pytest code check",
        project_dir=project_dir,
        test_folder=test_folder,
        markers=markers,
        verbosity=verbosity,
    )

    try:
        test_results = run_tests(
            project_dir,
            test_folder,
            python_executable,
            markers,
            verbosity,
            extra_args,
            env_vars,
            venv_path,
            keep_temp_files,
        )

        # CRITICAL FIX: Use get_test_summary to get formatted string, not a dictionary
        summary_text = get_test_summary(test_results)

        structured_logger.info(
            "Pytest code check completed",
            passed=test_results.summary.passed,
            failed=test_results.summary.failed,
            errors=test_results.summary.error,
            skipped=test_results.summary.skipped,
        )

        failed_tests_prompt = None
        if (test_results.summary.failed and test_results.summary.failed > 0) or (
            test_results.summary.error and test_results.summary.error > 0
        ):
            failed_tests_prompt = create_prompt_for_failed_tests(test_results)

        environment_info = None
        if test_results.environment_context:
            environment_info = {
                "python_version": test_results.environment_context.python_version,
                "pytest_version": test_results.environment_context.pytest_version,
                "platform": test_results.environment_context.platform_info,
                "plugins": test_results.environment_context.loaded_plugins,
                "working_directory": test_results.environment_context.working_directory,
                "system_info": {
                    "cpu": test_results.environment_context.cpu_info,
                    "memory": test_results.environment_context.memory_info,
                },
            }

        error_info = None
        if test_results.error_context:
            error_info = {
                "exit_code": test_results.error_context.exit_code,
                "meaning": test_results.error_context.exit_code_meaning,
                "suggestion": test_results.error_context.suggestion,
                "collection_errors": test_results.error_context.collection_errors,
            }

        # CRITICAL FIX: Return summary_text (string) not dictionary
        return {
            "success": True,
            "summary": summary_text,  # This is the formatted string from get_test_summary()
            "failed_tests_prompt": failed_tests_prompt,
            "test_results": test_results,
            "environment_info": environment_info,
            "error_info": error_info,
        }

    except Exception as e:
        structured_logger.error(
            "Pytest code check failed",
            error=str(e),
            error_type=type(e).__name__,
            project_dir=project_dir,
            test_folder=test_folder,
        )
        return {"success": False, "error": str(e)}
