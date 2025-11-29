"""
Functions for running pylint analysis and processing results.
"""

import logging
import os
import sys
from typing import List, Optional, Set

import structlog

from src.log_utils import log_function_call
from src.utils.subprocess_runner import execute_command

from .models import DEFAULT_CATEGORIES, PylintMessageType, PylintResult
from .parsers import parse_pylint_json_output

logger = logging.getLogger(__name__)
structured_logger = structlog.get_logger(__name__)


@log_function_call
def get_pylint_results(
    project_dir: str,
    disable_codes: Optional[List[str]] = None,
    python_executable: Optional[str] = None,
    target_directories: Optional[List[str]] = None,
) -> PylintResult:
    """
    Runs pylint on the specified project directory and returns the results.

    Args:
        project_dir: The path to the project directory.
        disable_codes: List of pylint codes to disable during analysis. Common codes include:
            - C0114: Missing module docstring
            - C0116: Missing function docstring
            - C0301: Line too long
            - C0303: Trailing whitespace
            - C0305: Trailing newlines
            - W0311: Bad indentation
            - W0611: Unused import
            - W1514: Unspecified encoding
        python_executable: Path to Python executable to use for running pylint. Defaults to sys.executable if None.
        target_directories: List of directories to analyze relative to project_dir.
            Defaults to ["src"] and conditionally "tests" if it exists.
            Examples: ["src"], ["src", "tests"], ["mypackage", "tests"], ["."]

    Returns:
        A PylintResult object containing the results of the pylint run.

    Raises:
        FileNotFoundError: If the project directory does not exist.
    """
    if not os.path.isdir(project_dir):
        raise FileNotFoundError(f"Project directory not found: {project_dir}")

    # Set default target directories if none provided
    if target_directories is None:
        target_directories = ["src"]
        if os.path.exists(os.path.join(project_dir, "tests")):
            target_directories.append("tests")

    # Validate that target directories exist
    valid_directories = []
    for directory in target_directories:
        full_path = os.path.join(project_dir, directory)
        if os.path.exists(full_path):
            valid_directories.append(directory)
        else:
            structured_logger.warning(
                "Target directory does not exist, skipping",
                directory=directory,
                full_path=full_path,
            )

    if not valid_directories:
        error_message = (
            f"No valid target directories found. Checked: {target_directories}"
        )
        structured_logger.error(
            "No valid directories to analyze",
            target_directories=target_directories,
            project_dir=project_dir,
        )
        return PylintResult(
            return_code=1,
            messages=[],
            error=error_message,
            raw_output=None,
        )

    structured_logger.info(
        "Starting pylint analysis",
        project_dir=project_dir,
        disable_codes=disable_codes,
        target_directories=valid_directories,
    )

    # Determine the Python executable from the parameter or fall back to sys.executable
    python_exe = python_executable if python_executable is not None else sys.executable

    # Construct the pylint command
    pylint_command = [
        python_exe,
        "-m",
        "pylint",
        "--output-format=json",
    ]

    if disable_codes and len(disable_codes) > 0:
        pylint_command.append(f"--disable={','.join(disable_codes)}")

    # Add all valid target directories
    pylint_command.extend(valid_directories)

    # Execute the subprocess
    subprocess_result = execute_command(
        command=pylint_command, cwd=project_dir, timeout_seconds=120
    )

    # Handle subprocess execution errors
    if subprocess_result.execution_error:
        return PylintResult(
            return_code=subprocess_result.return_code,
            messages=[],
            error=subprocess_result.execution_error,
            raw_output=None,
        )

    if subprocess_result.timed_out:
        return PylintResult(
            return_code=1,
            messages=[],
            error="Pylint execution timed out after 120 seconds",
            raw_output=None,
        )

    # Log subprocess results
    structured_logger.info(
        "Pylint subprocess completed",
        return_code=subprocess_result.return_code,
        has_stdout=bool(subprocess_result.stdout),
        has_stderr=bool(subprocess_result.stderr),
        stdout_empty=not subprocess_result.stdout
        or subprocess_result.stdout.strip() == "",
        stderr_empty=not subprocess_result.stderr
        or subprocess_result.stderr.strip() == "",
        command_executed=" ".join(pylint_command),
    )

    raw_output = subprocess_result.stdout

    # Parse pylint output from JSON
    messages, parse_error = parse_pylint_json_output(raw_output)

    if parse_error:
        return PylintResult(
            return_code=subprocess_result.return_code,
            messages=[],
            error=parse_error,
            raw_output=raw_output,
        )

    result = PylintResult(
        return_code=subprocess_result.return_code,
        messages=messages,
        raw_output=raw_output,
    )

    structured_logger.info(
        "Pylint analysis completed",
        return_code=subprocess_result.return_code,
        messages_count=len(messages),
        unique_codes=len(result.get_message_ids()),
    )

    return result


@log_function_call
def run_pylint_check(
    project_dir: str,
    categories: Optional[Set[PylintMessageType]] = None,
    disable_codes: Optional[List[str]] = None,
    python_executable: Optional[str] = None,
    target_directories: Optional[List[str]] = None,
) -> PylintResult:
    """
    Run pylint check on a project directory and returns the result.

    Args:
        project_dir: The path to the project directory to analyze.
        categories: Set of specific pylint categories to filter by. Available categories are:
            - PylintMessageType.CONVENTION: Style conventions (C)
            - PylintMessageType.REFACTOR: Refactoring suggestions (R)
            - PylintMessageType.WARNING: Python-specific warnings (W)
            - PylintMessageType.ERROR: Probable bugs in the code (E)
            - PylintMessageType.FATAL: Critical errors that prevent pylint from working (F)
            Defaults to {ERROR, FATAL} if None.
        disable_codes: Optional list of pylint codes to disable during analysis. Common codes include:
            - C0114: Missing module docstring
            - C0116: Missing function docstring
            - C0301: Line too long
            - C0303: Trailing whitespace
            - C0305: Trailing newlines
            - W0311: Bad indentation
            - W0611: Unused import
            - W1514: Unspecified encoding
        python_executable: Optional path to Python interpreter to use for running tests. If None, defaults to sys.executable.
        target_directories: Optional list of directories to analyze relative to project_dir.
            Defaults to ["src"] and conditionally "tests" if it exists.
            Examples: ["src"], ["src", "tests"], ["mypackage", "tests"], ["."]

    Returns:
        PylintResult with the analysis outcome.
    """
    # Default disable codes if none provided
    if disable_codes is None:
        disable_codes = [
            # not required for now
            "C0114",  # doc missing
            "C0116",  # doc missing
            #
            # can be solved with formatting / black
            "C0301",  # line-too-long
            "C0303",  # trailing-whitespace
            "C0305",  # trailing-newlines
            "W0311",  # bad-indentation   - instruction available
            #
            # can be solved with iSort
            "W0611",  # unused-import
            "W1514",  # unspecified-encoding
        ]

    return get_pylint_results(
        project_dir,
        disable_codes=disable_codes,
        python_executable=python_executable,
        target_directories=target_directories,
    )
