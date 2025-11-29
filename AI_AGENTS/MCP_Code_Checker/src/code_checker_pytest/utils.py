"""
Utility functions for code checker pytest operations.
"""

import json
import os
import platform
import subprocess
import sys
from typing import List, Tuple

from .models import EnvironmentContext, ErrorContext


def read_file(file_path: str) -> str:
    """
    Read the contents of a file.

    Args:
        file_path: Path to the file to read

    Returns:
        The contents of the file as a string

    Raises:
        FileNotFoundError: If the file does not exist
        PermissionError: If access to the file is denied
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()


def get_pytest_exit_code_info(exit_code: int) -> Tuple[str, str]:
    """
    Get detailed information and suggestions for pytest exit codes.

    Args:
        exit_code: The pytest exit code

    Returns:
        Tuple containing (meaning, suggestion)
    """
    exit_code_map = {
        0: (
            "All tests passed successfully",
            "No action needed.",
        ),
        1: (
            "Tests were collected and run but some tests failed",
            "Review the test failures and fix the issues in your code.",
        ),
        2: (
            "Test execution was interrupted by the user",
            "Re-run tests when ready.",
        ),
        3: (
            "Internal pytest error",
            "Check for pytest version compatibility issues or look for bugs in pytest plugins.",
        ),
        4: (
            "pytest command line usage error",
            "Verify your pytest command arguments and fix any syntax errors.",
        ),
        5: (
            "No tests were collected",
            "Check your test file naming patterns, verify imports, and ensure tests are properly defined.",
        ),
        # Custom exit codes for pytest plugins
        6: (
            "Coverage threshold not met (pytest-cov plugin)",
            "Increase test coverage to meet the defined threshold.",
        ),
        7: (
            "Doctests failed (pytest-doctests plugin)",
            "Fix issues in your doctest examples.",
        ),
        8: (
            "Benchmark regression detected (pytest-benchmark plugin)",
            "Performance has degraded from baseline, check recent code changes.",
        ),
        # Default for unknown exit codes
    }

    # Return the mapping or a default message if exit code is unknown
    info = exit_code_map.get(
        exit_code,
        (
            f"Unknown exit code {exit_code}",
            "Check pytest documentation for this exit code or review the log for specific error messages.",
        ),
    )
    return info


def collect_environment_info(command: List[str]) -> EnvironmentContext:
    """
    Collect detailed information about the test environment.

    Args:
        command: The pytest command used to run tests

    Returns:
        EnvironmentContext object with environment details
    """
    # Get Python version information
    python_version = f"{platform.python_implementation()} {platform.python_version()}"

    # Get pytest version
    try:
        pytest_version_output = subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        pytest_version = pytest_version_output.stdout.strip()
    except Exception:
        pytest_version = "Unknown"

    # Platform information
    platform_info = f"{platform.system()} {platform.release()} {platform.machine()}"

    # Get installed packages
    installed_packages = []
    try:
        pip_list_output = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=False,
        )
        if pip_list_output.returncode == 0:
            installed_packages = json.loads(pip_list_output.stdout)
    except Exception:
        pass  # Silently fail if pip list cannot be executed

    # Get list of loaded pytest plugins
    loaded_plugins = []
    try:
        # Add timeout to prevent hanging
        print("Getting pytest plugins info...")
        pytest_plugins_output = subprocess.run(
            [sys.executable, "-m", "pytest", "--trace-config"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,  # 10 second timeout
        )
        print(
            f"Plugins info command completed with return code: {pytest_plugins_output.returncode}"
        )

        # Extract plugin names from output
        for line in pytest_plugins_output.stderr.split("\n"):
            if "pluginmanager" in line and "registered" in line:
                parts = line.split("registered:")
                if len(parts) > 1:
                    plugin_name = parts[1].strip()
                    loaded_plugins.append(plugin_name)
    except subprocess.TimeoutExpired:
        print("Timed out while trying to get pytest plugins")
        loaded_plugins = ["Plugin detection timed out"]
    except Exception as e:
        print(f"Error getting pytest plugins: {e}")
        # Silently fail if plugin discovery fails

    # CPU information (if available)
    cpu_info = None
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                cpu_info_list = [line for line in f if "model name" in line]
                if cpu_info_list:
                    cpu_info = cpu_info_list[0].split(":")[1].strip()
        elif platform.system() == "Darwin":  # macOS
            cpu_info_output = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
            )
            if cpu_info_output.returncode == 0:
                cpu_info = cpu_info_output.stdout.strip()
        elif platform.system() == "Windows":
            cpu_info_output = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True,
                text=True,
                check=False,
            )
            if cpu_info_output.returncode == 0:
                lines = cpu_info_output.stdout.strip().split("\n")
                if len(lines) > 1:
                    cpu_info = lines[1].strip()
    except Exception:
        pass  # Silently fail if CPU info cannot be obtained

    # Memory information (if available)
    memory_info = None
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                mem_info_list = [line for line in f if "MemTotal" in line]
                if mem_info_list:
                    memory_info = mem_info_list[0].strip()
        elif platform.system() == "Darwin":  # macOS
            memory_output = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=False,
            )
            if memory_output.returncode == 0:
                memory_bytes = int(memory_output.stdout.strip())
                memory_info = f"Total Memory: {memory_bytes // (1024**3)} GB"
        elif platform.system() == "Windows":
            memory_output = subprocess.run(
                ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
                capture_output=True,
                text=True,
                check=False,
            )
            if memory_output.returncode == 0:
                lines = memory_output.stdout.strip().split("\n")
                if len(lines) > 1:
                    memory_bytes = int(lines[1].strip())
                    memory_info = f"Total Memory: {memory_bytes // (1024**3)} GB"
    except Exception:
        pass  # Silently fail if memory info cannot be obtained

    return EnvironmentContext(
        python_version=python_version,
        pytest_version=pytest_version,
        platform_info=platform_info,
        installed_packages=installed_packages,
        loaded_plugins=loaded_plugins,
        command_line=" ".join(command),
        working_directory=os.getcwd(),
        cpu_info=cpu_info,
        memory_info=memory_info,
    )


def create_error_context(exit_code: int, error_message: str) -> ErrorContext:
    """
    Create a detailed error context object with exit code interpretation.

    Args:
        exit_code: Pytest exit code
        error_message: Error message from pytest execution

    Returns:
        ErrorContext object with detailed error information
    """
    exit_code_meaning, suggestion = get_pytest_exit_code_info(exit_code)

    # Extract traceback if available
    traceback = None
    if "Traceback" in error_message:
        traceback_parts = error_message.split("Traceback (most recent call last):")
        if len(traceback_parts) > 1:
            traceback = "Traceback (most recent call last):" + traceback_parts[1]

    # Extract collection errors if present
    collection_errors = None
    if "FAILED TO COLLECT" in error_message:
        collection_error_lines = []
        in_collection_error = False
        for line in error_message.split("\n"):
            if "FAILED TO COLLECT" in line:
                in_collection_error = True
                collection_error_lines.append(line)
            elif in_collection_error and line.strip():
                collection_error_lines.append(line)
            elif (
                in_collection_error and not line.strip()
            ):  # Empty line ends the section
                in_collection_error = False

        if collection_error_lines:
            collection_errors = collection_error_lines

    return ErrorContext(
        exit_code=exit_code,
        exit_code_meaning=exit_code_meaning,
        error_message=error_message,
        suggestion=suggestion,
        traceback=traceback,
        collection_errors=collection_errors,
    )
