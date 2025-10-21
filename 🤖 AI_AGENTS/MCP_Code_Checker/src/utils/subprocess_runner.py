"""
Subprocess execution utilities with MCP STDIO isolation support.

This module provides functions for executing command-line tools with proper
timeout handling and STDIO isolation for Python commands in MCP server contexts.
"""

import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import structlog

from src.log_utils import log_function_call

logger = logging.getLogger(__name__)
structured_logger = structlog.get_logger(__name__)


@dataclass
class CommandResult:
    """Represents the result of a command execution."""

    return_code: int
    stdout: str
    stderr: str
    timed_out: bool
    execution_error: str | None = None
    command: list[str] | None = field(default=None)
    runner_type: str | None = field(default=None)
    execution_time_ms: int | None = field(default=None)


@dataclass
class CommandOptions:
    """Configuration options for command execution."""

    cwd: str | None = None
    timeout_seconds: int = 120
    env: dict[str, str] | None = None
    capture_output: bool = True
    text: bool = True
    check: bool = False
    shell: bool = False
    input_data: str | None = None


def is_python_command(command: list[str]) -> bool:
    """Check if a command is a Python execution command."""
    if not command:
        return False

    executable = Path(command[0]).name.lower()
    return (
        executable in ["python", "python3", "python.exe", "python3.exe"]
        or command[0] == sys.executable
    )


def get_python_isolation_env() -> dict[str, str]:
    """Get environment variables for Python subprocess isolation."""
    env = os.environ.copy()

    # Python-specific settings to prevent MCP STDIO conflicts
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONSTARTUP": "",
        }
    )

    # Remove MCP-specific variables
    for var in ["MCP_STDIO_TRANSPORT", "MCP_SERVER_NAME", "MCP_CLIENT_PARAMS"]:
        env.pop(var, None)

    return env


def _safe_preexec_fn() -> None:
    """
    Safely attempt to create a new session.

    This is used on Unix-like systems to isolate the subprocess.
    Errors are silently ignored as they may occur in restricted environments.
    """
    try:
        if hasattr(os, "setsid"):
            os.setsid()  # type: ignore[attr-defined]
    except (OSError, PermissionError, AttributeError):
        # Ignore errors - may already be session leader or restricted env
        pass


def _run_subprocess(
    command: list[str], options: CommandOptions, use_stdio_isolation: bool = False
) -> subprocess.CompletedProcess[str]:
    """
    Internal function to run subprocess with or without STDIO isolation.

    Args:
        command: Command to execute
        options: Execution options
        use_stdio_isolation: Whether to use file-based STDIO isolation

    Returns:
        CompletedProcess with execution results
    """
    # Prepare environment
    env = options.env or os.environ.copy()
    if is_python_command(command):
        env = get_python_isolation_env()
        if options.env:
            env.update(options.env)

    # Handle input data and stdin
    stdin_value = subprocess.DEVNULL if options.input_data is None else None

    # Prepare preexec_fn for Unix-like systems
    preexec_fn: Callable[[], Any] | None = None
    start_new_session = False
    if os.name != "nt":
        preexec_fn = _safe_preexec_fn
        start_new_session = True

    # Use file-based STDIO for Python commands if needed
    if use_stdio_isolation and options.capture_output:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout_file = Path(temp_dir) / "stdout.txt"
            stderr_file = Path(temp_dir) / "stderr.txt"

            process = None
            timed_out = False

            try:
                with (
                    open(stdout_file, "w", encoding="utf-8") as stdout_f,
                    open(stderr_file, "w", encoding="utf-8") as stderr_f,
                ):
                    process = subprocess.run(
                        command,
                        stdout=stdout_f,
                        stderr=stderr_f,
                        cwd=options.cwd,
                        text=options.text,
                        timeout=options.timeout_seconds,
                        env=env,
                        shell=options.shell,
                        stdin=stdin_value,
                        input=options.input_data,
                        start_new_session=start_new_session,
                        preexec_fn=preexec_fn,
                    )
            except subprocess.TimeoutExpired as e:
                # Mark timeout and set process info
                timed_out = True
                process = subprocess.CompletedProcess(
                    args=command,
                    returncode=1,
                    stdout="",
                    stderr="",
                )
                # Re-raise to be handled by the caller
                raise

            # Read output files after process completes
            # Use a small delay on Windows to avoid file locking issues
            if os.name == "nt" and not timed_out:
                time.sleep(0.1)

            # Read output files, handling potential errors
            stdout = ""
            stderr = ""

            try:
                if stdout_file.exists():
                    stdout = stdout_file.read_text(encoding="utf-8")
            except (OSError, PermissionError):
                # If we can't read the file, leave stdout empty
                pass

            try:
                if stderr_file.exists():
                    stderr = stderr_file.read_text(encoding="utf-8")
            except (OSError, PermissionError):
                # If we can't read the file, leave stderr empty
                pass

            if process is None:
                # This should not happen, but handle it gracefully
                process = subprocess.CompletedProcess(
                    args=command,
                    returncode=1,
                    stdout=stdout,
                    stderr=stderr,
                )

            return subprocess.CompletedProcess(
                args=command,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr,
            )
    else:
        # Regular execution
        return subprocess.run(
            command,
            capture_output=options.capture_output,
            cwd=options.cwd,
            text=options.text,
            timeout=options.timeout_seconds,
            env=env,
            shell=options.shell,
            stdin=stdin_value,
            input=options.input_data,
            start_new_session=start_new_session,
            preexec_fn=preexec_fn,
        )


@log_function_call
def execute_subprocess(
    command: list[str], options: CommandOptions | None = None
) -> CommandResult:
    """
    Execute a command with automatic STDIO isolation for Python commands.

    Args:
        command: Command and arguments as a list
        options: Execution options

    Returns:
        CommandResult with execution details
    """
    if command is None:
        raise TypeError("Command cannot be None")

    if options is None:
        options = CommandOptions()

    start_time = time.time()

    # Determine if we need STDIO isolation
    use_isolation = is_python_command(command)

    structured_logger.debug(
        "Starting subprocess execution",
        command=command[:3] if command else None,
        cwd=options.cwd,
        timeout_seconds=options.timeout_seconds,
        use_isolation=use_isolation,
    )

    try:
        process = _run_subprocess(command, options, use_isolation)

        # Handle check parameter
        if options.check and process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, command, process.stdout, process.stderr
            )

        execution_time_ms = int((time.time() - start_time) * 1000)

        return CommandResult(
            return_code=process.returncode,
            stdout=process.stdout or "",
            stderr=process.stderr or "",
            timed_out=False,
            command=command,
            runner_type="subprocess",
            execution_time_ms=execution_time_ms,
        )

    except subprocess.TimeoutExpired:
        execution_time_ms = int((time.time() - start_time) * 1000)
        return CommandResult(
            return_code=1,
            stdout="",
            stderr="",
            timed_out=True,
            execution_error=f"Process timed out after {options.timeout_seconds} seconds",
            command=command,
            runner_type="subprocess",
            execution_time_ms=execution_time_ms,
        )

    except subprocess.CalledProcessError as e:
        if options.check:
            raise
        execution_time_ms = int((time.time() - start_time) * 1000)
        return CommandResult(
            return_code=e.returncode,
            stdout=getattr(e, "stdout", "") or "",
            stderr=getattr(e, "stderr", "") or "",
            timed_out=False,
            command=command,
            runner_type="subprocess",
            execution_time_ms=execution_time_ms,
        )

    except Exception as e:
        # Handle all other exceptions (FileNotFoundError, PermissionError, etc.)
        execution_time_ms = int((time.time() - start_time) * 1000)
        structured_logger.error(
            "Subprocess execution failed",
            error=str(e),
            error_type=type(e).__name__,
            command_preview=command[:3] if command else None,
        )
        return CommandResult(
            return_code=1,
            stdout="",
            stderr="",
            timed_out=False,
            execution_error=f"{type(e).__name__}: {e}",
            command=command,
            runner_type="subprocess",
            execution_time_ms=execution_time_ms,
        )


def execute_command(
    command: list[str],
    cwd: str | None = None,
    timeout_seconds: int = 120,
    env: dict[str, str] | None = None,
) -> CommandResult:
    """
    Execute a command with automatic STDIO isolation for Python commands.

    Args:
        command: Complete command as list (e.g., ["python", "-m", "pylint", "src"])
        cwd: Working directory for subprocess
        timeout_seconds: Timeout in seconds
        env: Optional environment variables

    Returns:
        CommandResult with execution details and output
    """
    options = CommandOptions(
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        env=env,
    )
    return execute_subprocess(command, options)
