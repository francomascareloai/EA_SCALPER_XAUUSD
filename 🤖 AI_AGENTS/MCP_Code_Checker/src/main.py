"""Main entry point for the Code Checker MCP server."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import structlog

# Import logging utilities
from src.log_utils import setup_logging
from src.server import create_server

# Create loggers
stdlogger = logging.getLogger(__name__)
structured_logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="MCP Code Checker Server")
    parser.add_argument(
        "--project-dir",
        type=str,
        required=True,
        help="Base directory for code checking operations (required)",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        help="Path to Python interpreter to use for running tests. If not specified, defaults to the current Python interpreter (sys.executable)",
    )
    parser.add_argument(
        "--venv-path",
        type=str,
        help="Path to virtual environment to activate for running tests. When specified, the Python executable from this venv will be used instead of python-executable",
    )
    parser.add_argument(
        "--test-folder",
        type=str,
        default="tests",
        help="Path to the test folder (relative to project_dir). Defaults to 'tests'",
    )
    parser.add_argument(
        "--keep-temp-files",
        action="store_true",
        help="Keep temporary files after test execution. Useful for debugging when tests fail",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path for structured JSON logs (default: mcp_code_checker_{timestamp}.log in project_dir/logs/).",
    )
    parser.add_argument(
        "--console-only",
        action="store_true",
        help="Log only to console, ignore --log-file parameter.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the MCP server.
    """
    # Parse command line arguments
    args = parse_args()

    # Validate project directory first
    project_dir = Path(args.project_dir)
    if not project_dir.exists() or not project_dir.is_dir():
        print(
            f"Error: Project directory does not exist or is not a directory: {project_dir}"
        )
        sys.exit(1)

    # Convert to absolute path
    project_dir = project_dir.absolute()

    # Generate default log file path if not specified
    if args.console_only:
        log_file = None
    elif args.log_file:
        log_file = args.log_file
    else:
        # Create default log file in project_dir/logs/ with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logs_dir = project_dir / "logs"
        log_file = str(logs_dir / f"mcp_code_checker_{timestamp}.log")

    # Configure logging now that we have the project directory
    setup_logging(args.log_level, log_file)

    # Add debug logging after logger is initialized
    stdlogger.debug("Logger initialized in main")
    structured_logger.debug(
        "Structured logger initialized in main", log_level=args.log_level
    )

    stdlogger.info(
        f"Starting MCP Code Checker server with project directory: {project_dir}"
    )
    if log_file:
        structured_logger.info(
            "Starting MCP Code Checker server",
            project_dir=str(project_dir),
            log_level=args.log_level,
            log_file=log_file,
        )

    # Create and run the server
    server = create_server(
        project_dir,
        python_executable=args.python_executable,
        venv_path=args.venv_path,
        test_folder=args.test_folder,
        keep_temp_files=args.keep_temp_files,
    )

    stdlogger.info("Starting MCP server")
    structured_logger.info("Starting MCP server")
    stdlogger.debug("About to call server.run()")
    structured_logger.debug("About to call server.run()", project_dir=str(project_dir))
    server.run()
    stdlogger.debug(
        "After server.run() call - this line will only execute if server.run() returns"
    )


if __name__ == "__main__":
    main()
