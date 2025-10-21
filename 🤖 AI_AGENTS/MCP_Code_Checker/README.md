# MCP Code Checker

A Model Context Protocol (MCP) server providing code quality checking operations. This server offers a API for performing code quality checks within a specified project directory, following the MCP protocol design.

## Overview

This MCP server enables AI assistants like Claude (via Claude Desktop) or other MCP-compatible systems to perform quality checks on your code. With these capabilities, AI assistants can:

- Run pylint checks to identify code quality issues
- Execute pytest to identify failing tests
- Generate smart prompts for LLMs to explain issues and suggest fixes
- Combine multiple checks for comprehensive code quality analysis

All operations are securely contained within your specified project directory, giving you control while enabling powerful AI collaboration for code quality improvement.

By connecting your AI assistant to your code checking tools, you can transform your debugging workflow - describe what you need in natural language and let the AI identify and fix issues directly in your project files.

## Features

- `run_pylint_check`: Run pylint on the project code and generate smart prompts for LLMs
- `run_pytest_check`: Run pytest on the project code and generate smart prompts for LLMs
- `run_all_checks`: Run all code checks (pylint and pytest) and generate combined results

### Pylint Parameters

The pylint tools expose the following parameters for customization:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disable_codes` | list | None | List of pylint error codes to disable during analysis |
| `target_directories` | list | ["src", "tests"] | List of directories to analyze relative to project_dir |

**Target Directories Examples:**
- `["src"]` - Analyze only source code directory
- `["src", "tests"]` - Analyze both source and test directories (default)
- `["mypackage", "tests"]` - For projects with different package structures
- `["lib", "scripts", "tests"]` - For complex multi-directory projects
- `["."]` - Analyze entire project directory (may be slow for large projects)

Additionally, `run_all_checks` exposes:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `categories` | set | ERROR, FATAL | Set of pylint message categories to include (convention, refactor, warning, error, fatal) |

### Pytest Parameters

Both `run_pytest_check` and `run_all_checks` expose the following parameters for customization:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test_folder` | string | "tests" | Path to the test folder relative to project directory |
| `markers` | list | None | Optional list of pytest markers to filter tests |
| `verbosity` | integer | 2 | Pytest verbosity level (0-3) |
| `extra_args` | list | None | Optional list of additional pytest arguments |
| `env_vars` | dictionary | None | Optional environment variables for the subprocess |
| `keep_temp_files` | boolean | False | Whether to keep temporary files after execution |
| `continue_on_collection_errors` | boolean | True | Whether to continue on collection errors |
| `python_executable` | string | None | Path to Python interpreter to use for running tests |
| `venv_path` | string | None | Path to virtual environment to activate for running tests |

## Installation

```bash
# Clone the repository
git clone https://github.com/MarcusJellinghaus/mcp-code-checker.git
cd mcp-code-checker

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Running the Server

```bash
python -m src.main --project-dir /path/to/project [--python-executable /path/to/python] [--venv-path /path/to/venv] [--log-level LEVEL] [--log-file PATH]
```

The server uses FastMCP for operation. The project directory parameter (`--project-dir`) is **required** for security reasons. All code checking operations will be restricted to this directory.

Additional parameters:

- `--python-executable`: Optional path to Python interpreter to use for running tests
- `--venv-path`: Optional path to virtual environment to activate for running tests
- `--log-level`: Optional logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to INFO
- `--log-file`: Optional path for structured JSON logs. If not specified, logs to mcp_code_checker_{timestamp}.log in project_dir/logs/
- `--console-only`: Optional flag to log only to console, ignoring --log-file parameter

## Project Structure Support

The server automatically detects and analyzes Python code in standard project structures:

**Default Analysis:**
- `src/` directory (if present) - Main source code
- `tests/` directory (if present) - Test files

**Custom Project Structures:**
Use the `target_directories` parameter to specify different directories:

```python
# For a package-based structure
target_directories = ["mypackage", "tests"]

# For a simple project with code in root
target_directories = ["."]

# For complex multi-module projects
target_directories = ["module1", "module2", "shared", "tests"]
```

## Structured Logging

The server provides comprehensive logging capabilities:

- **Standard human-readable logs** to console for development/debugging
- **Structured JSON logs** to file for analysis and monitoring
- **Function call tracking** with parameters, timing, and results
- **Automatic error context capture** with full stack traces
- **Configurable log levels** (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Default timestamped log files** in `project_dir/logs/mcp_code_checker_{timestamp}.log`

Example structured log entries:
```json
{
  "timestamp": "2025-08-05 14:30:15",
  "level": "info",
  "event": "Starting pylint check",
  "project_dir": "/path/to/project",
  "disable_codes": ["C0114", "C0116"],
  "target_directories": ["src", "tests"]
}
```

Use `--console-only` to disable file logging for simple development scenarios.

## Using with Claude Desktop App

To enable Claude to use this code checking server for analyzing files in your local environment:

1. Create or modify the Claude configuration file:
   - Location: `%APPDATA%\Claude\claude_desktop_config.json` (on Windows)
   - On macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

2. Add the MCP server configuration to the file:

```json
{
    "mcpServers": {
        "code_checker": {
            "command": "C:\\path\\to\\mcp_code_checker\\.venv\\Scripts\\python.exe",
            "args": [                
                "C:\\path\\to\\mcp_code_checker\\src\\main.py",
                "--project-dir",
                "C:\\path\\to\\your\\project",
            "--python-executable",
            "C:\\path\\to\\python.exe",
            "--venv-path",
            "C:\\path\\to\\venv",
            "--log-level",
            "INFO"
                ],
            "env": {
        "PYTHONPATH": "C:\\path\\to\\mcp_code_checker\\"
    }
}
}
}
```

3. Replace all `C:\\path\\to\\` instances with your actual paths:
   - Point to your Python virtual environment 
   - Set the project directory to the folder you want Claude to check
   - Make sure the PYTHONPATH points to the mcp_code_checker root folder

4. Restart the Claude desktop app to apply changes

Claude will now be able to analyze code in your specified project directory.

5. Log files location:
   - Windows: `%APPDATA%\Claude\logs`
   - These logs can be helpful for troubleshooting issues with the MCP server connection

For more information on logging and troubleshooting, see the [MCP Documentation](https://modelcontextprotocol.io/quickstart/user#getting-logs-from-claude-for-desktop).

## Using MCP Inspector

MCP Inspector allows you to debug and test your MCP server:

1. Start MCP Inspector by running:

```bash
npx @modelcontextprotocol/inspector \
  uv \
  --directory C:\path\to\mcp_code_checker \
  run \
  src\main.py
```

2. In the MCP Inspector web UI, configure with the following:
   - Python interpreter: `C:\path\to\mcp_code_checker\.venv\Scripts\python.exe`
   - Arguments: `C:\path\to\mcp_code_checker\src\main.py --project-dir C:\path\to\your\project --log-level DEBUG`
   - Environment variables:
     - Name: `PYTHONPATH`
     - Value: `C:\path\to\mcp_code_checker\`

3. This will launch the server and provide a debug interface for testing the available tools.

## Available Tools

The server exposes the following MCP tools:

### Run Pylint Check
- Runs pylint on the project code and generates smart prompts for LLMs
- Returns: A string containing either pylint results or a prompt for an LLM to interpret
- Helps identify code quality issues, style problems, and potential bugs
- Customizable with parameters for disabling specific pylint codes and targeting specific directories
- Supports flexible project structures through `target_directories` parameter

### Run Pytest Check
- Runs pytest on the project code and generates smart prompts for LLMs
- Returns: A string containing either pytest results or a prompt for an LLM to interpret
- Identifies failing tests and provides detailed information about test failures
- Customizable with parameters for test selection, environment, and verbosity

### Run All Checks
- Runs all code checks (pylint and pytest) and generates combined results
- Returns: A string containing results from all checks and/or LLM prompts
- Provides a comprehensive analysis of code quality in a single operation
- Supports customization parameters for both pylint and pytest, including target directories

## Security Features

- All checks are performed within the specified project directory
- Code execution is limited to the Python test files within the project
- Results are formatted for easy interpretation by both humans and LLMs
- Directory traversal protection through validation of target directories

## Development

### Setting up the development environment on windows

```cmd
REM Clone the repository
git clone https://github.com/yourusername/mcp-code-checker.git
cd mcp-code-checker

REM Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

REM Install dependencies
pip install -e .

REM Install development dependencies
pip install -e ".[dev]"

```

## Running with MCP Dev Tools

```bash
# Set the PYTHONPATH and run the server module using mcp dev
set PYTHONPATH=. && mcp dev src/server.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive license that allows reuse with minimal restrictions. It permits use, copying, modification, and distribution with proper attribution.

## Links

- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Filesystem Tools](https://github.com/MarcusJellinghaus/mcp_server_filesystem)
