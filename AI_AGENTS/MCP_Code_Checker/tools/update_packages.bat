@echo off
echo Updating packages from pyproject.toml...

REM Check if we're in a virtual environment
if defined VIRTUAL_ENV (
    echo Using existing virtual environment: %VIRTUAL_ENV%
    uv pip install -e .
    echo Installing development dependencies...
    uv pip install -e .[dev]
) else (
    echo Creating/using project virtual environment...
    uv sync
    echo Installing development dependencies...
    uv sync --extra dev
)

echo Package update complete!
