@echo off
setlocal enabledelayedexpansion

REM Run pylint with specified parameters
pylint -E ./src ./tests > checks_output.txt 2>&1
set PYLINT_EXIT_CODE=%errorlevel%

REM Check if Pylint found any issues
if %PYLINT_EXIT_CODE% neq 0 (
    (
        echo INSTRUCTIONS FOR LLM: PYLINT ANALYSIS
        echo Pylint has detected potential critical errors in the source code:
        echo - Review serious code quality issues
        echo - Focus on:
        echo   1. Critical syntax errors
        echo   2. Import errors
        echo   3. Undefined variables
        echo.
        type checks_output.txt
    ) > checks_clipboard.txt

    type checks_clipboard.txt | clip
    echo Pylint found critical code errors. Output copied to clipboard.
    del checks_output.txt
    del checks_clipboard.txt
    exit /b 1
)

REM Run pytest after Pylint passes
pytest tests > checks_output.txt 2>&1
set PYTEST_EXIT_CODE=%errorlevel%

REM Check pytest results
if %PYTEST_EXIT_CODE% neq 0 (
    (
        echo INSTRUCTIONS FOR LLM: PYTEST RESULTS
        echo Pytest has found issues in the test suite:
        echo - Carefully review test failures and errors
        echo - Investigate potential causes:
        echo   1. Broken test assertions
        echo   2. Unexpected test behaviors
        echo   3. Potential code implementation issues
        echo - Provide specific recommendations for fixing test failures
        echo.
        type checks_output.txt
    ) > checks_clipboard.txt

    type checks_clipboard.txt | clip
    echo Pytest detected test failures. Output copied to clipboard.
    del checks_output.txt
    del checks_clipboard.txt
    exit /b 1
)

REM Run mypy with strict checks if Pylint and Pytest passed
python -m mypy --strict src tests > checks_output.txt 2>&1
set MYPY_EXIT_CODE=%errorlevel%

REM Check mypy results
if %MYPY_EXIT_CODE% neq 0 (
    (
        echo INSTRUCTIONS FOR LLM: MYPY TYPE CHECKING RESULTS
        echo Mypy has found type checking issues in the code:
        echo - Review type annotation problems
        echo - Fix issues related to:
        echo   1. Missing type annotations
        echo   2. Incompatible types
        echo   3. Optional/None handling errors
        echo   4. Function return type mismatches
        echo - Ensure all code follows strict typing standards
        echo - special cases:
        echo   - in the case of `@pytest.fixture`, if necessary `# type: ignore[misc]` could be added
        echo.
        type checks_output.txt
    ) > checks_clipboard.txt

    type checks_clipboard.txt | clip
    echo Mypy detected type checking errors. Output copied to clipboard.
    del checks_output.txt
    del checks_clipboard.txt
    exit /b 1
)

REM If all checks pass
echo All checks passed successfully. No issues detected.
del checks_output.txt 2>nul
exit /b 0