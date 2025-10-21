"""
Tests for the code_checker_pytest reporting functionality.
"""

from src.code_checker_pytest import (
    create_prompt_for_failed_tests,
    get_test_summary,
    parse_pytest_report,
)
from tests.test_code_checker.test_code_checker_pytest_common import SAMPLE_JSON


def test_create_prompt_no_failed_tests() -> None:
    """Test creating a prompt when there are no failed tests."""
    json_no_failed_tests = """
    {
        "created": 1518371686.7981803,
        "duration": 0.1235666275024414,
        "exitcode": 0,
        "root": "/path/to/tests",
        "environment": {
        "Python": "3.6.4",
        "Platform": "Linux-4.56.78-9-ARCH-x86_64-with-arch",
        "Packages": {
            "pytest": "3.4.0",
            "py": "1.5.2",
            "pluggy": "0.6.0"
        },
        "Plugins": {
            "json-report": "0.4.1",
            "xdist": "1.22.0",
            "metadata": "1.5.1",
            "forked": "0.2",
            "cov": "2.5.1"
        },
        "foo": "bar"
    },
    "summary": {
        "collected": 10,
        "passed": 10,
        "total": 10
    },
        "collectors": [],
        "tests": [],
        "warnings": []
    }
    """
    report = parse_pytest_report(json_no_failed_tests)
    prompt = create_prompt_for_failed_tests(report)
    assert prompt is None


def test_create_prompt_for_failed_tests() -> None:
    """Test creating a prompt for failed tests."""
    report = parse_pytest_report(SAMPLE_JSON)
    prompt = create_prompt_for_failed_tests(report)
    assert prompt is not None
    assert "The following tests failed during the test session:" in prompt
    assert "Test ID: test_foo.py::test_fail" in prompt

    # Safely check for attributes with None checks
    assert report.tests is not None
    assert report.tests[0].call is not None
    assert report.tests[0].call.crash is not None
    assert (
        "Error Message: TypeError: unsupported operand type(s) for -: 'int' and 'NoneType'"
        in prompt
    )

    assert "Stdout:" in prompt
    assert "Stderr:" in prompt
    assert "Traceback:" in prompt
    assert "test_foo.py:65 - " in prompt
    assert (
        "Can you provide an explanation for why these tests failed and suggest how they could be fixed?"
        in prompt
    )


def test_create_prompt_for_failed_tests_longrepr() -> None:
    """Test creating a prompt for failed tests with longrepr."""
    json_with_longrepr_tests = """
  {
      "created": 1518371686.7981803,
      "duration": 0.1235666275024414,
      "exitcode": 1,
      "root": "/path/to/tests",
      "environment": {
          "Python": "3.6.4",
          "Platform": "Linux-4.56.78-9-ARCH-x86_64-with-arch",
          "Packages": {
              "pytest": "3.4.0",
              "py": "1.5.2",
              "pluggy": "0.6.0"
          },
          "Plugins": {
              "json-report": "0.4.1",
              "xdist": "1.22.0",
              "metadata": "1.5.1",
              "forked": "0.2",
              "cov": "2.5.1"
          },
          "foo": "bar"
      },
      "summary": {
          "collected": 10,
          "passed": 2,
          "failed": 1,
          "total": 10
      },
      "collectors": [],
      "tests": [
          {
              "nodeid": "test_foo.py::test_fail",
              "lineno": 50,
              "keywords": [
                  "test_fail",
                  "test_foo.py",
                  "test_foo0"
              ],
              "outcome": "failed",
              "setup": {
                  "duration": 0.00018835067749023438,
                  "outcome": "passed"
              },
              "call": {
                  "duration": 0.00018835067749023438,
                  "outcome": "failed",
                  "longrepr": "def test_fail_nested():\\n    a = 1\\n    b = None\\n    a - b",
                  "crash": {
                      "path": "/path/to/tests/test_foo.py",
                      "lineno": 54,
                      "message": "TypeError: unsupported operand type(s) for -: 'int' and 'NoneType'"
                  },
                  "traceback": [
                      {
                          "path": "test_foo.py",
                          "lineno": 65,
                          "message": ""
                      }
                  ]
              },
              "teardown": {
                  "duration": 0.00018835067749023438,
                  "outcome": "passed"
              },
              "metadata": {
                  "foo": "bar"
              }
          }
      ],
      "warnings": []
  }
  """
    report = parse_pytest_report(json_with_longrepr_tests)
    prompt = create_prompt_for_failed_tests(report)
    assert prompt is not None
    assert "Longrepr:" in prompt
    assert "def test_fail_nested():" in prompt


def test_get_test_summary() -> None:
    """Test generating a human-readable summary of test results."""
    report = parse_pytest_report(SAMPLE_JSON)
    summary = get_test_summary(report)

    assert "Collected 10 tests in 0.12 seconds" in summary
    assert "✅ Passed: 2" in summary
    assert "❌ Failed: 3" in summary
    assert "⚠️ Error: 2" in summary
    assert "⏭️ Skipped: 1" in summary
    assert "🔶 Expected failures: 1" in summary
    assert "🔶 Unexpected passes: 1" in summary


def test_get_test_summary_minimal() -> None:
    """Test generating a summary with minimal test results."""
    json_minimal = """
    {
        "created": 1518371686.7981803,
        "duration": 0.1235666275024414,
        "exitcode": 0,
        "root": "/path/to/tests",
        "environment": {},
        "summary": {
            "collected": 5,
            "passed": 5,
            "total": 5
        }
    }
    """
    report = parse_pytest_report(json_minimal)
    summary = get_test_summary(report)

    assert "Collected 5 tests in 0.12 seconds" in summary
    assert "✅ Passed: 5" in summary
    assert "❌ Failed:" not in summary
    assert "⚠️ Error:" not in summary
