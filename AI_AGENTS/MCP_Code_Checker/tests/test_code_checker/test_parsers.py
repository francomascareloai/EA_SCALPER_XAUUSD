"""
Tests for the code_checker_pytest report parsing functionality.
"""

from src.code_checker_pytest import PytestReport, parse_pytest_report
from tests.test_code_checker.test_code_checker_pytest_common import SAMPLE_JSON


def test_parse_report() -> None:
    """Test that pytest JSON report is correctly parsed into a PytestReport object."""
    report = parse_pytest_report(SAMPLE_JSON)

    assert isinstance(report, PytestReport)
    assert report.duration == 0.1235666275024414
    assert report.exitcode == 1
    assert report.summary.total == 10
    assert report.summary.passed == 2
    assert report.summary.failed == 3

    assert report.collectors is not None
    assert len(report.collectors) == 3
    assert report.collectors[0].nodeid == ""
    assert report.collectors[1].nodeid == "test_foo.py"
    assert report.collectors[2].nodeid == "test_bar.py"

    assert report.tests is not None
    assert len(report.tests) == 2
    assert report.tests[0].nodeid == "test_foo.py::test_fail"
    assert report.tests[1].nodeid == "test_bar.py::test_error"

    assert report.tests[0].call is not None
    assert report.tests[0].call.duration > 0
    assert report.tests[0].call.outcome == "failed"

    assert report.tests[0].call.crash is not None
    assert (
        report.tests[0].call.crash.message
        == "TypeError: unsupported operand type(s) for -: 'int' and 'NoneType'"
    )

    assert report.tests[0].call.traceback is not None
    assert len(report.tests[0].call.traceback) == 4
    assert report.tests[0].call.traceback[0].message == ""

    assert report.warnings is not None
    assert len(report.warnings) == 1
    assert report.warnings[0].nodeid == "test_foo.py::TestFoo"


def test_parse_report_no_collectors() -> None:
    """Test parsing a report without collectors."""
    json_no_collectors = """
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
        "failed": 3,
        "xfailed": 1,
        "xpassed": 1,
        "error": 2,
        "skipped": 1,
        "total": 10
    },
        "tests": [],
        "warnings": []
    }
    """
    report = parse_pytest_report(json_no_collectors)
    assert isinstance(report, PytestReport)
    assert report.collectors is None


def test_parse_report_no_tests() -> None:
    """Test parsing a report without tests."""
    json_no_tests = """
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
        "failed": 3,
        "xfailed": 1,
        "xpassed": 1,
        "error": 2,
        "skipped": 1,
        "total": 10
    },
       "collectors": [],
        "warnings": []
    }
    """
    report = parse_pytest_report(json_no_tests)
    assert isinstance(report, PytestReport)
    assert report.tests is None


def test_parse_report_with_log() -> None:
    """Test parsing a report with log entries."""
    json_with_log = """
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
        "failed": 3,
        "xfailed": 1,
        "xpassed": 1,
        "error": 2,
        "skipped": 1,
        "total": 10
    },
        "collectors": [],
        "tests": [{
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
                },
                {
                "path": "test_foo.py",
                "lineno": 63,
                "message": "in foo"
                },
                {
                "path": "test_foo.py",
                "lineno": 63,
                "message": "in <listcomp>"
                },
                {
                "path": "test_foo.py",
                "lineno": 54,
                "message": "TypeError"
                }
            ],
            "stdout": "foo\\nbar\\n",
            "stderr": "baz\\n",
             "log": [{
                "name": "root",
                "msg": "This is a warning.",
                "args": null,
                "levelname": "WARNING",
                "levelno": 30,
                "pathname": "/path/to/tests/test_foo.py",
                "filename": "test_foo.py",
                "module": "test_foo",
                "exc_info": null,
                "exc_text": null,
                "stack_info": null,
                "lineno": 8,
                "funcName": "foo",
                "created": 1519772464.291738,
                "msecs": 291.73803329467773,
                "relativeCreated": 332.90839195251465,
                "thread": 140671803118912,
                "threadName": "MainThread",
                "processName": "MainProcess",
                "process": 31481
            }]
            },
            "teardown": {
            "duration": 0.00018835067749023438,
            "outcome": "passed"
            }
        }],
        "warnings": []
    }
    """
    report = parse_pytest_report(json_with_log)
    assert isinstance(report, PytestReport)
    assert report.tests is not None
    assert len(report.tests) == 1
    assert report.tests[0].call is not None
    assert report.tests[0].call.log is not None
    assert report.tests[0].call.log.logs is not None
    assert report.tests[0].call.log.logs[0].msg == "This is a warning."
