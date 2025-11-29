"""
Tests for the code_checker_pytest models.
"""

from src.code_checker_pytest import Crash, PytestReport, StageInfo, TracebackEntry
from src.code_checker_pytest.parsers import parse_pytest_report
from tests.test_code_checker.test_code_checker_pytest_common import SAMPLE_JSON


def test_model_instantiation() -> None:
    """Test that models can be instantiated with the right data."""
    crash = Crash(path="/path/to/file.py", lineno=10, message="Error message")
    assert crash.path == "/path/to/file.py"
    assert crash.lineno == 10
    assert crash.message == "Error message"

    traceback_entry = TracebackEntry(
        path="/path/to/file.py", lineno=10, message="Error message"
    )
    assert traceback_entry.path == "/path/to/file.py"
    assert traceback_entry.lineno == 10
    assert traceback_entry.message == "Error message"

    test_stage = StageInfo(
        duration=0.1, outcome="failed", crash=crash, traceback=[traceback_entry]
    )
    assert test_stage.duration == 0.1
    assert test_stage.outcome == "failed"
    assert test_stage.crash == crash
    assert test_stage.traceback == [traceback_entry]


def test_report_model_structure() -> None:
    """Test that the PytestReport model structure matches expected JSON structure."""
    report = parse_pytest_report(SAMPLE_JSON)

    # Verify top-level attributes
    assert isinstance(report, PytestReport)
    assert report.created == 1518371686.7981803
    assert report.duration == 0.1235666275024414
    assert report.exitcode == 1
    assert report.root == "/path/to/tests"

    # Verify summary section
    assert report.summary.collected == 10
    assert report.summary.total == 10
    assert report.summary.passed == 2
    assert report.summary.failed == 3
    assert report.summary.xfailed == 1
    assert report.summary.xpassed == 1
    assert report.summary.error == 2
    assert report.summary.skipped == 1

    # Verify collectors
    assert report.collectors is not None
    assert len(report.collectors) == 3

    # Verify collector structure
    collector = report.collectors[1]  # Second collector
    assert collector.nodeid == "test_foo.py"
    assert collector.outcome == "passed"
    assert len(collector.result) == 2

    # Verify collector result structure
    collector_result = collector.result[0]
    assert collector_result.nodeid == "test_foo.py::test_pass"
    assert collector_result.type == "Function"
    assert collector_result.lineno == 24
    assert collector_result.deselected is True

    # Verify tests section
    assert report.tests is not None
    assert len(report.tests) == 2

    # Verify test structure
    test = report.tests[0]
    assert test.nodeid == "test_foo.py::test_fail"
    assert test.lineno == 50
    assert len(test.keywords) == 3
    assert test.outcome == "failed"
    assert test.metadata is not None
    assert test.metadata["foo"] == "bar"

    # Verify test stages
    assert test.setup is not None
    assert test.setup.duration == 0.00018835067749023438
    assert test.setup.outcome == "passed"

    assert test.call is not None
    assert test.call.duration == 0.00018835067749023438
    assert test.call.outcome == "failed"

    # Verify crash information in test.call
    assert test.call.crash is not None
    assert test.call.crash.path == "/path/to/tests/test_foo.py"
    assert test.call.crash.lineno == 54
    assert (
        test.call.crash.message
        == "TypeError: unsupported operand type(s) for -: 'int' and 'NoneType'"
    )

    # Verify traceback in test.call
    assert test.call.traceback is not None
    assert len(test.call.traceback) == 4
    assert test.call.traceback[0].path == "test_foo.py"
    assert test.call.traceback[0].lineno == 65
    assert test.call.traceback[0].message == ""

    # Verify stdout and stderr
    assert test.call.stdout == "foo\nbar\n"
    assert test.call.stderr == "baz\n"

    # Verify warnings section
    assert report.warnings is not None
    assert len(report.warnings) == 1
    assert report.warnings[0].code == "C1"
    assert report.warnings[0].path == "/path/to/tests/test_foo.py"
    assert report.warnings[0].nodeid == "test_foo.py::TestFoo"
    assert (
        report.warnings[0].message
        == "cannot collect test class 'TestFoo' because it has a __init__ constructor"
    )
