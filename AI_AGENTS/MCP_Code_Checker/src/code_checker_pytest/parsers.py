"""
Functions for parsing pytest test results and output.
"""

import json
from typing import Any, Dict

from .models import (
    Collector,
    CollectorResult,
    Crash,
    Log,
    LogRecord,
    PytestReport,
    StageInfo,
    Summary,
    Test,
    TracebackEntry,
    Warning,
)


def parse_test_stage(stage_data: Dict[str, Any]) -> StageInfo:
    """
    Parse test stage data from the pytest JSON report.

    Args:
        stage_data: Dictionary containing test stage data from JSON

    Returns:
        StageInfo object populated with data from JSON
    """
    crash = None
    if "crash" in stage_data and stage_data["crash"]:
        crash = Crash(**stage_data["crash"])

    traceback = None
    if "traceback" in stage_data and stage_data["traceback"]:
        traceback = [TracebackEntry(**entry) for entry in stage_data["traceback"]]

    log = None
    if "log" in stage_data and stage_data["log"]:
        log_records = [
            LogRecord(**log_record_data) for log_record_data in stage_data["log"]
        ]
        log = Log(logs=log_records)

    return StageInfo(
        duration=stage_data["duration"],
        outcome=stage_data["outcome"],
        crash=crash,
        traceback=traceback,
        stdout=stage_data.get("stdout"),
        stderr=stage_data.get("stderr"),
        log=log,
        longrepr=stage_data.get("longrepr"),
    )


def parse_pytest_report(json_data: str) -> PytestReport:
    """
    Parse a JSON string into a PytestReport object.

    Args:
        json_data: JSON string from pytest json report

    Returns:
        PytestReport object with test results
    """
    data = json.loads(json_data)

    summary = Summary(**data["summary"])

    environment = data["environment"]

    collectors = None
    if "collectors" in data and data["collectors"]:
        collectors = []
        for collector_data in data["collectors"]:
            result_data_list = []
            for result_data in collector_data["result"]:
                result_data_list.append(CollectorResult(**result_data))
            collector = Collector(
                nodeid=collector_data["nodeid"],
                outcome=collector_data["outcome"],
                result=result_data_list,
                longrepr=collector_data.get("longrepr"),
            )
            collectors.append(collector)

    tests = None
    if "tests" in data and data["tests"]:
        tests = []
        for test_data in data["tests"]:
            setup_stage = None
            call_stage = None
            teardown_stage = None

            if "setup" in test_data:
                setup_stage = parse_test_stage(test_data["setup"])
            if "call" in test_data:
                call_stage = parse_test_stage(test_data["call"])
            if "teardown" in test_data:
                teardown_stage = parse_test_stage(test_data["teardown"])

            test = Test(
                nodeid=test_data["nodeid"],
                lineno=test_data["lineno"],
                keywords=test_data["keywords"],
                outcome=test_data["outcome"],
                setup=setup_stage,
                call=call_stage,
                teardown=teardown_stage,
                metadata=test_data.get("metadata"),
            )
            tests.append(test)

    warnings = None
    if "warnings" in data and data["warnings"]:
        warnings = [Warning(**warning_data) for warning_data in data["warnings"]]

    return PytestReport(
        created=data["created"],
        duration=data["duration"],
        exitcode=data["exitcode"],
        root=data["root"],
        environment=environment,
        summary=summary,
        collectors=collectors,
        tests=tests,
        warnings=warnings,
    )
