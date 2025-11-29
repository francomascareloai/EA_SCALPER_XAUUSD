"""
Data models for pytest test results and reports.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Crash:
    path: str
    lineno: int
    message: str


@dataclass
class TracebackEntry:
    path: str
    lineno: int
    message: str


@dataclass
class LogRecord:
    name: str
    msg: str
    args: Optional[Any]
    levelname: str
    levelno: int
    pathname: str
    filename: str
    module: str
    exc_info: Optional[Any]
    exc_text: Optional[str]
    stack_info: Optional[str]
    lineno: int
    funcName: str
    created: float
    msecs: float
    relativeCreated: float
    thread: int
    threadName: str
    processName: str
    process: int
    taskName: str = ""
    asctime: str = ""


@dataclass
class Log:
    logs: List[LogRecord]


@dataclass
class StageInfo:
    duration: float
    outcome: str
    crash: Optional[Crash] = None
    traceback: Optional[List[TracebackEntry]] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    log: Optional[Log] = None
    longrepr: Optional[str] = None


@dataclass
class Test:
    nodeid: str
    lineno: int
    keywords: List[str]
    outcome: str
    setup: Optional[StageInfo] = None
    call: Optional[StageInfo] = None
    teardown: Optional[StageInfo] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CollectorResult:
    nodeid: str
    type: str
    lineno: Optional[int] = None
    deselected: Optional[bool] = None


@dataclass
class Collector:
    nodeid: str
    outcome: str
    result: List[CollectorResult]
    longrepr: Optional[str] = None


@dataclass
class Summary:
    collected: int
    total: int
    deselected: Optional[int] = None
    passed: Optional[int] = None
    failed: Optional[int] = None
    xfailed: Optional[int] = None
    xpassed: Optional[int] = None
    error: Optional[int] = None
    skipped: Optional[int] = None


@dataclass
class Warning:
    message: str
    code: Optional[str] = None
    path: Optional[str] = None
    nodeid: Optional[str] = None
    when: Optional[str] = None
    category: Optional[str] = None
    filename: Optional[str] = None
    lineno: Optional[str] = None


@dataclass
class EnvironmentContext:
    python_version: str
    pytest_version: str
    platform_info: str
    installed_packages: List[Dict[str, str]]
    loaded_plugins: List[str]
    command_line: str
    working_directory: str
    cpu_info: Optional[str] = None
    memory_info: Optional[str] = None


@dataclass
class ErrorContext:
    exit_code: int
    exit_code_meaning: str
    error_message: str
    suggestion: str
    traceback: Optional[str] = None
    collection_errors: Optional[List[str]] = None


@dataclass
class PytestReport:
    created: float
    duration: float
    exitcode: int
    root: str
    environment: Dict[str, Any]
    summary: Summary
    collectors: Optional[List[Collector]] = None
    tests: Optional[List[Test]] = None
    warnings: Optional[List[Warning]] = None
    environment_context: Optional[EnvironmentContext] = None
    error_context: Optional[ErrorContext] = None
