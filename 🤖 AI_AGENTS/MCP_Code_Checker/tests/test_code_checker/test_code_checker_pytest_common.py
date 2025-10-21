"""
Common test utilities and constants for code_checker_pytest tests.
"""

import os
import shutil
import sys
from pathlib import Path

# Add source directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Sample JSON report for testing
SAMPLE_JSON = """
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
    "collectors": [
        {
            "nodeid": "",
            "outcome": "passed",
            "result": [
                {
                    "nodeid": "test_foo.py",
                    "type": "Module"
                }
            ]
        },
        {
            "nodeid": "test_foo.py",
            "outcome": "passed",
            "result": [
                {
                    "nodeid": "test_foo.py::test_pass",
                    "type": "Function",
                    "lineno": 24,
                    "deselected": true
                },
                {
                    "nodeid": "test_foo.py::test_fail",
                    "type": "Function",
                    "lineno": 50
                }
            ]
        },
        {
            "nodeid": "test_bar.py",
            "outcome": "failed",
            "result": [],
            "longrepr": "/usr/lib/python3.6 ... invalid syntax"
        }
    ],
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
                "stderr": "baz\\n"
            },
             "teardown": {
              "duration": 0.00018835067749023438,
              "outcome": "passed"
            },
            "metadata": {
                "foo": "bar"
            }
        },
        {
            "nodeid": "test_bar.py::test_error",
             "lineno": 50,
            "keywords": [
                "test_fail",
                "test_bar.py",
                "test_bar0"
            ],
            "outcome": "error",
           "setup": {
              "duration": 0.00018835067749023438,
              "outcome": "error",
               "longrepr": "/usr/lib/python3.6 ... invalid syntax"
            }
        }
    ],
    "warnings": [
        {
            "code": "C1",
            "path": "/path/to/tests/test_foo.py",
            "nodeid": "test_foo.py::TestFoo",
            "message": "cannot collect test class 'TestFoo' because it has a __init__ constructor"
        }
    ]
}
"""


def _create_test_project(test_dir: Path) -> None:
    """Creates a sample test project with a passing and a failing test."""
    (test_dir / "src").mkdir(parents=True, exist_ok=True)
    (test_dir / "tests").mkdir(parents=True, exist_ok=True)

    with open(test_dir / "src" / "__init__.py", "w") as f:
        f.write("")
    with open(test_dir / "tests" / "__init__.py", "w") as f:
        f.write("")
    with open(test_dir / "tests" / "test_sample.py", "w") as f:
        f.write(
            """
import pytest

def test_passing():
    assert 1 == 1

def test_failing():
    assert 1 == 2
"""
        )


def _cleanup_test_project(test_dir: Path) -> None:
    """Removes the sample test project after the test."""
    shutil.rmtree(test_dir)
