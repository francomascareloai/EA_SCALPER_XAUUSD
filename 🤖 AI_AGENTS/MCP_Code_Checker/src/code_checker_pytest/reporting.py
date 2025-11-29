"""
Functions for formatting and reporting pytest test results.
"""

import logging
from typing import Optional

import structlog

from src.log_utils import log_function_call

from .models import PytestReport

logger = logging.getLogger(__name__)
structured_logger = structlog.get_logger(__name__)


@log_function_call
def create_prompt_for_failed_tests(
    test_session_result: PytestReport, max_number_of_tests_reported: int = 1
) -> Optional[str]:
    """
    Creates a prompt for an LLM based on the failed tests from a test session result.

    Args:
        test_session_result: The test session result to analyze
        max_number_of_tests_reported: Maximum number of tests to include in the prompt

    Returns:
        A prompt string, or None if no tests failed
    """
    prompt_parts = []

    failed_collectors = []
    if test_session_result.collectors:
        failed_collectors = [
            collector
            for collector in test_session_result.collectors
            if collector.outcome == "failed"
        ]
    if len(failed_collectors) > 0:
        prompt_parts.append(
            "The following collectors failed during the test session:\n"
        )
    for failed_collector in failed_collectors:
        prompt_parts.append(
            f"Collector ID: {failed_collector.nodeid} - outcome {failed_collector.outcome}\n"
        )
        if failed_collector.longrepr:
            prompt_parts.append(f"  Longrepr: {failed_collector.longrepr}\n")
        if failed_collector.result:
            for result in failed_collector.result:
                prompt_parts.append(f"  Result: {result}\n")
        prompt_parts.append("\n")

    failed_tests = []
    if test_session_result.tests:
        failed_tests = [
            test
            for test in test_session_result.tests
            if test.outcome in ["failed", "error"]
        ]

    test_count = 0
    if len(failed_tests) > 0:
        prompt_parts.append("The following tests failed during the test session:\n")
    for test in failed_tests:
        prompt_parts.append(f"Test ID: {test.nodeid} - outcome {test.outcome}\n")
        if test.call and test.call.crash:
            prompt_parts.append(f"  Error Message: {test.call.crash.message}\n")
        if test.call and test.call.traceback:
            prompt_parts.append("  Traceback:\n")
            for entry in test.call.traceback:
                prompt_parts.append(
                    f"   - {entry.path}:{entry.lineno} - {entry.message}\n"
                )
        if test.call and test.call.stdout:
            prompt_parts.append(f"  Stdout:\n```\n{test.call.stdout}\n```\n")
        if test.call and test.call.stderr:
            prompt_parts.append(f"  Stderr:\n```\n{test.call.stderr}\n```\n")
        if test.call and test.call.longrepr:
            prompt_parts.append(f"  Longrepr:\n```\n{test.call.longrepr}\n```\n")

        if test.setup and test.setup.outcome == "failed":
            prompt_parts.append(f"  Test Setup Outcome: {test.setup.outcome}\n")
            if test.setup.crash:
                prompt_parts.append(
                    f"  Test Setup Crash Error Message: {test.setup.crash.message}\n"
                )
                prompt_parts.append(
                    f"  Test Setup Crash Error Path: {test.setup.crash.path}\n"
                )
                prompt_parts.append(
                    f"  Setup Crash Error Line: {test.setup.crash.lineno}\n"
                )
            if test.setup.traceback:
                prompt_parts.append("  Test Setup Traceback:\n")
                for entry in test.setup.traceback:
                    prompt_parts.append(
                        f"   - {entry.path}:{entry.lineno} - {entry.message}\n"
                    )
            if test.setup.stdout:
                prompt_parts.append(
                    f"  Test Setup Stdout:\n```\n{test.setup.stdout}\n```\n"
                )
            if test.setup.stderr:
                prompt_parts.append(
                    f"  Test Setup Stderr:\n```\n{test.setup.stderr}\n```\n"
                )
            if test.setup.longrepr:
                prompt_parts.append(
                    f"  Test Setup Longrepr:\n```\n{test.setup.longrepr}\n```\n"
                )
            if test.setup.traceback and len(test.setup.traceback) > 0:
                prompt_parts.append("  Test Setup Traceback:\n")
                for entry in test.setup.traceback:
                    prompt_parts.append(
                        f"   - {entry.path}:{entry.lineno} - {entry.message}\n"
                    )

        prompt_parts.append("\n")

        test_count = test_count + 1
        if test_count >= max_number_of_tests_reported:
            break

        prompt_parts.append(
            "===============================================================================\n"
        )
        prompt_parts.append("\n")

    if len(prompt_parts) > 0:
        prompt_parts.append(
            "Can you provide an explanation for why these tests failed and suggest how they could be fixed?"
        )
        return "".join(prompt_parts)
    else:
        return None


def get_test_summary(test_session_result: PytestReport) -> str:
    """
    Generate a human-readable summary of the test results.

    Args:
        test_session_result: The test session result to summarize

    Returns:
        A string with the test summary
    """
    summary = test_session_result.summary

    parts = []
    parts.append(
        f"Collected {summary.collected} tests in {test_session_result.duration:.2f} seconds"
    )

    if summary.passed:
        parts.append(f"✅ Passed: {summary.passed}")
    if summary.failed:
        parts.append(f"❌ Failed: {summary.failed}")
    if summary.error:
        parts.append(f"⚠️ Error: {summary.error}")
    if summary.skipped:
        parts.append(f"⏭️ Skipped: {summary.skipped}")
    if summary.xfailed:
        parts.append(f"🔶 Expected failures: {summary.xfailed}")
    if summary.xpassed:
        parts.append(f"🔶 Unexpected passes: {summary.xpassed}")

    return " | ".join(parts)
