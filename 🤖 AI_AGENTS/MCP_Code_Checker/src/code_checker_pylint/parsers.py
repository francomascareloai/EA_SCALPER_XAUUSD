"""
Functions for parsing pylint output.
"""

import json
import logging
from typing import List

import structlog

from .models import PylintMessage

logger = logging.getLogger(__name__)
structured_logger = structlog.get_logger(__name__)


def parse_pylint_json_output(
    raw_output: str,
) -> tuple[List[PylintMessage], str | None]:
    """
    Parse pylint JSON output into PylintMessage objects.

    Args:
        raw_output: Raw JSON output from pylint

    Returns:
        Tuple of (list of PylintMessage objects, error message if any)
    """
    messages: List[PylintMessage] = []
    error_message = None

    # Check if we have any output to parse
    if not raw_output or raw_output.strip() == "":
        structured_logger.info("Pylint produced no output")
        return messages, None

    try:
        pylint_output = json.loads(raw_output)
        if not isinstance(pylint_output, list):
            error_message = (
                f"Expected JSON array from pylint, got {type(pylint_output).__name__}"
            )
            structured_logger.error(
                "Invalid pylint output format", output_type=type(pylint_output).__name__
            )
            return messages, error_message

        # Log details about JSON parsing success
        structured_logger.debug(
            "Successfully parsed pylint JSON output",
            json_array_length=len(pylint_output),
            first_item_keys=list(pylint_output[0].keys()) if pylint_output else None,
        )

        for item in pylint_output:
            if not isinstance(item, dict):
                structured_logger.warning(
                    "Skipping non-dict item in pylint output",
                    item_type=type(item).__name__,
                )
                continue

            messages.append(
                PylintMessage(
                    type=item.get("type", ""),
                    module=item.get("module", ""),
                    obj=item.get("obj", ""),
                    line=item.get("line", -1),
                    column=item.get("column", -1),
                    path=item.get("path", ""),
                    symbol=item.get("symbol", ""),
                    message=item.get("message", ""),
                    message_id=item.get("message-id", ""),
                )
            )
    except json.JSONDecodeError as e:
        if len(raw_output) > 200:
            error_message = (
                f"Failed to parse Pylint JSON output: {e}. "
                f"First 200 chars of output: {raw_output[:200]}..."
            )
        else:
            error_message = (
                f"Failed to parse Pylint JSON output: {e}. Output: {raw_output}"
            )

        structured_logger.error(
            "JSON parse error",
            error=str(e),
            output_length=len(raw_output),
            output_preview=raw_output[:100],
        )

    return messages, error_message
