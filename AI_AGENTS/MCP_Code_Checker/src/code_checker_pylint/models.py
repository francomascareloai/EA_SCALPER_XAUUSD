"""
Data models for pylint analysis results.
"""

from enum import Enum
from typing import List, NamedTuple, Optional, Set


class PylintMessageType(Enum):
    """Categories for Pylint message types."""

    CONVENTION = "convention"
    REFACTOR = "refactor"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


# Default categories for pylint checks - used when no categories are specified
DEFAULT_CATEGORIES: Set[PylintMessageType] = {
    PylintMessageType.ERROR,
    PylintMessageType.FATAL,
}


class PylintMessage(NamedTuple):
    """Represents a single Pylint message."""

    type: str
    module: str
    obj: str
    line: int
    column: int
    # endLine and endColumn missing
    path: str
    symbol: str
    message: str
    message_id: str


class PylintResult(NamedTuple):
    """Represents the overall result of a Pylint run."""

    return_code: int
    messages: List[PylintMessage]
    error: Optional[str] = None  # Capture any execution errors
    raw_output: Optional[str] = None  # Capture raw output from pylint

    def get_message_ids(self) -> Set[str]:
        """Returns a set of all unique message IDs."""
        return {message.message_id for message in self.messages}

    def get_messages_filtered_by_message_id(
        self, message_id: str
    ) -> List[PylintMessage]:
        """Returns a list of messages filtered by the given message ID."""
        return [
            message for message in self.messages if message.message_id == message_id
        ]


# For backward compatibility
PylintCategory = PylintMessageType
