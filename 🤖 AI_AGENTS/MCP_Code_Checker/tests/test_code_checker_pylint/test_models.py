"""Unit tests for pylint models module."""

from src.code_checker_pylint.models import (
    DEFAULT_CATEGORIES,
    PylintMessage,
    PylintMessageType,
    PylintResult,
)


def test_pylint_message_type_enum() -> None:
    """Test PylintMessageType enum values."""
    assert PylintMessageType.CONVENTION.value == "convention"
    assert PylintMessageType.REFACTOR.value == "refactor"
    assert PylintMessageType.WARNING.value == "warning"
    assert PylintMessageType.ERROR.value == "error"
    assert PylintMessageType.FATAL.value == "fatal"


def test_default_categories() -> None:
    """Test default categories."""
    assert DEFAULT_CATEGORIES == {PylintMessageType.ERROR, PylintMessageType.FATAL}


def test_pylint_message_creation() -> None:
    """Test PylintMessage named tuple creation."""
    msg = PylintMessage(
        type="error",
        module="test_module",
        obj="test_function",
        line=10,
        column=5,
        path="/path/to/file.py",
        symbol="undefined-variable",
        message="Undefined variable 'x'",
        message_id="E0602",
    )

    assert msg.type == "error"
    assert msg.module == "test_module"
    assert msg.obj == "test_function"
    assert msg.line == 10
    assert msg.column == 5
    assert msg.path == "/path/to/file.py"
    assert msg.symbol == "undefined-variable"
    assert msg.message == "Undefined variable 'x'"
    assert msg.message_id == "E0602"


def test_pylint_result_creation() -> None:
    """Test PylintResult named tuple creation."""
    messages = [
        PylintMessage(
            type="error",
            module="test_module",
            obj="test_function",
            line=10,
            column=5,
            path="/path/to/file.py",
            symbol="undefined-variable",
            message="Undefined variable 'x'",
            message_id="E0602",
        ),
        PylintMessage(
            type="warning",
            module="test_module",
            obj="test_function2",
            line=20,
            column=10,
            path="/path/to/file.py",
            symbol="unused-variable",
            message="Unused variable 'y'",
            message_id="W0612",
        ),
    ]

    result = PylintResult(
        return_code=0,
        messages=messages,
        error=None,
        raw_output="test output",
    )

    assert result.return_code == 0
    assert len(result.messages) == 2
    assert result.error is None
    assert result.raw_output == "test output"


def test_pylint_result_get_message_ids() -> None:
    """Test PylintResult.get_message_ids method."""
    messages = [
        PylintMessage(
            type="error",
            module="test",
            obj="",
            line=1,
            column=1,
            path="test.py",
            symbol="undefined-variable",
            message="Test",
            message_id="E0602",
        ),
        PylintMessage(
            type="warning",
            module="test",
            obj="",
            line=2,
            column=1,
            path="test.py",
            symbol="unused-variable",
            message="Test",
            message_id="W0612",
        ),
        PylintMessage(
            type="error",
            module="test",
            obj="",
            line=3,
            column=1,
            path="test.py",
            symbol="undefined-variable",
            message="Test",
            message_id="E0602",  # Duplicate
        ),
    ]

    result = PylintResult(return_code=0, messages=messages)
    message_ids = result.get_message_ids()

    assert message_ids == {"E0602", "W0612"}


def test_pylint_result_get_messages_filtered_by_message_id() -> None:
    """Test PylintResult.get_messages_filtered_by_message_id method."""
    messages = [
        PylintMessage(
            type="error",
            module="test1",
            obj="",
            line=1,
            column=1,
            path="test.py",
            symbol="undefined-variable",
            message="Test1",
            message_id="E0602",
        ),
        PylintMessage(
            type="warning",
            module="test2",
            obj="",
            line=2,
            column=1,
            path="test.py",
            symbol="unused-variable",
            message="Test2",
            message_id="W0612",
        ),
        PylintMessage(
            type="error",
            module="test3",
            obj="",
            line=3,
            column=1,
            path="test.py",
            symbol="undefined-variable",
            message="Test3",
            message_id="E0602",
        ),
    ]

    result = PylintResult(return_code=0, messages=messages)
    filtered = result.get_messages_filtered_by_message_id("E0602")

    assert len(filtered) == 2
    assert all(msg.message_id == "E0602" for msg in filtered)
    assert filtered[0].module == "test1"
    assert filtered[1].module == "test3"
