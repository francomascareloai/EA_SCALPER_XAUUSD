"""Unit tests for pylint parsers module."""

import json

from src.code_checker_pylint.parsers import parse_pylint_json_output


class TestParsePylintJsonOutput:
    """Test cases for parse_pylint_json_output function."""

    def test_parse_valid_json_output(self) -> None:
        """Test parsing valid JSON output from pylint."""
        json_data = [
            {
                "type": "error",
                "module": "test_module",
                "obj": "test_function",
                "line": 10,
                "column": 5,
                "path": "/path/to/file.py",
                "symbol": "undefined-variable",
                "message": "Undefined variable 'x'",
                "message-id": "E0602",
            },
            {
                "type": "warning",
                "module": "test_module",
                "obj": "test_function2",
                "line": 20,
                "column": 10,
                "path": "/path/to/file.py",
                "symbol": "unused-variable",
                "message": "Unused variable 'y'",
                "message-id": "W0612",
            },
        ]
        raw_output = json.dumps(json_data)

        messages, error = parse_pylint_json_output(raw_output)

        assert error is None
        assert len(messages) == 2

        # Check first message
        assert messages[0].type == "error"
        assert messages[0].module == "test_module"
        assert messages[0].obj == "test_function"
        assert messages[0].line == 10
        assert messages[0].column == 5
        assert messages[0].path == "/path/to/file.py"
        assert messages[0].symbol == "undefined-variable"
        assert messages[0].message == "Undefined variable 'x'"
        assert messages[0].message_id == "E0602"

        # Check second message
        assert messages[1].type == "warning"
        assert messages[1].message_id == "W0612"

    def test_parse_empty_output(self) -> None:
        """Test parsing empty output."""
        messages, error = parse_pylint_json_output("")

        assert error is None
        assert messages == []

    def test_parse_whitespace_only_output(self) -> None:
        """Test parsing whitespace-only output."""
        messages, error = parse_pylint_json_output("   \n  \t  ")

        assert error is None
        assert messages == []

    def test_parse_empty_json_array(self) -> None:
        """Test parsing empty JSON array."""
        messages, error = parse_pylint_json_output("[]")

        assert error is None
        assert messages == []

    def test_parse_invalid_json(self) -> None:
        """Test parsing invalid JSON."""
        raw_output = "This is not valid JSON"

        messages, error = parse_pylint_json_output(raw_output)

        assert messages == []
        assert error is not None
        assert "Failed to parse Pylint JSON output" in error
        assert "This is not valid JSON" in error

    def test_parse_json_object_instead_of_array(self) -> None:
        """Test parsing JSON object instead of expected array."""
        raw_output = json.dumps({"type": "error", "message": "test"})

        messages, error = parse_pylint_json_output(raw_output)

        assert messages == []
        assert error is not None
        assert "Expected JSON array from pylint, got dict" in error

    def test_parse_json_with_non_dict_items(self) -> None:
        """Test parsing JSON array with non-dict items."""
        json_data = [
            {
                "type": "error",
                "module": "test_module",
                "obj": "",
                "line": 1,
                "column": 1,
                "path": "test.py",
                "symbol": "test",
                "message": "Test message",
                "message-id": "E0001",
            },
            "This is a string, not a dict",
            42,
            {
                "type": "warning",
                "module": "test_module2",
                "obj": "",
                "line": 2,
                "column": 1,
                "path": "test2.py",
                "symbol": "test2",
                "message": "Test message 2",
                "message-id": "W0001",
            },
        ]
        raw_output = json.dumps(json_data)

        messages, error = parse_pylint_json_output(raw_output)

        assert error is None
        assert len(messages) == 2  # Only dict items are processed
        assert messages[0].message_id == "E0001"
        assert messages[1].message_id == "W0001"

    def test_parse_json_with_missing_fields(self) -> None:
        """Test parsing JSON with missing fields."""
        json_data = [
            {
                "type": "error",
                # Missing most fields
            },
            {
                "type": "warning",
                "module": "test_module",
                "line": 10,
                # Missing other fields
            },
        ]
        raw_output = json.dumps(json_data)

        messages, error = parse_pylint_json_output(raw_output)

        assert error is None
        assert len(messages) == 2

        # Check default values for missing fields
        assert messages[0].type == "error"
        assert messages[0].module == ""
        assert messages[0].obj == ""
        assert messages[0].line == -1
        assert messages[0].column == -1
        assert messages[0].path == ""
        assert messages[0].symbol == ""
        assert messages[0].message == ""
        assert messages[0].message_id == ""

    def test_parse_very_long_output(self) -> None:
        """Test parsing very long output (error message truncation)."""
        raw_output = "x" * 300  # Invalid JSON, longer than 200 chars

        messages, error = parse_pylint_json_output(raw_output)

        assert messages == []
        assert error is not None
        assert "Failed to parse Pylint JSON output" in error
        assert "First 200 chars of output:" in error
        assert "xxx..." in error  # Should be truncated
