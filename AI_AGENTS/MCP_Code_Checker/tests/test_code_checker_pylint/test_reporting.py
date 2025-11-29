"""Unit tests for pylint reporting module."""

from src.code_checker_pylint.models import PylintMessage, PylintResult
from src.code_checker_pylint.reporting import (
    get_direct_instruction_for_pylint_code,
    get_prompt_for_known_pylint_code,
    get_prompt_for_unknown_pylint_code,
)


class TestGetDirectInstructionForPylintCode:
    """Test cases for get_direct_instruction_for_pylint_code function."""

    def test_known_codes(self) -> None:
        """Test instructions for known pylint codes."""
        # Test a few known codes
        instruction_r0902 = get_direct_instruction_for_pylint_code("R0902")
        assert (
            instruction_r0902 is not None and "Refactor the class" in instruction_r0902
        )

        instruction_c0411 = get_direct_instruction_for_pylint_code("C0411")
        assert (
            instruction_c0411 is not None
            and "Organize your imports" in instruction_c0411
        )

        instruction_w0612 = get_direct_instruction_for_pylint_code("W0612")
        assert (
            instruction_w0612 is not None
            and "use the variable or remove" in instruction_w0612
        )

        instruction_w0621 = get_direct_instruction_for_pylint_code("W0621")
        assert (
            instruction_w0621 is not None
            and "Avoid shadowing variables" in instruction_w0621
        )

        instruction_w0311 = get_direct_instruction_for_pylint_code("W0311")
        assert instruction_w0311 is not None and "4 spaces" in instruction_w0311

    def test_unknown_code(self) -> None:
        """Test that unknown codes return None."""
        assert get_direct_instruction_for_pylint_code("X9999") is None
        assert get_direct_instruction_for_pylint_code("") is None
        assert get_direct_instruction_for_pylint_code("INVALID") is None


class TestGetPromptForKnownPylintCode:
    """Test cases for get_prompt_for_known_pylint_code function."""

    def test_known_code_with_messages(self) -> None:
        """Test generating prompt for a known code with messages."""
        messages = [
            PylintMessage(
                type="warning",
                module="test_module",
                obj="test_function",
                line=10,
                column=5,
                path="/home/user/project/src/test.py",
                symbol="unused-variable",
                message="Unused variable 'x'",
                message_id="W0612",
            ),
            PylintMessage(
                type="warning",
                module="test_module",
                obj="another_function",
                line=20,
                column=10,
                path="/home/user/project/src/test.py",
                symbol="unused-variable",
                message="Unused variable 'y'",
                message_id="W0612",
            ),
        ]

        result = PylintResult(return_code=0, messages=messages)
        prompt = get_prompt_for_known_pylint_code("W0612", "/home/user/project", result)

        assert prompt is not None
        assert "W0612" in prompt
        assert "use the variable or remove" in prompt
        # Check that the normalized path appears in the prompt
        # The path could be represented as src/test.py or src\\\\test.py in JSON
        assert any(
            path in prompt for path in ["src/test.py", "src\\\\test.py", "src\\test.py"]
        )
        assert "test_function" in prompt
        assert "another_function" in prompt
        assert "line" in prompt
        assert "10" in prompt
        assert "20" in prompt

    def test_unknown_code_returns_none(self) -> None:
        """Test that unknown codes return None."""
        messages = [
            PylintMessage(
                type="error",
                module="test",
                obj="",
                line=1,
                column=1,
                path="test.py",
                symbol="unknown",
                message="Test",
                message_id="X9999",
            ),
        ]

        result = PylintResult(return_code=0, messages=messages)
        prompt = get_prompt_for_known_pylint_code("X9999", "/project", result)

        assert prompt is None


class TestGetPromptForUnknownPylintCode:
    """Test cases for get_prompt_for_unknown_pylint_code function."""

    def test_unknown_code_prompt(self) -> None:
        """Test generating prompt for an unknown code."""
        messages = [
            PylintMessage(
                type="error",
                module="test_module",
                obj="test_function",
                line=10,
                column=5,
                path="/home/user/project/src/test.py",
                symbol="some-unknown-check",
                message="Some unknown issue",
                message_id="X9999",
            ),
        ]

        result = PylintResult(return_code=0, messages=messages)
        prompt = get_prompt_for_unknown_pylint_code(
            "X9999", "/home/user/project", result
        )

        assert prompt is not None
        assert "X9999" in prompt
        assert "some-unknown-check" in prompt
        assert "Please do two things:" in prompt
        assert "provide 1 direct instruction" in prompt
        assert "apply that instruction" in prompt
        # Check that the normalized path appears in the prompt
        # The path could be represented as src/test.py or src\\\\test.py in JSON
        assert any(
            path in prompt for path in ["src/test.py", "src\\\\test.py", "src\\test.py"]
        )
        assert "test_function" in prompt
        assert "Some unknown issue" in prompt

    def test_multiple_messages_same_code(self) -> None:
        """Test generating prompt for multiple messages with the same unknown code."""
        messages = [
            PylintMessage(
                type="error",
                module="module1",
                obj="func1",
                line=10,
                column=5,
                path="/project/src/file1.py",
                symbol="custom-check",
                message="Issue 1",
                message_id="Z0001",
            ),
            PylintMessage(
                type="error",
                module="module2",
                obj="func2",
                line=20,
                column=10,
                path="/project/src/file2.py",
                symbol="custom-check",
                message="Issue 2",
                message_id="Z0001",
            ),
        ]

        result = PylintResult(return_code=0, messages=messages)
        prompt = get_prompt_for_unknown_pylint_code("Z0001", "/project", result)

        assert prompt is not None
        assert "Z0001" in prompt
        assert "custom-check" in prompt
        assert "module1" in prompt
        assert "module2" in prompt
        assert "func1" in prompt
        assert "func2" in prompt
        # Check that normalized paths appear in the prompt
        # The paths could be represented as src/file.py or src\\\\file.py in JSON
        assert any(
            path in prompt
            for path in ["src/file1.py", "src\\\\file1.py", "src\\file1.py"]
        )
        assert any(
            path in prompt
            for path in ["src/file2.py", "src\\\\file2.py", "src\\file2.py"]
        )
