"""
Functions for generating reports and prompts from pylint analysis results.
"""

import json
import logging
from typing import Optional, Set

import structlog

from src.log_utils import log_function_call

from .models import DEFAULT_CATEGORIES, PylintMessageType, PylintResult
from .runners import get_pylint_results
from .utils import filter_pylint_codes_by_category, normalize_path

logger = logging.getLogger(__name__)
structured_logger = structlog.get_logger(__name__)


def get_direct_instruction_for_pylint_code(code: str) -> Optional[str]:
    """
    Provides a direct instruction for a given Pylint code.

    Args:
        code: The Pylint code (e.g., "R0902", "C0411", "W0612").

    Returns:
        A direct instruction string or None if the code is not recognized.
    """
    instructions = {
        "R0902": "Refactor the class by breaking it into smaller classes or using data structures to reduce the number of instance attributes.",  # too-many-instance-attributes
        "C0411": "Organize your imports into three groups: standard library imports, third-party library imports, and local application/project imports, separated by blank lines, with each group sorted alphabetically.",  # wrong-import-order
        "W0612": "Either use the variable or remove the variable assignment if it is not needed.",  # unused-variable
        "W0621": "Avoid shadowing variables from outer scopes.",  # redefined-outer-name
        "W0311": "Ensure consistent indentation using 4 spaces for each level of nesting.",  # bad-indentation
        "W0718": "Explicitly catch only the specific exceptions you expect and handle them appropriately, rather than using a bare `except:` clause.",  # broad-exception-caught
        "E0601": "Ensure a variable is assigned a value before it is used within its scope.",  # used-before-assignment
        "E0602": "Before using a variable, ensure it is either defined within the current scope (e.g., function, class, or global) or imported correctly from a module.",  # undefined-variable
        "E1120": "Provide a value for each parameter in the function call that doesn't have a default value.",  # no-value-for-parameter
        "E0401": "Verify that the module or package you are trying to import is installed and accessible in your Python environment, and that the import statement matches its name and location.",  # import-error
        "E0611": "Verify that the name you are trying to import (e.g., function, class, variable) actually exists within the specified module or submodule and that the spelling is correct.",  # no-name-in-module
        "W4903": "Replace the deprecated argument with its recommended alternative; consult the documentation for the function or method to identify the correct replacement.",  # deprecated-argument
        "W1203": "Use string formatting with the `%` operator or `.format()` method when passing variables to logging functions instead of f-strings; for example, `logging.info('Value: %s', my_variable)` or `logging.info('Value: {}'.format(my_variable))`",  # logging-fstring-interpolation
        "W0613": "Remove the unused argument from the function definition if it is not needed, or if it is needed for compatibility or future use, use `_` as the argument's name to indicate it's intentionally unused, or use it within the function logic.",  # unused-argument
        "C0415": "Move the import statement to the beginning of the file, outside of any conditional blocks, functions, or classes; all imports should be at the top of the file.",  # import-outside-toplevel
        "E0704": "Ensure that a `raise` statement without an exception object or exception type only appears inside an `except` block; it should re-raise the exception that was caught, not be used outside of an exception handling context.",  # misplaced-bare-raise
        "E0001": "Carefully review the indicated line (and potentially nearby lines) for syntax errors such as typos, mismatched parentheses/brackets/quotes, invalid operators, or incorrect use of keywords; consult the Python syntax rules to correct the issue.",  # syntax-error
        "R0911": "Refactor the function to reduce the number of return statements, potentially by simplifying the logic or using helper functions.",  # too-many-return-statements
        "W0707": "When raising a new exception inside an except block, use raise ... from original_exception to preserve the original exception's traceback.",  # raise-missing-from
        "E1125": "Fix the function call by using keyword arguments for parameters that are defined as keyword-only in the function signature. Use the parameter name as a keyword when passing the value.",  # missing-kwoa
        "E1101": "Define the missing attribute or method directly in the class declaration. If the attribute or method should be inherited from a parent class, ensure that the parent class is correctly specified in the class definition.",  # no-member
        "E0213": "Ensure the first parameter of instance methods in a class is named 'self'. This parameter represents the instance of the class and is automatically passed when calling the method on an instance.",  # no-self-argument
        "E1123": "Make sure to only use keyword arguments that are defined in the function or method definition you're calling.",  # unexpected-keyword-argument
    }

    return instructions.get(code)


def get_prompt_for_known_pylint_code(
    code: str, project_dir: str, pylint_results: PylintResult
) -> Optional[str]:
    """
    Generate a prompt for a known pylint code with instructions and details.

    Args:
        code: The pylint code (e.g., "E0602")
        project_dir: The project directory path
        pylint_results: The pylint analysis results

    Returns:
        A formatted prompt string or None if no instruction is found for the code
    """
    instruction = get_direct_instruction_for_pylint_code(code)
    if not instruction:
        return None

    pylint_results_filtered = pylint_results.get_messages_filtered_by_message_id(code)
    details_lines = []

    for message in pylint_results_filtered:
        path = normalize_path(message.path, project_dir)
        # Create a dictionary and dump the entire structure
        issue_dict = {
            "module": message.module,
            "obj": message.obj,
            "line": message.line,
            "column": message.column,
            "path": path,
            "message": message.message,
        }
        # Get JSON string for the whole object and add comma for the list format
        details_lines.append(json.dumps(issue_dict, indent=4) + ",")

    details_str = "\n".join(details_lines)
    query = f"""pylint found some issues related to code {code}.
    {instruction}
    Please consider especially the following locations in the source code:
    {details_str}"""
    return query


def get_prompt_for_unknown_pylint_code(
    code: str, project_dir: str, pylint_results: PylintResult
) -> str:
    """
    Generate a prompt for an unknown pylint code with issue details.

    Args:
        code: The pylint code (e.g., "E0602")
        project_dir: The project directory path
        pylint_results: The pylint analysis results

    Returns:
        A formatted prompt string requesting instructions for this code
    """
    pylint_results_filtered = pylint_results.get_messages_filtered_by_message_id(code)

    first_result = next(iter(pylint_results_filtered))
    symbol = first_result.symbol

    details_lines = []
    for message in pylint_results_filtered:
        path = normalize_path(message.path, project_dir)
        # Create a dictionary and dump the entire structure
        issue_dict = {
            "module": message.module,
            "obj": message.obj,
            "line": message.line,
            "column": message.column,
            "path": path,
            "message": message.message,
        }
        # Get JSON string for the whole object and add comma for the list format
        details_lines.append(json.dumps(issue_dict, indent=4) + ",")

    # Store the entire details section in a variable first
    details_str = "\n".join(details_lines)

    query = f"""pylint found some issues related to code {code} / symbol {symbol}.
    
    Please do two things:
    1. Please provide 1 direct instruction on how to fix pylint code "{code}" ({symbol}) in the general comment of the response.
    
    2. Please apply that instruction   
    Please consider especially the following locations in the source code:
    {details_str}"""
    return query


@log_function_call
def get_pylint_prompt(
    project_dir: str,
    categories: Optional[Set[PylintMessageType]] = None,
    disable_codes: Optional[list[str]] = None,
    python_executable: Optional[str] = None,
    target_directories: Optional[list[str]] = None,
) -> Optional[str]:
    """
    Generate a prompt for fixing pylint issues based on the analysis of a project.

    Args:
        project_dir: The path to the project directory to analyze.
        categories: Set of specific pylint categories to filter by. Available categories are:
            - PylintMessageType.CONVENTION: Style conventions (C)
            - PylintMessageType.REFACTOR: Refactoring suggestions (R)
            - PylintMessageType.WARNING: Python-specific warnings (W)
            - PylintMessageType.ERROR: Probable bugs in the code (E)
            - PylintMessageType.FATAL: Critical errors that prevent pylint from working (F)
            Defaults to {ERROR, FATAL} if None.
        disable_codes: Optional list of pylint codes to disable during analysis. Common codes include:
            - C0114: Missing module docstring
            - C0116: Missing function docstring
            - C0301: Line too long
            - C0303: Trailing whitespace
            - C0305: Trailing newlines
            - W0311: Bad indentation
            - W0611: Unused import
            - W1514: Unspecified encoding
        python_executable: Optional path to Python interpreter to use for running tests. If None, defaults to sys.executable.
        target_directories: Optional list of directories to analyze relative to project_dir.
            Defaults to ["src"] and conditionally "tests" if it exists.
            Examples: ["src"], ["src", "tests"], ["mypackage", "tests"], ["."]

    Returns:
        A prompt string with issue details and instructions, or None if no issues were found.
        Returns the error message as a prompt if pylint execution failed (e.g., timeout).
    """
    # Use default categories if none provided
    if categories is None:
        categories = DEFAULT_CATEGORIES

    structured_logger.info(
        "Starting pylint prompt generation",
        project_dir=project_dir,
        categories=[cat.value for cat in categories],
        disable_codes=disable_codes,
    )

    # Default disable codes if none provided
    if disable_codes is None:
        disable_codes = [
            # not required for now
            "C0114",  # doc missing
            "C0116",  # doc missing
            #
            # can be solved with formatting / black
            "C0301",  # line-too-long
            "C0303",  # trailing-whitespace
            "C0305",  # trailing-newlines
            "W0311",  # bad-indentation   - instruction available
            #
            # can be solved with iSort
            "W0611",  # unused-import
            "W1514",  # unspecified-encoding
        ]

    pylint_results = get_pylint_results(
        project_dir,
        disable_codes=disable_codes,
        python_executable=python_executable,
        target_directories=target_directories,
    )

    # Check if there was an error running pylint (e.g., timeout, execution failure)
    if pylint_results.error:
        structured_logger.error(
            "Pylint execution error detected",
            error=pylint_results.error,
            return_code=pylint_results.return_code,
        )
        # Return the error as a prompt so the MCP client gets the error message
        return f"Pylint analysis failed: {pylint_results.error}"

    codes = pylint_results.get_message_ids()
    if len(categories) > 0:
        codes = filter_pylint_codes_by_category(codes, categories=categories)

    if len(codes) > 0:
        code = list(codes)[0]
        structured_logger.info(
            "Pylint issues found, generating prompt",
            total_codes=len(codes),
            first_code=code,
        )
        prompt = get_prompt_for_known_pylint_code(code, project_dir, pylint_results)
        if prompt is not None:
            return prompt
        else:
            prompt = get_prompt_for_unknown_pylint_code(
                code, project_dir=project_dir, pylint_results=pylint_results
            )
            return prompt  # just for the first code
    else:
        structured_logger.info("No pylint issues found", project_dir=project_dir)
        return None
