"""
File operation utilities.
"""

from typing import Optional


def read_file(file_path: str, encoding: Optional[str] = "utf-8") -> str:
    """
    Read the contents of a file with automatic encoding fallback.

    Args:
        file_path: Path to the file to read
        encoding: Initial encoding to try (defaults to utf-8)

    Returns:
        The contents of the file as a string

    Raises:
        FileNotFoundError: If the file does not exist
        PermissionError: If access to the file is denied
    """
    try:
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()
