"""
Path utilities for the AI Folder Organizer.

This module provides helpers for resolving and validating filesystem paths
used by the CLI. All user-specified root paths should be passed through
resolve_root() before being used elsewhere.
"""

from __future__ import annotations

from pathlib import Path

from ..core.errors import PathNotFoundError, PathNotDirectoryError

def resolve_root(path_str: str) -> Path:
    """
    Resolve a user-provided path string into an absolute directory Path.

    Steps:
    - Expand '~' and environment variables.
    - Resolve to an absolute path.
    - Ensure the path exists.
    - Ensure the path is a directory (not a file).

    Raises:
    - PathNotFoundError: if the resolved path does not exist.
    - PathNotDirectoryError: if the path exists but is not a directory.
    """
    # Expand '~' and environment variables, then resolve symlinks / relative parts.
    root = Path(path_str).expanduser().resolve()

    if not root.exists():
        raise PathNotFoundError(f"Path not found: {root}")

    if not root.is_dir():
        raise PathNotDirectoryError(f"Provided path is not a directory: {root}")

    return root

