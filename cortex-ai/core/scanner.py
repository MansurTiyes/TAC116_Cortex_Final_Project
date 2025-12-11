from __future__ import annotations
"""
Filesystem scanner for the AI Folder Organizer.

Responsibilities:

- Recursively walk a root PATH.
- For each file, collect:
  - file_name
  - extension
  - full_path
  - size_bytes
  - modified_time
- Return a Pandas DataFrame with those columns.
- If PATH does not exist, raise PathNotFoundError.
- If PATH exists but is not a directory, raise PathNotDirectoryError.
- If there are no files under PATH, raise NoFilesFoundError.
"""

from pathlib import Path
from typing import List

import os
import pandas as pd

from .errors import PathNotFoundError, PathNotDirectoryError, NoFilesFoundError
from .models import FileRecord

def scan_directory(root: Path) -> pd.DataFrame:
    """
    Recursively scan the given root directory and return a DataFrame of files.

    The returned DataFrame has columns:

    - file_name       (str)
    - extension       (str; lowercase, including leading dot, e.g. ".pdf")
    - full_path       (str; absolute path)
    - size_bytes      (int)
    - modified_time   (float; POSIX timestamp)

    Behavior:

    - If `root` does not exist, raise PathNotFoundError.
    - If `root` exists but is not a directory, raise PathNotDirectoryError.
    - If `root` contains no files (recursively), raise NoFilesFoundError.

    Parameters
    ----------
    root : Path
        Root directory to scan (typically already resolved via utils.paths.resolve_root).

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per discovered file.

    Raises
    ------
    PathNotFoundError
        If the path does not exist.
    PathNotDirectoryError
        If the path is not a directory.
    NoFilesFoundError
        If the directory tree contains no files.
    """
    # Defensive checks in case caller didn't already validate the path.
    if not root.exists():
        raise PathNotFoundError(f"Path not found: {root}")

    if not root.is_dir():
        raise PathNotDirectoryError(f"Provided path is not a directory: {root}")

    records: List[FileRecord] = []

    # Recursively walk the directory tree.
    for dirpath, dirnames, filenames in os.walk(root):
        dir_path = Path(dirpath)

        for name in filenames:
            full_path = dir_path / name

            # Best-effort stat; if a file disappears between walk and stat or
            # is otherwise inaccessible, we simply skip it rather than failing
            # the entire scan.
            try:
                stat = full_path.stat()
            except OSError:
                # You could log this in verbose mode later if desired.
                continue

            record = FileRecord(
                file_name=name,
                extension=full_path.suffix.lower(),
                full_path=full_path,
                size_bytes=stat.st_size,
                modified_time=stat.st_mtime,
            )
            records.append(record)

    if not records:
        # No files found anywhere under root.
        raise NoFilesFoundError(f"No files found under '{root}'.")

    # Convert dataclasses into a DataFrame with the required columns.
    data = [
        {
            "file_name": r.file_name,
            "extension": r.extension,
            "full_path": str(r.full_path),
            "size_bytes": r.size_bytes,
            "modified_time": r.modified_time,
        }
        for r in records
    ]

    df = pd.DataFrame(
        data,
        columns=[
            "file_name",
            "extension",
            "full_path",
            "size_bytes",
            "modified_time",
        ],
    )

    return df