from __future__ import annotations
"""
Move log writer for the AI Folder Organizer.

Responsibilities:
- Write a CSV log of move outcomes.

Expected usage:
- The executor builds a list[MoveRecord] while moving files.
- At the end, call write_move_log(records, root_path) and then print the path
  using console.print_move_log_written(...).
"""

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

from .errors import LoggingError
from .models import MoveRecord

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Keep logs inside the scanned root so the project remains self-contained.
DEFAULT_LOG_DIRNAME = ".organizer"
DEFAULT_LOG_SUBDIR = "logs"
DEFAULT_LOG_PREFIX = "move_log"

def write_move_log(
    records: Sequence[MoveRecord] | Iterable[MoveRecord],
    root_path: Path,
    *,
    log_dirname: str = DEFAULT_LOG_DIRNAME,
    log_subdir: str = DEFAULT_LOG_SUBDIR,
    filename_prefix: str = DEFAULT_LOG_PREFIX,
) -> Path:
    """
    Write a CSV move log and return the written file path.

    The log is written under:
        {root_path}/{log_dirname}/{log_subdir}/{filename_prefix}_YYYYmmdd_HHMMSSZ.csv

    Always write a CSV file (even if there are 0 records) so the CLI can
    reliably report a log path.

    Parameters
    ----------
    records:
        Iterable of MoveRecord entries (can be empty).
    root_path:
        The scan/organize root directory where logs should live.
    log_dirname:
        Top-level hidden directory under root (default: ".organizer").
    log_subdir:
        Subdirectory under log_dirname for logs (default: "logs").
    filename_prefix:
        Prefix for the log filename (default: "move_log").

    Returns
    -------
    Path
        The path to the written CSV file.

    Raises
    ------
    LoggingError
        If the log directory cannot be created or the file cannot be written.
    """
    try:
        root_path = Path(root_path)
    except Exception as exc:
        raise LoggingError(f"Invalid root path for logging: {root_path}") from exc

    # Prepare destination directory.
    log_dir = root_path / log_dirname / log_subdir
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise LoggingError(f"Failed to create log directory: {log_dir}") from exc

    # Timestamped filename (UTC) for stable ordering.
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    log_path = log_dir / f"{filename_prefix}_{ts}.csv"

    # Normalize records to a list so we can iterate safely multiple times.
    try:
        record_list: List[MoveRecord] = list(records)
    except Exception as exc:
        raise LoggingError("Failed to materialize move records for logging.") from exc

    # Convert to DataFrame with stable, explicit column order.
    rows = []
    for r in record_list:
        d = asdict(r)

        # Ensure Paths serialize as strings
        d["old_path"] = str(r.old_path)
        d["new_path"] = str(r.new_path)

        # Confidence can be None; keep as-is (pandas will write blank)
        d["confidence"] = r.confidence

        rows.append(d)

    columns = [
        "file_name",
        "old_path",
        "new_path",
        "category",
        "classifier_type",
        "confidence",
        "status",
    ]

    df = pd.DataFrame(rows, columns=columns)

    # Write CSV
    try:
        df.to_csv(log_path, index=False)
    except Exception as exc:
        raise LoggingError(f"Failed to write move log to '{log_path}'.") from exc

    return log_path