from __future__ import annotations

"""
Executor for the AI Folder Organizer.

Responsibilities (MVP):
- Take a PlanSummary + root path.
- For each row in plan.df:
    - Compute target_dir = root / category
    - Create directory if needed
    - Move old_path -> new_path
    - Record outcome (success / failed: <reason> / skipped)
- Build a moves DataFrame with:
    file_name, old_path, new_path, category, classifier_type, confidence, status
- Pass move records to logger.write_move_log(...)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import shutil
import pandas as pd

from .errors import FileMoveError
from .models import MoveRecord, PlanSummary
from .logger import write_move_log

def apply_plan(plan: PlanSummary, root_path: Path) -> tuple[pd.DataFrame, Path]:
    """
    Apply the organization plan by moving files into category subfolders.

    Parameters
    ----------
    plan : PlanSummary
        The plan to apply. Expects plan.df columns at least:
        - file_name, full_path, category, classifier_type, confidence
    root_path : Path
        Root folder where category directories will be created.

    Returns
    -------
    (moves_df, log_path) : (pd.DataFrame, Path)
        moves_df includes one row per file attempt with columns:
            file_name, old_path, new_path, category, classifier_type, confidence, status
        log_path is the CSV path written by logger.write_move_log(...)

    Raises
    ------
    FileMoveError
        For catastrophic failures that prevent execution from proceeding at all.
        (Most per-file failures are recorded in the move log and do not abort.)
    """
    try:
        root_path = Path(root_path)
    except Exception as exc:
        raise FileMoveError(f"Invalid root path: {root_path}") from exc

    # Defensive: ensure plan.df exists and looks like a DataFrame.
    if plan is None or getattr(plan, "df", None) is None:
        raise FileMoveError("PlanSummary is missing required DataFrame 'df'.")

    df = plan.df

    required_cols = {"file_name", "full_path", "category", "classifier_type", "confidence"}
    missing = required_cols.difference(set(df.columns))
    if missing:
        raise FileMoveError(f"Plan DataFrame is missing required columns: {sorted(missing)}")

    # Track category directory creation failures to avoid repeated mkdir attempts.
    category_dir_failures: Dict[str, str] = {}

    move_records: List[MoveRecord] = []
    move_rows: List[dict] = []

    for _, row in df.iterrows():
        file_name = str(row.get("file_name", ""))
        old_path_raw = row.get("full_path", "")
        category_raw = row.get("category", None)

        category = str(category_raw) if category_raw not in (None, "", "nan") else "uncertain_review"
        classifier_type = row.get("classifier_type", None)
        confidence = row.get("confidence", None)

        old_path = Path(str(old_path_raw))

        target_dir = root_path / category
        new_path = target_dir / old_path.name

        status: str

        # If category directory already known to fail, mark as failed quickly.
        if category in category_dir_failures:
            reason = category_dir_failures[category]
            status = f"failed: {reason}"
            _append_move(
                move_records,
                move_rows,
                file_name=file_name,
                old_path=old_path,
                new_path=new_path,
                category=category,
                classifier_type=classifier_type,
                confidence=confidence,
                status=status,
            )
            continue

        # Create category dir if needed.
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            reason = f"could not create target directory '{target_dir}': {exc}"
            category_dir_failures[category] = reason
            status = f"failed: {reason}"
            _append_move(
                move_records,
                move_rows,
                file_name=file_name,
                old_path=old_path,
                new_path=new_path,
                category=category,
                classifier_type=classifier_type,
                confidence=confidence,
                status=status,
            )
            continue

        # If old path doesn't exist, treat as per-file failure.
        if not old_path.exists():
            status = "failed: source file not found"
            _append_move(
                move_records,
                move_rows,
                file_name=file_name,
                old_path=old_path,
                new_path=new_path,
                category=category,
                classifier_type=classifier_type,
                confidence=confidence,
                status=status,
            )
            continue

        # Avoid moving a file onto itself.
        try:
            if old_path.resolve() == new_path.resolve():
                status = "skipped: already in target"
                _append_move(
                    move_records,
                    move_rows,
                    file_name=file_name,
                    old_path=old_path,
                    new_path=new_path,
                    category=category,
                    classifier_type=classifier_type,
                    confidence=confidence,
                    status=status,
                )
                continue
        except Exception:
            # If resolve fails (permissions/symlinks), just proceed with best-effort move.
            pass

        # If destination exists, pick a non-colliding filename.
        new_path_final = _resolve_collision(new_path)

        # Perform move (per-file failure should not abort).
        try:
            shutil.move(str(old_path), str(new_path_final))
            status = "success"
        except Exception as exc:
            status = f"failed: {exc}"

        _append_move(
            move_records,
            move_rows,
            file_name=file_name,
            old_path=old_path,
            new_path=new_path_final,
            category=category,
            classifier_type=classifier_type,
            confidence=confidence,
            status=status,
        )

    # Build DataFrame for callers / future summarization needs.
    moves_df = pd.DataFrame(
        move_rows,
        columns=[
            "file_name",
            "old_path",
            "new_path",
            "category",
            "classifier_type",
            "confidence",
            "status",
        ],
    )

    # Write CSV log using logger (expects MoveRecord sequence based on your logger.py).
    log_path = write_move_log(move_records, root_path=root_path)

    return moves_df, log_path


def _resolve_collision(path: Path) -> Path:
    """
    If 'path' already exists, produce a new Path by appending ' (n)' before suffix.

    Example:
      report.pdf -> report (1).pdf -> report (2).pdf
    """
    if not path.exists():
        return path

    parent = path.parent
    stem = path.stem
    suffix = path.suffix

    i = 1
    while True:
        candidate = parent / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def _append_move(
    record_list: List[MoveRecord],
    row_list: List[dict],
    *,
    file_name: str,
    old_path: Path,
    new_path: Path,
    category: str,
    classifier_type: object,
    confidence: object,
    status: str,
) -> None:
    """
    Append a MoveRecord (for logger) and a dict row (for moves_df).
    """
    # Normalize classifier_type to the expected literals if possible.
    cls_type: str
    if classifier_type in ("rule", "llm"):
        cls_type = str(classifier_type)
    else:
        cls_type = str(classifier_type) if classifier_type is not None else "rule"

    conf_val: Optional[float]
    try:
        conf_val = None if confidence is None else float(confidence)
    except Exception:
        conf_val = None

    record = MoveRecord(
        file_name=file_name,
        old_path=old_path,
        new_path=new_path,
        category=category,
        classifier_type=cls_type,  # type: ignore[arg-type]
        confidence=conf_val,
        status=status,
    )
    record_list.append(record)

    row_list.append(
        {
            "file_name": file_name,
            "old_path": str(old_path),
            "new_path": str(new_path),
            "category": category,
            "classifier_type": cls_type,
            "confidence": conf_val,
            "status": status,
        }
    )
