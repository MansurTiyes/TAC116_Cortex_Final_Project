from __future__ import annotations
"""
Planner for the AI Folder Organizer.

Responsibilities:

- Take the classified DataFrame + root path.
- Compute aggregate stats per category:
  * file_count
  * total_size_mb
  * avg_confidence
  * 2–3 sample filenames
- Compute global stats:
  * total_files
  * total_size_mb
  * num_uncertain (files in 'uncertain_review')
- Return a PlanSummary dataclass that the CLI and executor can use.

Input DataFrame is expected to come from classifier.classify_files and must
include at least:

- file_name        (str)
- extension        (str)
- full_path        (str)
- size_bytes       (int)
- modified_time    (float)
- category         (str)
- classifier_type  (str; "rule" or "llm")
- confidence       (float or None)
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .errors import ClassificationError
from .models import PlanSummary, PlanCategorySummary
from .classifier import CATEGORIES

def build_plan(df: pd.DataFrame, root_path: Path) -> PlanSummary:
    """
    Build a PlanSummary from a classified DataFrame and root path.

    Parameters
    ----------
    df : pd.DataFrame
        Classified DataFrame with required columns (see module docstring).
    root_path : Path
        Root folder for this plan.

    Returns
    -------
    PlanSummary
        Structured summary with per-category aggregates and global stats.

    Raises
    ------
    ClassificationError
        If the DataFrame is missing required classification columns.
    """
    required_columns = {
        "file_name",
        "extension",
        "full_path",
        "size_bytes",
        "modified_time",
        "category",
        "classifier_type",
        "confidence",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ClassificationError(
            f"DataFrame is missing required columns for planning: {sorted(missing)}"
        )

    # Work on a copy to avoid surprising callers.
    df = df.copy()

    # Ensure types we rely on are sensible.
    df["size_bytes"] = df["size_bytes"].astype(float)
    # confidence may be None / NaN; keep as numeric where possible.
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    # Global stats
    total_files = int(len(df))
    total_size_mb = float(df["size_bytes"].sum() / (1024 * 1024))

    num_uncertain = int((df["category"] == "uncertain_review").sum())

    # Per-category summaries
    categories: List[PlanCategorySummary] = []

    # Group by category; skip rows without a category just in case.
    grouped = df.dropna(subset=["category"]).groupby("category", sort=False)

    # Iterate categories in the canonical order from CATEGORIES, but only if present,
    # then add any unexpected categories (for robustness).
    seen = set()

    def _make_summary_for_category(cat: str) -> Optional[PlanCategorySummary]:
        if cat not in grouped.groups:
            return None
        group = grouped.get_group(cat)

        file_count = int(len(group))
        total_size_mb_cat = float(group["size_bytes"].sum() / (1024 * 1024))

        # For avg_confidence we consider all non-null confidence values
        # (these are primarily LLM-based, but rule-based entries may have 1.0 too).
        confidences = group["confidence"].dropna()
        avg_confidence: Optional[float]
        if len(confidences) == 0:
            avg_confidence = None
        else:
            avg_confidence = float(confidences.mean())

        # Pick up to 3 sample filenames (stable order by file_name for determinism).
        sample_files = (
            group.sort_values("file_name")["file_name"].head(3).astype(str).tolist()
        )

        return PlanCategorySummary(
            category=cat,
            file_count=file_count,
            total_size_mb=total_size_mb_cat,
            avg_confidence=avg_confidence,
            sample_files=sample_files,
        )

    # First, use known categories in a friendly, stable order.
    for cat in CATEGORIES:
        summary = _make_summary_for_category(cat)
        if summary is not None:
            categories.append(summary)
            seen.add(cat)

    # Then, handle any unexpected categories the classifier might have produced.
    for cat in grouped.groups.keys():
        if cat in seen:
            continue
        summary = _make_summary_for_category(cat)
        if summary is not None:
            categories.append(summary)

    # Build and return final PlanSummary
    plan = PlanSummary(
        root_path=root_path,
        df=df,
        categories=categories,
        total_files=total_files,
        total_size_mb=total_size_mb,
        num_uncertain=num_uncertain,
    )

    return plan

