from __future__ import annotations
"""
Core domain models for the AI Folder Organizer.

These dataclasses define the structured data passed between:
- scanner -> classifier -> planner -> executor -> CLI
and allow the console layer (Rich output) to render summaries without
knowing internal implementation details.

They are intentionally small, immutable-ish containers with no heavy logic.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Basic types
# ---------------------------------------------------------------------------

ClassifierType = Literal["rule", "llm"]

@dataclass
class FileRecord:
    """
    Metadata for a single file discovered during scanning.

    This is the "raw" view used before and during classification. It can be
    constructed directly from filesystem stats (name, path, size, mtime) and
    optionally used for LLM prompts.
    """
    file_name: str
    extension: str
    full_path: Path
    size_bytes: int
    modified_time: float  # POSIX timestamp (time.time() / stat.st_mtime)

@dataclass
class LlmClassificationResult:
    """
    Result of classifying a single file via the LLM.

    This is a convenient structured form for outputs from the LLM client
    before they are merged back into the main DataFrame.
    """

    full_path: Path
    category: str
    confidence: float
    reason: Optional[str] = None

# ---------------------------------------------------------------------------
# Planning / summary models
# ---------------------------------------------------------------------------

@dataclass
class PlanCategorySummary:
    """
    Aggregated summary for a single category in the proposed organization plan.

    - Category
    - Number of files
    - Total size in MB
    - Average confidence (for LLM-based classifications)
    - Sample filenames (used for the 'Sample files:' section)
    """

    category: str
    file_count: int
    total_size_mb: float
    avg_confidence: Optional[float]
    sample_files: List[str]

@dataclass
class PlanSummary:
    """
    Full organization plan summary for a single run.

    Produced by the planner after classification. It is the main object
    consumed by the console helpers to render Rich tables and overview text,
    and by the executor to know which files belong to which category.
    """

    root_path: Path
    # Classified DataFrame with columns at least:
    # - file_name, extension, full_path, size_bytes, modified_time
    # - category, classifier_type, confidence
    df: pd.DataFrame

    categories: List[PlanCategorySummary]

    total_files: int
    total_size_mb: float
    num_uncertain: int  # number of files in 'uncertain_review'

# ---------------------------------------------------------------------------
# Move logging models
# ---------------------------------------------------------------------------

@dataclass
class MoveRecord:
    """
    Represents the outcome of moving a single file when applying a plan.

    This is the in-memory representation that can be converted to a DataFrame
    and written out as a CSV move log.

    - file_name
    - old_path
    - new_path
    - category
    - classifier_type
    - confidence
    - status (e.g. 'success', 'failed: <reason>')
    """

    file_name: str
    old_path: Path
    new_path: Path
    category: str
    classifier_type: ClassifierType
    confidence: Optional[float]
    status: str
