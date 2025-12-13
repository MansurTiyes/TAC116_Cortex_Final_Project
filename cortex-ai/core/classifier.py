from __future__ import annotations
"""
File classification logic for the AI Folder Organizer.

Responsibilities:

- Define a small rule-based classifier using DEFAULT_EXTENSION_MAP.
- Provide classify_files(df, min_confidence):
  * Initialize 'category', 'classifier_type', 'confidence' columns.
  * Apply extension-based rules (classifier_type = "rule", confidence = 1.0).
  * For remaining files without a category:
      - Build simple metadata list.
      - Call _llm_classify_files(...) to get category + confidence.
  * Apply min_confidence threshold:
      - Any LLM result with confidence < min_confidence is mapped to
        'uncertain_review'.
- Surface clean, domain-specific errors:
  * MissingApiKeyError (from env helper) if OPENAI_API_KEY is missing.
  * LlmUnavailableError when the LLM API call/network fails.
  * LlmResponseParseError when the LLM response cannot be parsed as expected.
  * ClassificationError for unexpected internal issues.

The CLI layer (Typer commands) will catch these errors and use console helpers
to print friendly user-facing messages.
"""

from pathlib import Path
from typing import Dict, List, Any

import json
import pandas as pd
from openai import OpenAI

from .errors import (
    MissingApiKeyError,
    LlmUnavailableError,
    LlmResponseParseError,
    ClassificationError
)
from .models import LlmClassificationResult
from ..utils.env import get_llm_api_key

# ---------------------------------------------------------------------------
# Categories and rule-based mapping
# ---------------------------------------------------------------------------

CATEGORIES: List[str] = [
    "school_work",
    "code_projects",
    "photos_images",
    "videos_media",
    "music_audio",
    "archives_installers",
    "documents_misc",
    "trash_or_temp",
    "uncertain_review",
]

# Basic rule-based mapping from extension to category.
DEFAULT_EXTENSION_MAP: Dict[str, str] = {
    ".py": "code_projects",
    ".java": "code_projects",
    ".cpp": "code_projects",
    ".c": "code_projects",
    ".h": "code_projects",
    ".hpp": "code_projects",
    ".js": "code_projects",
    ".ts": "code_projects",
    ".ipynb": "code_projects",

    ".jpg": "photos_images",
    ".jpeg": "photos_images",
    ".png": "photos_images",
    ".heic": "photos_images",
    ".gif": "photos_images",
    ".tif": "photos_images",
    ".tiff": "photos_images",

    ".mov": "videos_media",
    ".mp4": "videos_media",
    ".mkv": "videos_media",
    ".avi": "videos_media",

    ".mp3": "music_audio",
    ".wav": "music_audio",
    ".flac": "music_audio",
    ".m4a": "music_audio",

    ".zip": "archives_installers",
    ".tar": "archives_installers",
    ".tar.gz": "archives_installers",
    ".tgz": "archives_installers",
    ".rar": "archives_installers",
    ".dmg": "archives_installers",
    ".pkg": "archives_installers",
    ".exe": "archives_installers",

    ".pdf": "documents_misc",
    ".doc": "documents_misc",
    ".docx": "documents_misc",
    ".ppt": "documents_misc",
    ".pptx": "documents_misc",
    ".xls": "documents_misc",
    ".xlsx": "documents_misc",
    ".txt": "documents_misc",
    ".md": "documents_misc",

    # Obvious junk
    ".tmp": "trash_or_temp",
    ".log": "trash_or_temp",
    ".DS_Store".lower(): "trash_or_temp",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def classify_files(df: pd.DataFrame, min_confidence: float = 0.65) -> pd.DataFrame:
    """
    Classify files into categories using a hybrid rule-based and LLM approach.

    Steps:

    1. Validate that df has the required base columns:
       - "file_name", "extension", "full_path", "size_bytes", "modified_time"

    2. Add/initialize classification columns:
       - "category" (str)
       - "classifier_type" (str; "rule" or "llm")
       - "confidence" (float)

    3. Apply rule-based classification using DEFAULT_EXTENSION_MAP:
       - For each matching extension:
           category = DEFAULT_EXTENSION_MAP[extension]
           classifier_type = "rule"
           confidence = 1.0

    4. For rows still missing a category:
       - Build a simple list of dicts with metadata:
         {file_name, extension, full_path, size_bytes, modified_time}
       - Call _llm_classify_files(...) to obtain category + confidence.
       - Fill "category", "classifier_type" = "llm", "confidence".

    5. Apply min_confidence threshold:
       - For rows with classifier_type == "llm" and confidence < min_confidence:
           category = "uncertain_review"

    Returns a NEW DataFrame (copy) with the classification columns added.

    May raise:
    - MissingApiKeyError: if OPENAI_API_KEY is not set (via get_llm_api_key()).
    - LlmUnavailableError: if the LLM API is unreachable or fails.
    - LlmResponseParseError: if the LLM response is malformed or not JSON.
    - ClassificationError: for unexpected internal issues (e.g. bad df schema).
    """
    required_columns = {"file_name", "extension", "full_path", "size_bytes", "modified_time"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ClassificationError(
            f"DataFrame is missing required columns for classification: {sorted(missing)}"
        )

    # Work on a copy to avoid surprising external callers.
    df = df.copy()

    # Normalize extension strings to lowercase.
    df["extension"] = df["extension"].astype(str).str.lower()

    # Initialize classification columns if they don't exist yet.
    if "category" not in df.columns:
        df["category"] = None
    if "classifier_type" not in df.columns:
        df["classifier_type"] = None
    if "confidence" not in df.columns:
        df["confidence"] = None

    # ------------------------------------------------------------------
    # 1. Rule-based classification
    # ------------------------------------------------------------------
    for ext, category in DEFAULT_EXTENSION_MAP.items():
        mask = df["extension"] == ext
        if not mask.any():
            continue
        df.loc[mask, "category"] = category
        df.loc[mask, "classifier_type"] = "rule"
        df.loc[mask, "confidence"] = 1.0

    # ------------------------------------------------------------------
    # 2. LLM classification for ambiguous rows
    # ------------------------------------------------------------------
    ambiguous_mask = df["category"].isna()
    ambiguous_df = df[ambiguous_mask]

    if not ambiguous_df.empty:
        # Build simple metadata objects for the LLM.
        rows_for_llm: List[Dict[str, Any]] = []
        for _, row in ambiguous_df.iterrows():
            rows_for_llm.append(
                {
                    "file_name": str(row["file_name"]),
                    "extension": str(row["extension"]),
                    "full_path": str(row["full_path"]),
                    "size_bytes": int(row["size_bytes"]),
                    "modified_time": float(row["modified_time"]),
                }
            )

        # Ask the LLM to classify these files.
        llm_results = _llm_classify_files(rows_for_llm)

        # Merge results back into df.
        for idx, row in ambiguous_df.iterrows():
            full_path = str(row["full_path"])
            res = llm_results.get(full_path)

            if res is None:
                # If the model failed to return a result for this file,
                # treat it as uncertain.
                df.at[idx, "category"] = "uncertain_review"
                df.at[idx, "classifier_type"] = "llm"
                df.at[idx, "confidence"] = 0.0
                continue

            category = res.category
            confidence = res.confidence

            # If LLM proposes an unknown category, default to uncertain_review.
            if category not in CATEGORIES:
                category = "uncertain_review"

            df.at[idx, "category"] = category
            df.at[idx, "classifier_type"] = "llm"
            df.at[idx, "confidence"] = confidence

    # ------------------------------------------------------------------
    # 3. Apply min_confidence threshold for LLM classifications
    # ------------------------------------------------------------------
    llm_mask = df["classifier_type"] == "llm"
    low_conf_mask = llm_mask & (df["confidence"].astype(float) < float(min_confidence))
    df.loc[low_conf_mask, "category"] = "uncertain_review"

    return df

# ---------------------------------------------------------------------------
# Internal: LLM integration
# ---------------------------------------------------------------------------
def _llm_classify_files(rows: List[Dict[str, Any]]) -> Dict[str, LlmClassificationResult]:
    """
        Call the LLM to classify a list of files.

        Input:
            rows: list of dicts with keys:
                - file_name
                - extension
                - full_path
                - size_bytes
                - modified_time

        Behavior:

        - Uses OPENAI_API_KEY (enforced by get_llm_api_key()).
        - Builds a single prompt instructing the model to return a JSON object:
            {
              "<full_path>": {
                "category": "<one_of_categories>",
                "confidence": <float_between_0_and_1>
              },
              ...
            }
          where category is one of CATEGORIES.

        - Parses the response as JSON.
        - Returns a mapping: full_path -> LlmClassificationResult.

        Raises:
        - MissingApiKeyError: if OPENAI_API_KEY is not set.
        - LlmUnavailableError: if the OpenAI API call fails or times out.
        - LlmResponseParseError: if the response cannot be parsed as expected.
    """
    if not rows:
        return {}

    # Ensure the API key is present; get_llm_api_key will raise MissingApiKeyError if not.
    api_key = get_llm_api_key()
    client = OpenAI(api_key=api_key)

    # Prepare categories list as text.
    categories_str = ", ".join(f'"{c}"' for c in CATEGORIES)

    # Construct the user input: a JSON list of file metadata.
    files_json = json.dumps(rows, indent=2)

    prompt = (
        "You are helping to organize a user's filesystem.\n"
        "For each file below, assign the most appropriate category from this list:\n"
        f"[{categories_str}]\n\n"
        "Return a single JSON object mapping each file's full_path to an object "
        "with fields 'category' and 'confidence' (a number between 0 and 1).\n\n"
        "Example format:\n"
        "{\n"
        '  "/absolute/path/to/file1.pdf": {"category": "documents_misc", "confidence": 0.92},\n'
        '  "/absolute/path/to/photo.jpg": {"category": "photos_images", "confidence": 0.88}\n'
        "}\n\n"
        "Here is the list of files (as JSON):\n"
        f"{files_json}\n"
    )

    try:
        response = client.responses.create(
            model="gpt-5-nano",
            input=prompt,
        )
    except Exception as exc:
        # Wrap any client/network error as LlmUnavailableError so the CLI
        # can show a friendly message and abort safely.
        raise LlmUnavailableError(f"LLM API call failed: {exc}") from exc

    # the textual output is in `output_text`.
    try:
        raw_text = response.output_text
    except Exception as exc:
        raise LlmResponseParseError(
            f"Unexpected LLM response structure: {exc}"
        ) from exc

    # Parse the model's content as JSON.
    try:
        parsed = json.loads(raw_text)
    except Exception as exc:
        raise LlmResponseParseError(
            f"Failed to parse LLM response as JSON: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise LlmResponseParseError(
            f"LLM response JSON must be an object mapping full_path -> result, got {type(parsed)}"
        )

    results: Dict[str, LlmClassificationResult] = {}

    for full_path_str, info in parsed.items():
        if not isinstance(info, dict):
            # Skip malformed entries, but don't crash entire classification.
            continue

        category = info.get("category", "uncertain_review")
        confidence_raw = info.get("confidence", 0.0)

        # Best-effort numeric conversion + clamp.
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0

        if confidence < 0.0:
            confidence = 0.0
        if confidence > 1.0:
            confidence = 1.0

        # If the model proposes a category not in allowed CATEGORIES, we still
        # accept it here and let classify_files() normalize to 'uncertain_review'
        # afterward. This keeps this helper focused on I/O concerns.
        results[full_path_str] = LlmClassificationResult(
            full_path=Path(full_path_str),
            category=str(category),
            confidence=confidence,
        )

    return results
