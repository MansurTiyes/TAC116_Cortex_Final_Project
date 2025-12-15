from __future__ import annotations

from pathlib import Path

from ..core import scanner, classifier, planner, executor
from ..core.errors import (
    PathNotFoundError,
    PathNotDirectoryError,
    NoFilesFoundError,
    MissingApiKeyError,
    LlmUnavailableError,
    LlmResponseParseError,
    ClassificationError,
    FileMoveError,
    LoggingError
)
from ..utils.paths import resolve_root
from ..utils.console import (
    # scan
    print_scan_start,
    print_scan_classifying,
    print_scan_building_plan,
    print_scan_plan_summary,
    print_scan_error_path_not_found,
    print_scan_empty_directory,
    print_scan_error_missing_api_key,
    print_scan_error_llm_unavailable,
    # organize
    print_organize_start,
    print_organize_scanning,
    print_organize_classifying,
    print_organize_building_plan,
    print_organize_plan_summary,
    print_apply_prompt,
    print_no_changes_applied,
    print_applying_plan,
    print_move_warning,
    print_uncertain_review_behavior,
    print_organize_move_overview,
    print_organize_move_summary,
    print_move_log_written,
    print_organize_error_path_not_found,
    print_organize_empty_directory,
    print_organize_error_missing_api_key,
    print_organize_error_llm_unavailable,
    print_no_files_were_moved,
    # generic
    print_error,
)

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(no_args_is_help=True, help="Cortex AI – AI-assisted folder organizer.")

@app.command()
def version():
    """Display version information."""
    print("organizer 0.1.0")

@app.command()
def scan(
    path: str = typer.Argument(
        ...,
        metavar="PATH",
        help="Root folder to analyze (e.g. ~/Downloads)",
    ),
    min_confidence: float = typer.Option(
        0.65,
        "--min-confidence",
        help="Minimum LLM confidence required to auto-assign a category.",
    ),
) -> None:
    """
    Read-only scan + classification of all files under PATH.

    Pipeline:
      - resolve PATH
      - scan folder -> DataFrame
      - classify (rules + LLM)
      - build a plan
      - print summary (categories, counts, sizes, samples, confidence)
    """
    # 1) Resolve root path (expand ~, check exist/dir)
    try:
        root: Path = resolve_root(path)
    except PathNotFoundError:
        print_scan_error_path_not_found(path)
        raise typer.Exit(code=1)
    except PathNotDirectoryError:
        print_error(f"Error: Provided path is not a directory: {path}")
        raise typer.Exit(code=1)

    # 2) Scan filesystem into DataFrame
    print_scan_start(root)

    try:
        df = scanner.scan_directory(root)
    except NoFilesFoundError:
        # Soft “empty directory” case
        print_scan_empty_directory(root)
        raise typer.Exit(code=0)

    # 3) Classify files (rules + LLM)
    print_scan_classifying()

    try:
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
        ) as progress:
            progress.add_task(description="Classifying with LLM...", total=None)
            classified_df = classifier.classify_files(df, min_confidence=min_confidence)
    except MissingApiKeyError:
        print_scan_error_missing_api_key()
        raise typer.Exit(code=1)
    except LlmUnavailableError:
        print_scan_error_llm_unavailable()
        raise typer.Exit(code=1)
    except LlmResponseParseError as exc:
        print_error(f"Error: Failed to parse LLM response: {exc}")
        raise typer.Exit(code=1)
    except ClassificationError as exc:
        print_error(f"Error: Classification failed: {exc}")
        raise typer.Exit(code=1)

    # 4) Build plan summary
    print_scan_building_plan()

    try:
        plan = planner.build_plan(classified_df, root_path=root)
    except ClassificationError as exc:
        print_error(f"Error: Failed to build organization plan: {exc}")
        raise typer.Exit(code=1)

    # 5) Print final read-only summary
    print_scan_plan_summary(plan)
    # Typer will exit with code 0 on success

@app.command()
def organize(
    path: str = typer.Argument(
        ...,
        metavar="PATH",
        help="Root folder to organize (e.g. ~/Downloads)",
    ),
    min_confidence: float = typer.Option(
        0.65,
        "--min-confidence",
        help="Minimum LLM confidence required to auto-assign a category.",
    ),
) -> None:
    """
    Organize files under PATH.

    Pipeline:
      - resolve PATH
      - scan folder -> DataFrame
      - classify (rules + LLM)
      - build a plan + print summary
      - prompt user
      - apply moves + write CSV move log
    """
    # 1) Resolve root path
    try:
        root: Path = resolve_root(path)
    except PathNotFoundError:
        print_organize_error_path_not_found(path)
        raise typer.Exit(code=1)
    except PathNotDirectoryError:
        print_error(f"Error: Provided path is not a directory: {path}")
        print_no_files_were_moved()
        raise typer.Exit(code=1)

    # 2) Scan filesystem into DataFrame
    print_organize_start(root)
    print_organize_scanning()

    try:
        df = scanner.scan_directory(root)
    except NoFilesFoundError:
        print_organize_empty_directory(root)
        raise typer.Exit(code=0)

    # 3) Classify files (rules + LLM)
    print_organize_classifying()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Classifying with LLM...", total=None)
            classified_df = classifier.classify_files(df, min_confidence=min_confidence)
    except MissingApiKeyError:
        print_organize_error_missing_api_key()
        raise typer.Exit(code=1)
    except LlmUnavailableError:
        print_organize_error_llm_unavailable()
        raise typer.Exit(code=1)
    except LlmResponseParseError as exc:
        print_error(f"Error: Failed to parse LLM response: {exc}")
        print_no_files_were_moved()
        raise typer.Exit(code=1)
    except ClassificationError as exc:
        print_error(f"Error: Classification failed: {exc}")
        print_no_files_were_moved()
        raise typer.Exit(code=1)

    # 4) Build plan summary
    print_organize_building_plan()
    try:
        plan = planner.build_plan(classified_df, root_path=root)
    except ClassificationError as exc:
        print_error(f"Error: Failed to build organization plan: {exc}")
        print_no_files_were_moved()
        raise typer.Exit(code=1)

    # 5) Print plan summary + confirm
    print_organize_plan_summary(plan)

    print_apply_prompt()
    choice = input().strip().lower()
    if choice not in ("y", "yes"):
        print_no_changes_applied()
        raise typer.Exit(code=0)

    # 6) Apply plan (moves + CSV log)
    print_applying_plan()

    try:
        moves_df, log_path = executor.apply_plan(plan, root_path=root)
    except FileMoveError as exc:
        print_error(f"Error: Failed to apply plan: {exc}")
        print_no_files_were_moved()
        raise typer.Exit(code=1)
    except LoggingError as exc:
        # Note: files may have been moved already, but logging failed.
        print_error(f"Error: Failed to write move log: {exc}")
        raise typer.Exit(code=1)

    # 7) Print per-file warnings for failures
    if not moves_df.empty and "status" in moves_df.columns:
        for _, r in moves_df.iterrows():
            status = str(r.get("status", ""))
            if status.startswith("failed:"):
                reason = status[len("failed:") :].strip() or status
                print_move_warning(str(r.get("old_path", "")), reason)

    # 8) End-of-run summaries
    if moves_df.empty:
        moved_success = 0
        failed_count = 0
        category_count = 0
    else:
        moved_success = int((moves_df["status"] == "success").sum())
        failed_count = int(moves_df["status"].astype(str).str.startswith("failed").sum())
        category_count = int(
            moves_df.loc[moves_df["status"] == "success", "category"].nunique()
        )

    print_organize_move_overview(
        moved_count=moved_success,
        category_count=category_count,
        uncertain_count=int(plan.num_uncertain),
    )
    print_organize_move_summary(
        moved_success=moved_success,
        failed_count=failed_count,
    )

    # Special note for uncertain_review behavior (if any)
    if not moves_df.empty and "category" in moves_df.columns:
        uncertain_samples = (
            moves_df.loc[
                (moves_df["category"] == "uncertain_review")
                & (moves_df["status"] == "success"),
                "file_name",
            ]
            .astype(str)
            .head(10)
            .tolist()
        )
        if uncertain_samples:
            print_uncertain_review_behavior(uncertain_samples)

    # 9) Log path
    print_move_log_written(log_path, had_moves=(len(moves_df) > 0))
