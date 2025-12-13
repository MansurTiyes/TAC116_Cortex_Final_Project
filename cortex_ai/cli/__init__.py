from __future__ import annotations

from pathlib import Path

from ..core import scanner, classifier, planner
from ..core.errors import (
    PathNotFoundError,
    PathNotDirectoryError,
    NoFilesFoundError,
    MissingApiKeyError,
    LlmUnavailableError,
    LlmResponseParseError,
    ClassificationError,
)
from ..utils.paths import resolve_root
from ..utils.console import (
    print_scan_start,
    print_scan_classifying,
    print_scan_building_plan,
    print_scan_plan_summary,
    print_scan_error_path_not_found,
    print_scan_empty_directory,
    print_scan_error_missing_api_key,
    print_scan_error_llm_unavailable,
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