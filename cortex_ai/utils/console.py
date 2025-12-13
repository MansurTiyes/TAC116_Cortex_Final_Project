from __future__ import annotations

"""
Console utilities for the `organizer` CLI.

This module centralizes **all** user-facing terminal output and uses Rich
for styling and tables. Typer command handlers should call these helpers
instead of printing directly.
"""

from pathlib import Path
from typing import Iterable, Sequence

from rich.console import Console
from rich.table import Table

from ..core.models import PlanSummary, PlanCategorySummary

# Single shared console instance
console = Console()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _path_str(path: Path | str) -> str:
    """Normalize a Path/str into a string for printing."""
    if isinstance(path, Path):
        return str(path)
    return str(path)

def _format_confidence(value: float | None) -> str:
    """Format an average confidence for display in tables."""
    if value is None:
        return "-"
    return f"{value:.2f}"

def _print_plan_summary_common(plan: PlanSummary) -> None:
    """
    Internal helper that prints the common classification/plan summary
    used by both `scan` and `organize`.
    """
    root_str = _path_str(plan.root_path)

    # Header and overview
    console.print(
        f"[bold]Proposed organization plan for: {root_str}[/bold]"
    )
    console.print(
        f"Total files: {plan.total_files} | "
        f"Total size: {plan.total_size_mb:.2f} MB | "
        f"'uncertain_review' files: {plan.num_uncertain}"
    )

    # Text header + line, then Rich table
    console.print("Category Files Size (MB) Avg confidence")
    console.print("-------------------------------------------------------")

    table = Table("Category", "Files", "Size (MB)", "Avg confidence")

    for cat in plan.categories:
        table.add_row(
            cat.category,
            str(cat.file_count),
            f"{cat.total_size_mb:.2f}",
            _format_confidence(cat.avg_confidence),
        )

    console.print(table)

    # Sample filenames for each category
    for cat in plan.categories:
        if not cat.sample_files:
            continue
        console.print()
        console.print(f"[bold]{cat.category}[/bold]")
        console.print("Sample files:")
        for name in cat.sample_files:
            console.print(f" - {name}")

    # Special note for uncertain_review, if present
    has_uncertain = any(c.category == "uncertain_review" and c.file_count > 0 for c in plan.categories)
    if has_uncertain:
        console.print(
            "Note: Files in the 'uncertain_review' category have low classification confidence."
        )

# ---------------------------------------------------------------------------
# Generic helpers (errors, warnings, success)
# ---------------------------------------------------------------------------

def print_error(message: str) -> None:
    """Print a generic error message in bold red."""
    console.print(f"[bold red]{message}[/bold red]")

def print_warning(message: str) -> None:
    """Print a generic warning message in yellow."""
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}")

def print_success(message: str) -> None:
    """Print a generic success/completion message in green."""
    console.print(f"[green]{message}[/green]")

def print_info(message: str) -> None:
    """Print a generic informational message (unstyled)."""
    console.print(message)

# ---------------------------------------------------------------------------
# Command: organizer scan PATH
# ---------------------------------------------------------------------------

def print_scan_start(path: Path | str) -> None:
    """Printed when a `scan` run begins."""
    console.print(f"[bold]Scanning files under '{_path_str(path)}'...[/bold]")


def print_scan_classifying() -> None:
    """Printed during `scan` classification phase."""
    console.print("Classifying files (rule-based and LLM)...")

def print_scan_building_plan() -> None:
    """Printed when `scan` is building the organization plan."""
    console.print("Building organization plan...")


def print_scan_plan_summary(plan: PlanSummary) -> None:
    """
    Print the full plan summary for `scan`, including the read-only footer.
    """
    _print_plan_summary_common(plan)
    console.print(
        "[green]Scan complete. This was a read-only run. No files were moved.[/green]"
    )

# Scan-specific / shared errors


def print_scan_error_path_not_found(path: Path | str) -> None:
    """`scan`: path not found."""
    msg = f"Error: Path not found: {_path_str(path)}"
    print_error(msg)


def print_scan_empty_directory(path: Path | str) -> None:
    """`scan`: empty directory (no files)."""
    console.print(f"No files found under '{_path_str(path)}'.")


def print_scan_error_missing_api_key() -> None:
    """`scan`: missing API key."""
    msg = "Error: OPENAI_API_KEY not set. Please set it in your environment or .env file."
    print_error(msg)


def print_scan_error_llm_auth_failure() -> None:
    """`scan`: invalid API key / auth failure."""
    msg = "Error: LLM API authentication failed. Please check your API key."
    print_error(msg)


def print_scan_error_llm_unavailable() -> None:
    """`scan`: LLM service unavailable / timeout / network error."""
    print_error(
        "Error: LLM classification service is currently unavailable. Please try again later."
    )
    console.print(
        "Classification aborted due to LLM error. No files were moved."
    )


# ---------------------------------------------------------------------------
# Command: organizer organize PATH
# ---------------------------------------------------------------------------


def print_organize_start(path: Path | str) -> None:
    """Printed at the beginning of `organize`."""
    console.print(f"[bold]Analyzing folder '{_path_str(path)}'...[/bold]")


def print_organize_scanning() -> None:
    """`organize`: scanning phase message."""
    console.print("Scanning files...")


def print_organize_classifying() -> None:
    """`organize`: classification phase message."""
    console.print("Classifying files (rule-based and LLM)...")


def print_organize_building_plan() -> None:
    """`organize`: plan-building message."""
    console.print("Building organization plan...")


def print_organize_plan_summary(plan: PlanSummary) -> None:
    """
    Print the plan summary for `organize` (same as `scan` but without
    the read-only footer).
    """
    _print_plan_summary_common(plan)


def print_apply_prompt() -> None:
    """
    Print the confirmation prompt:
    'Apply this plan? (y/n): '

    This function only prints; command code is responsible for reading input.
    """
    # We keep the exact text, adding a subtle styling marker.
    console.print("Apply this plan? (y/n): ", end="")


def print_no_changes_applied() -> None:
    """Printed when user declines to apply the plan."""
    console.print("No changes applied.")
    console.print("No files were moved.")


def print_applying_plan() -> None:
    """Printed just before moves during `organize`."""
    console.print("Applying organization plan...")


def print_move_warning(file_path: Path | str, error_reason: str) -> None:
    """
    Per-file permission/OS error warning during move phase.

    Text: "Warning: failed to move '{file_path}': {error_reason}"
    """
    message = f"failed to move '{_path_str(file_path)}': {error_reason}"
    print_warning(message)


def print_uncertain_review_behavior(sample_files: Sequence[str]) -> None:
    """
    Printed after moves for uncertain_review behavior, with example filenames.
    """
    console.print(
        "Files in the 'uncertain_review' category were moved to 'uncertain_review' directory due to low classification confidence."
    )
    if sample_files:
        console.print("Uncertain files")
        for name in sample_files:
            console.print(f" - {name}")


def print_organize_move_overview(
    moved_count: int, category_count: int, uncertain_count: int
) -> None:
    """
    End-of-run combined overview:

    - "Moved {moved_count} files into {category_count} categories."
    - "{uncertain_count} files left as 'uncertain_review'."
    """
    console.print(
        f"Moved {moved_count} files into {category_count} categories."
    )
    console.print(
        f"{uncertain_count} files left as 'uncertain_review'."
    )


def print_organize_move_summary(
    moved_success: int, failed_count: int
) -> None:
    """
    Base summary for moves:

    - "Moved {moved_success} files successfully."
    - "Failed to move {failed_count} files (see warnings above)."
    """
    console.print(
        f"Moved {moved_success} files successfully."
    )
    console.print(
        f"Failed to move {failed_count} files (see warnings above)."
    )


def print_move_log_written(log_path: Path | str, had_moves: bool) -> None:
    """
    Print log path summary:

    - "Move log written to '{log_path}'."
    - or "Move log written to '{log_path}' (no moves recorded)."
    """
    path_str = _path_str(log_path)
    if had_moves:
        console.print(f"Move log written to '{path_str}'.")
    else:
        console.print(
            f"Move log written to '{path_str}' (no moves recorded)."
        )


# Organize-specific / shared errors


def print_organize_error_path_not_found(path: Path | str) -> None:
    """`organize`: path not found (no moves)."""
    msg = f"Error: Path not found: {_path_str(path)}"
    print_error(msg)
    console.print("No files were moved.")


def print_organize_empty_directory(path: Path | str) -> None:
    """`organize`: empty directory (no moves)."""
    console.print(f"No files found under '{_path_str(path)}'.")
    console.print("No files were moved.")


def print_organize_error_missing_api_key() -> None:
    """`organize`: missing API key (no moves)."""
    msg = "Error: OPENAI_API_KEY not set. Please set it in your environment or .env file."
    print_error(msg)
    console.print("No files were moved.")


def print_organize_error_llm_auth_failure() -> None:
    """`organize`: invalid API key / auth failure (classification aborted, no moves)."""
    msg = "Error: LLM API authentication failed. Please check your API key."
    print_error(msg)
    console.print("Classification aborted. No files were moved.")


def print_organize_error_llm_unavailable() -> None:
    """`organize`: LLM service unavailable (classification aborted, no moves)."""
    print_error(
        "Error: LLM classification service is currently unavailable. Please try again later."
    )
    console.print("Classification aborted. No files were moved.")


def print_no_files_were_moved() -> None:
    """
    Cross-cutting helper for printing:

    - "No files were moved."
    """
    console.print("No files were moved.")


# ---------------------------------------------------------------------------
# Command: organizer version
# ---------------------------------------------------------------------------


def print_version(version: str, python_version: str, api_key_set: bool) -> None:
    """
    Print version information:

    - "organizer {version}"
    - "Python {python_version}"
    - "LLM_API_KEY: set" / "LLM_API_KEY: not set"
    """
    console.print(f"[bold]organizer {version}[/bold]")
    console.print(f"Python {python_version}")
    if api_key_set:
        console.print("LLM_API_KEY: set")
    else:
        console.print("LLM_API_KEY: not set")