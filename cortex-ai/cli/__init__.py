from ..utils.paths import resolve_root
from ..utils import console
from ..core import scanner, errors

import typer

app = typer.Typer(no_args_is_help=True, help="Cortex AI – AI-assisted folder organizer.")

@app.command()
def version():
    """Display version information."""
    print("organizer 0.1.0")

@app.command()
def scan(path: str):
    """Scan a folder and show how many files it contains."""
    try:
        root = resolve_root(path)
        df = scanner.scan_directory(root)
        if df.empty:
            console.print_error(f"No files found under {root}")
            raise typer.Exit(code=0)

        console.print_success(f"Scanned {len(df)} files under {root}")
    except errors.OrganizerError as e:
        console.print_error(str(e))
        raise typer.Exit(code=1)