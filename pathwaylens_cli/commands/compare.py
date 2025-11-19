"""
Comparison commands for PathwayLens CLI.
"""

import typer
from typing import Optional, List
from pathlib import Path
from rich.console import Console

app = typer.Typer(
    name="compare",
    help="Compare multiple datasets",
    rich_markup_mode="rich"
)

console = Console()

@app.command()
def overlap(
    inputs: List[str] = typer.Option(..., "--inputs", "-i", help="Input files to compare"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Compare datasets for overlap."""
    for input_file in inputs:
        input_path = Path(input_file)
        if not input_path.exists():
            console.print(f"[red]Error: Input file '{input_file}' does not exist[/red]")
            raise typer.Exit(code=1)
        if not input_path.is_file():
            console.print(f"[red]Error: '{input_file}' is not a file[/red]")
            raise typer.Exit(code=1)
    
    console.print(f"Comparing overlap between {len(inputs)} datasets")

if __name__ == "__main__":
    app()