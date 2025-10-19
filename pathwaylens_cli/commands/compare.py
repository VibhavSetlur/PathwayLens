"""
Comparison commands for PathwayLens CLI.
"""

import typer
from typing import Optional, List
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
    console.print(f"Comparing overlap between {len(inputs)} datasets")

if __name__ == "__main__":
    app()