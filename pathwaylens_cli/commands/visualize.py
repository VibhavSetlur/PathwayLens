"""
Visualization commands for PathwayLens CLI.
"""

import typer
from typing import Optional
from pathlib import Path
from rich.console import Console

app = typer.Typer(
    name="visualize",
    help="Generate visualizations",
    rich_markup_mode="rich"
)

console = Console()

@app.command()
def dot_plot(
    input: str = typer.Option(..., "--input", "-i", help="Input analysis results file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Create a dot plot."""
    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file '{input}' does not exist[/red]")
        raise typer.Exit(code=1)
    
    if not input_path.is_file():
        console.print(f"[red]Error: '{input}' is not a file[/red]")
        raise typer.Exit(code=1)
    
    console.print(f"Creating dot plot from {input}")

if __name__ == "__main__":
    app()