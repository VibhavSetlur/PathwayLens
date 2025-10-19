"""
Analysis commands for PathwayLens CLI.
"""

import typer
from typing import Optional, List
from rich.console import Console

app = typer.Typer(
    name="analyze",
    help="Perform pathway analysis",
    rich_markup_mode="rich"
)

console = Console()

@app.command()
def ora(
    input: str = typer.Option(..., "--input", "-i", help="Input gene list file"),
    databases: List[str] = typer.Option(["kegg"], "--databases", "-d", help="Databases to use"),
    species: str = typer.Option("human", "--species", "-s", help="Species"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Perform Over-Representation Analysis (ORA)."""
    console.print(f"Running ORA analysis on {input}")

@app.command()
def gsea(
    input: str = typer.Option(..., "--input", "-i", help="Input ranked gene list file"),
    databases: List[str] = typer.Option(["kegg"], "--databases", "-d", help="Databases to use"),
    species: str = typer.Option("human", "--species", "-s", help="Species"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Perform Gene Set Enrichment Analysis (GSEA)."""
    console.print(f"Running GSEA analysis on {input}")

if __name__ == "__main__":
    app()