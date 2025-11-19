"""
Analysis commands for PathwayLens CLI.
"""

import asyncio
import typer
from typing import Optional, List
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from pathwaylens_core.analysis import AnalysisEngine
from pathwaylens_core.analysis.schemas import AnalysisType, DatabaseType, AnalysisParameters, CorrectionMethod
from pathwaylens_cli.utils.batch_processor import BatchProcessor

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
    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file '{input}' does not exist[/red]")
        raise typer.Exit(code=1)
    
    if not input_path.is_file():
        console.print(f"[red]Error: '{input}' is not a file[/red]")
        raise typer.Exit(code=1)
    
    # Create output directory if output file is specified
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold]Running ORA analysis on {input}[/bold]")

@app.command()
def gsea(
    input: str = typer.Option(..., "--input", "-i", help="Input ranked gene list file"),
    databases: List[str] = typer.Option(["kegg"], "--databases", "-d", help="Databases to use"),
    species: str = typer.Option("human", "--species", "-s", help="Species"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Perform Gene Set Enrichment Analysis (GSEA)."""
    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file '{input}' does not exist[/red]")
        raise typer.Exit(code=1)
    
    if not input_path.is_file():
        console.print(f"[red]Error: '{input}' is not a file[/red]")
        raise typer.Exit(code=1)
    
    console.print(f"[bold]Running GSEA analysis on {input}[/bold]")

@app.command()
def batch(
    inputs: List[str] = typer.Option(..., "--inputs", "-i", help="Input files to process"),
    method: str = typer.Option("ora", "--method", "-m", help="Analysis method (ora, gsea, etc.)"),
    databases: List[str] = typer.Option(["kegg"], "--databases", "-d", help="Databases to use"),
    species: str = typer.Option("human", "--species", "-s", help="Species"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Process multiple files in batch."""
    console.print(f"[bold]Processing {len(inputs)} files in batch[/bold]")
    
    # Implementation would use BatchProcessor
    # This is a placeholder

if __name__ == "__main__":
    app()