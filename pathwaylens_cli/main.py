#!/usr/bin/env python3
"""
PathwayLens CLI - Standalone command-line interface.

This module provides the main entry point for the PathwayLens CLI,
enabling direct invocation as 'pathwaylens' command.
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pathwaylens_cli.commands import (
    normalize,
    analyze, 
    compare,
    visualize,
    config,
    info
)

# Initialize the main CLI app
app = typer.Typer(
    name="pathwaylens",
    help="🧬 PathwayLens: Next-generation computational biology platform",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Add subcommands
app.add_typer(normalize.app, name="normalize", help="Convert gene identifiers across formats")
app.add_typer(analyze.app, name="analyze", help="Perform pathway analysis")
app.add_typer(compare.app, name="compare", help="Compare multiple datasets")
app.add_typer(visualize.app, name="visualize", help="Generate visualizations")
app.add_typer(config.app, name="config", help="Manage configuration")
app.add_typer(info.app, name="info", help="Display system information")

# Global options
@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, 
        "--version", 
        "-v", 
        help="Show version and exit"
    ),
    verbose: bool = typer.Option(
        False, 
        "--verbose", 
        help="Enable verbose output"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    )
):
    """
    🧬 PathwayLens: Next-generation computational biology platform
    
    A comprehensive tool for pathway analysis across bulk RNA-seq, scRNA-seq/snRNA-seq,
    ATAC-seq, proteomics, and arbitrary gene lists with support for multiple pathway
    databases and robust ID conversion.
    
    Examples:
        pathwaylens normalize genes.csv --species human --target-type symbol
        pathwaylens analyze deseq2_results.csv --databases kegg,reactome
        pathwaylens compare dataset1.csv dataset2.csv --species human
    """
    if version:
        console = Console()
        console.print(Panel.fit(
            "[bold blue]PathwayLens v2.0.0[/bold blue]\n"
            "Next-generation computational biology platform\n"
            "Built with ❤️ for the bioinformatics community",
            title="🧬 PathwayLens"
        ))
        raise typer.Exit()

if __name__ == "__main__":
    app()