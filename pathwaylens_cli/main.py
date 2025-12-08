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
    info,
    workflow,
    plugin,
    database, # Added database import
)

# Initialize the main CLI app
app = typer.Typer(
    name="pathwaylens",
    help="PathwayLens: Research-grade pathway analysis tool", # Modified help message
    rich_markup_mode="rich", # Removed add_completion and no_args_is_help
)

# Add subcommands
app.add_typer(normalize.app, name="normalize", help="Convert gene identifiers across formats")
app.add_typer(analyze.app, name="analyze", help="Perform pathway analysis")
app.command(name="compare", help="Compare multiple datasets")(compare.compare)
app.add_typer(database.app, name="database", help="Manage and query pathway databases") # Added database subcommand
app.add_typer(visualize.app, name="visualize", help="Generate visualizations")
app.add_typer(config.app, name="config", help="Manage configuration")
app.add_typer(info.app, name="info", help="Display system information")
app.add_typer(workflow.app, name="workflow", help="Run and validate workflows")
app.add_typer(plugin.app, name="plugin", help="Manage and execute plugins")

# Add command aliases for convenience
app.add_typer(normalize.app, name="norm", help="Alias for normalize")
app.add_typer(analyze.app, name="ana", help="Alias for analyze")
app.add_typer(visualize.app, name="viz", help="Alias for visualize")
app.add_typer(workflow.app, name="wf", help="Alias for workflow")

# Global options
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, 
        "--version", 
        help="Show version and exit"
    ),
    verbose: bool = typer.Option(
        False, 
        "--verbose", 
        "-v",
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
    # Store config file in context for commands to use
    if config_file:
        ctx.meta["config_file"] = str(config_file)
    
    if version:
        console = Console()
        console.print("PathwayLens v1.0.0")
        raise typer.Exit(code=0)

def cli_entry():
    """Entry point for console script."""
    from pathwaylens_cli.utils.exceptions import CLIException
    
    try:
        app()
    except CLIException as e:
        console = Console()
        console.print(f"[bold red]Error:[/bold red] {e.message}")
        if e.context:
            console.print(Panel(str(e.context), title="Context", border_style="red"))
        sys.exit(1)
    except Exception as e:
        console = Console()
        console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        console.print_exception()
        sys.exit(1)

if __name__ == "__main__":
    cli_entry()