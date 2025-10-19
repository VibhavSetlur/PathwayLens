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
from typer import Context
from rich.console import Console
from rich.panel import Panel

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import commands with graceful error handling
try:
    from pathwaylens_cli.commands import (
        normalize,
        analyze, 
        compare,
        visualize,
        config,
        info
    )
    _commands_available = True
except ImportError as e:
    _commands_available = False
    _import_error = e

# Initialize the main CLI app
app = typer.Typer(
    name="pathwaylens",
    help="🧬 PathwayLens: Next-generation computational biology platform",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Add subcommands with graceful error handling
if _commands_available:
    app.add_typer(normalize.app, name="normalize", help="Convert gene identifiers across formats")
    app.add_typer(analyze.app, name="analyze", help="Perform pathway analysis")
    app.add_typer(compare.app, name="compare", help="Compare multiple datasets")
    app.add_typer(visualize.app, name="visualize", help="Generate visualizations")
    app.add_typer(config.app, name="config", help="Manage configuration")
    app.add_typer(info.app, name="info", help="Display system information")
else:
    # Add a help command that explains missing dependencies
    @app.command()
    def install_extras():
        """Install additional features for PathwayLens."""
        console = Console()
        console.print(Panel.fit(
            "[bold yellow]⚠️  Additional features are not installed.[/bold yellow]\n\n"
            "[bold cyan]📦 To install all features:[/bold cyan]\n"
            "pip install pathwaylens[all]\n\n"
            "[bold cyan]🔧 To install specific features:[/bold cyan]\n"
            "pip install pathwaylens[analysis]  # Core analysis tools\n"
            "pip install pathwaylens[viz]       # Visualization tools\n"
            "pip install pathwaylens[pathways]  # Pathway databases\n"
            "pip install pathwaylens[api]       # Web API\n"
            "pip install pathwaylens[jobs]      # Background jobs\n"
            "pip install pathwaylens[database]  # Database support\n\n"
            "[bold cyan]🛠️  For development:[/bold cyan]\n"
            "pip install pathwaylens[dev]",
            title="🔧 Installation Required",
            border_style="yellow"
        ))

# Global options
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
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
    PathwayLens: Next-generation computational biology platform
    
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
            "Built for the bioinformatics community",
            title="🧬 PathwayLens"
        ))
        raise typer.Exit()
    
    # Show installation help if commands are not available
    if not _commands_available:
        console = Console()
        console.print(Panel.fit(
            "[bold yellow]ℹ️  Core features installed, but additional features are missing.[/bold yellow]\n\n"
            "[bold cyan]Available commands:[/bold cyan]\n"
            "pathwaylens install-extras  # Show installation options\n\n"
            "[bold cyan]📦 To install all features:[/bold cyan]\n"
            "pip install pathwaylens[all]\n\n"
            "[bold cyan]🔧 To install specific features:[/bold cyan]\n"
            "pip install pathwaylens[analysis]  # Core analysis tools\n"
            "pip install pathwaylens[viz]       # Visualization tools\n"
            "pip install pathwaylens[pathways]  # Pathway databases",
            title="🔧 Installation Options",
            border_style="blue"
        ))

def main():
    """Main entry point for the CLI."""
    app()

if __name__ == "__main__":
    main()