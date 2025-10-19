"""
Info commands for PathwayLens CLI.
"""

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="info",
    help="Display system information",
    rich_markup_mode="rich"
)

console = Console()

@app.command()
def version():
    """Show version information."""
    console.print(Panel.fit(
        "[bold blue]PathwayLens v2.0.0[/bold blue]\n"
        "Next-generation computational biology platform",
        title="🧬 PathwayLens"
    ))

@app.command()
def status():
    """Show system status."""
    console.print("System status: [green]OK[/green]")

if __name__ == "__main__":
    app()