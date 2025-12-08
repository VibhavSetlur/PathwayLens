"""
Database management commands for PathwayLens CLI.
"""

import asyncio
import typer
from typing import Optional
from rich.console import Console
from rich.table import Table

from pathwaylens_core.data import DatabaseManager

app = typer.Typer(
    name="database",
    help="Manage pathway databases",
    rich_markup_mode="rich"
)

console = Console()

@app.command()
def update(
    force: bool = typer.Option(False, "--force", "-f", help="Force update even if up to date"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Check for and download database updates."""
    console.print("[bold]Checking for database updates...[/bold]")
    
    db_manager = DatabaseManager()
    
    try:
        # Run update
        results = asyncio.run(db_manager.update_databases(force=force))
        
        # Display results
        table = Table(title="Database Update Status")
        table.add_column("Database", style="cyan")
        table.add_column("Status", style="green")
        
        updated_count = 0
        for name, updated in results.items():
            status = "[green]Updated[/green]" if updated else "Up to date"
            table.add_row(name, status)
            if updated:
                updated_count += 1
                
        console.print(table)
        
        if updated_count > 0:
            console.print(f"[green]Successfully updated {updated_count} databases.[/green]")
        else:
            console.print("All databases are up to date.")
            
    except Exception as e:
        console.print(f"[red]Update failed: {e}[/red]")
        raise typer.Exit(code=1)

@app.command()
def status():
    """Show status of installed databases."""
    console.print("[bold]Database Status[/bold]")
    
    db_manager = DatabaseManager()
    
    try:
        # Get info
        info = asyncio.run(db_manager.get_all_database_info())
        
        table = Table(title="Installed Databases")
        table.add_column("Database", style="cyan")
        table.add_column("Version", style="magenta")
        table.add_column("Pathways", style="blue")
        table.add_column("Genes", style="yellow")
        
        for name, data in info.items():
            table.add_row(
                name,
                data.get("version", "Unknown"),
                str(data.get("pathway_count", "N/A")),
                str(data.get("gene_count", "N/A"))
            )
            
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Failed to get status: {e}[/red]")
        raise typer.Exit(code=1)
