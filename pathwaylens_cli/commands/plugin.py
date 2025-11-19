"""
Plugin management CLI commands for PathwayLens.
"""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from pathwaylens_core.plugins.plugin_system import PluginSystem


app = typer.Typer(
    name="plugin",
    help="List, enable/disable, and execute plugins",
    rich_markup_mode="rich",
)

console = Console()


async def _ensure_system() -> PluginSystem:
    ps = PluginSystem()
    await ps.initialize()
    return ps


@app.command()
def list():
    """List available plugins."""
    async def _list():
        ps = await _ensure_system()
        return ps.plugin_manager.list_plugins()

    plugins = asyncio.run(_list())
    table = Table(title="Plugins")
    table.add_column("name")
    for name in plugins:
        table.add_row(name)
    console.print(table)


@app.command()
def exec(
    name: str = typer.Argument(..., help="Plugin name"),
    input: Optional[str] = typer.Option(None, "--input", "-i", help="Input data (path or JSON)"),
):
    """Execute a plugin by name with optional input."""
    async def _exec():
        ps = await _ensure_system()
        return await ps.execute_plugin(name, input)

    try:
        result = asyncio.run(_exec())
        console.print("[green]Plugin executed[/green]")
        console.print_json(data={"plugin": name, "result": result})
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)












