"""
Workflow CLI commands for PathwayLens.
"""

from pathlib import Path
from typing import Optional

import asyncio
import typer
from rich.console import Console
from rich.table import Table

from pathwaylens_core.workflow import WorkflowManager, WorkflowValidationError
from pathwaylens_core.workflow.templates import get_template, list_templates


app = typer.Typer(
    name="workflow",
    help="Run and validate YAML/JSON workflows",
    rich_markup_mode="rich",
)

console = Console()


@app.command()
def validate(
    file: Path = typer.Argument(..., help="Path to workflow YAML/JSON"),
):
    """Validate a workflow specification without executing it."""
    try:
        manager = WorkflowManager()
        spec = manager.load(file)
        steps = manager.validate(spec)
        table = Table(title="Workflow Steps")
        table.add_column("#")
        table.add_column("step_id")
        table.add_column("type")
        for idx, s in enumerate(steps):
            table.add_row(str(idx + 1), s.step_id, s.type)
        console.print(table)
        console.print("[green]Validation passed[/green]")
    except WorkflowValidationError as e:
        console.print(f"[red]Validation error:[/red] {e}")
        raise typer.Exit(code=2)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def run(
    file: Path = typer.Argument(..., help="Path to workflow YAML/JSON"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for results"),
    checkpoint: bool = typer.Option(True, "--checkpoint/--no-checkpoint", help="Enable checkpointing"),
):
    """Execute a workflow specification."""
    async def _run():
        manager = WorkflowManager()
        spec = manager.load(file)
        steps = manager.validate(spec)
        
        if checkpoint:
            from pathwaylens_core.workflow.checkpoint import CheckpointManager
            checkpoint_mgr = CheckpointManager()
            # Check for existing checkpoint
            checkpoint_data = checkpoint_mgr.load_checkpoint(file)
            if checkpoint_data:
                console.print(f"[yellow]Found checkpoint, resuming from step {checkpoint_data.get('last_step')}[/yellow]")
        
        result = await manager.run(steps)
        
        if checkpoint:
            checkpoint_mgr.save_checkpoint(file, result)
        
        return result

    try:
        result = asyncio.run(_run())
        console.print("[green]Workflow completed[/green]")
        
        if output:
            import json
            with open(output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            console.print(f"[green]Results saved to {output}[/green]")
        else:
            console.print_json(data=result)
    except WorkflowValidationError as e:
        console.print(f"[red]Validation error:[/red] {e}")
        raise typer.Exit(code=2)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def template(
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Template name"),
    output: Path = typer.Option(Path("workflow_template.yaml"), "--output", "-o", help="Output file"),
    list_templates_flag: bool = typer.Option(False, "--list", "-l", help="List available templates"),
):
    """Generate a workflow template."""
    try:
        if list_templates_flag:
            templates = list_templates()
            console.print("[bold]Available workflow templates:[/bold]")
            for template_name in templates:
                console.print(f"  - {template_name}")
            return
        
        template_name = name or "default"
        template_content = get_template(template_name)
        
        with open(output, 'w') as f:
            f.write(template_content)
        
        console.print(f"[green]Template '{template_name}' saved to {output}[/green]")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error generating template:[/red] {e}")
        raise typer.Exit(code=1)










