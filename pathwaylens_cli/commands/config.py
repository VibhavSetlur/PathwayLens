"""
Config command for PathwayLens CLI.
"""

import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

app = typer.Typer(
    name="config",
    help="Manage PathwayLens configuration",
    rich_markup_mode="rich",
)

console = Console()


@app.command()
def show(
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    )
):
    """
    Show current configuration.
    
    This command displays the current PathwayLens configuration,
    including database settings, analysis parameters, and other options.
    
    Examples:
        pathwaylens config show
        pathwaylens config show --config custom_config.yml
    """
    
    # Load configuration
    config = _load_config(config_file)
    
    # Display configuration
    console.print(Panel.fit(
        "[bold blue]PathwayLens Configuration[/bold blue]",
        title="🧬 Config"
    ))
    
    # Database configuration
    if 'databases' in config:
        console.print("\n[bold blue]Database Configuration:[/bold blue]")
        db_table = Table()
        db_table.add_column("Database", style="cyan")
        db_table.add_column("Enabled", style="green")
        db_table.add_column("Rate Limit", style="yellow")
        db_table.add_column("Base URL", style="blue")
        
        for db_name, db_config in config['databases'].items():
            db_table.add_row(
                db_name,
                "✅" if db_config.get('enabled', True) else "❌",
                str(db_config.get('rate_limit', 'N/A')),
                db_config.get('base_url', 'N/A')
            )
        
        console.print(db_table)
    
    # Analysis configuration
    if 'analysis' in config:
        console.print("\n[bold blue]Analysis Configuration:[/bold blue]")
        analysis_table = Table()
        analysis_table.add_column("Parameter", style="cyan")
        analysis_table.add_column("Value", style="green")
        
        for param, value in config['analysis'].items():
            analysis_table.add_row(param, str(value))
        
        console.print(analysis_table)
    
    # Cache configuration
    if 'cache' in config:
        console.print("\n[bold blue]Cache Configuration:[/bold blue]")
        cache_table = Table()
        cache_table.add_column("Parameter", style="cyan")
        cache_table.add_column("Value", style="green")
        
        for param, value in config['cache'].items():
            cache_table.add_row(param, str(value))
        
        console.print(cache_table)
    
    # Output configuration
    if 'output' in config:
        console.print("\n[bold blue]Output Configuration:[/bold blue]")
        output_table = Table()
        output_table.add_column("Parameter", style="cyan")
        output_table.add_column("Value", style="green")
        
        for param, value in config['output'].items():
            output_table.add_row(param, str(value))
        
        console.print(output_table)


@app.command()
def set(
    key: str = typer.Argument(
        ...,
        help="Configuration key to set (e.g., 'databases.kegg.enabled')"
    ),
    value: str = typer.Argument(
        ...,
        help="Value to set"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    )
):
    """
    Set a configuration value.
    
    This command sets a specific configuration value in the PathwayLens
    configuration file.
    
    Examples:
        pathwaylens config set databases.kegg.enabled true
        pathwaylens config set analysis.significance_threshold 0.01
        pathwaylens config set cache.enabled false
    """
    
    # Load configuration
    config = _load_config(config_file)
    
    # Parse key and set value
    keys = key.split('.')
    current = config
    
    # Navigate to the parent of the target key
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    
    # Set the value
    target_key = keys[-1]
    
    # Try to convert value to appropriate type
    converted_value = _convert_value(value)
    current[target_key] = converted_value
    
    # Save configuration
    _save_config(config, config_file)
    
    console.print(f"[green]✅ Set {key} = {converted_value}[/green]")


@app.command()
def get(
    key: str = typer.Argument(
        ...,
        help="Configuration key to get (e.g., 'databases.kegg.enabled')"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    )
):
    """
    Get a configuration value.
    
    This command retrieves a specific configuration value from the
    PathwayLens configuration file.
    
    Examples:
        pathwaylens config get databases.kegg.enabled
        pathwaylens config get analysis.significance_threshold
        pathwaylens config get cache.enabled
    """
    
    # Load configuration
    config = _load_config(config_file)
    
    # Parse key and get value
    keys = key.split('.')
    current = config
    
    try:
        for k in keys:
            current = current[k]
        
        console.print(f"[green]{key} = {current}[/green]")
        
    except KeyError:
        console.print(f"[red]Configuration key '{key}' not found[/red]")
        raise typer.Exit(1)


@app.command()
def init(
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing configuration file"
    )
):
    """
    Initialize configuration file.
    
    This command creates a new PathwayLens configuration file with
    default settings.
    
    Examples:
        pathwaylens config init
        pathwaylens config init --config custom_config.yml
        pathwaylens config init --force
    """
    
    # Set default config file path
    if config_file is None:
        config_file = Path.home() / ".pathwaylens" / "config.yml"
    
    # Check if file exists
    if config_file.exists() and not force:
        console.print(f"[red]Configuration file already exists: {config_file}[/red]")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)
    
    # Create default configuration
    default_config = {
        "version": "2.0.0",
        "debug": False,
        "verbose": False,
        
        "databases": {
            "ensembl": {
                "name": "ensembl",
                "enabled": True,
                "rate_limit": 15,
                "base_url": "https://rest.ensembl.org"
            },
            "ncbi": {
                "name": "ncbi",
                "enabled": True,
                "rate_limit": 3,
                "base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
            },
            "mygene": {
                "name": "mygene",
                "enabled": True,
                "rate_limit": 10,
                "base_url": "https://mygene.info/v3"
            },
            "kegg": {
                "name": "kegg",
                "enabled": True,
                "rate_limit": 10,
                "base_url": "https://rest.kegg.jp"
            },
            "reactome": {
                "name": "reactome",
                "enabled": True,
                "rate_limit": 10,
                "base_url": "https://reactome.org/AnalysisService"
            },
            "go": {
                "name": "go",
                "enabled": True,
                "rate_limit": 5,
                "base_url": "http://current.geneontology.org"
            }
        },
        
        "analysis": {
            "ambiguity_policy": "expand",
            "species_required": True,
            "cross_species_allowed": False,
            "multiple_testing_correction": "fdr_bh",
            "min_pathway_size": 5,
            "max_pathway_size": 500,
            "consensus_method": "stouffer",
            "background_source": "database",
            "gsea_permutations": 1000,
            "gsea_min_size": 15,
            "gsea_max_size": 500
        },
        
        "cache": {
            "enabled": True,
            "base_dir": ".pathwaylens/cache",
            "max_size_mb": 1000,
            "ttl_days": 90,
            "compression": True
        },
        
        "output": {
            "base_dir": ".pathwaylens/results",
            "formats": ["json", "markdown", "html", "graphml"],
            "include_plots": True,
            "include_tables": True,
            "include_graphs": True,
            "interactive_plots": True
        }
    }
    
    # Create directory if it doesn't exist
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    _save_config(default_config, config_file)
    
    console.print(f"[green]✅ Configuration file created: {config_file}[/green]")


@app.command()
def validate(
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    )
):
    """
    Validate configuration file.
    
    This command validates the PathwayLens configuration file,
    checking for syntax errors and invalid values.
    
    Examples:
        pathwaylens config validate
        pathwaylens config validate --config custom_config.yml
    """
    
    try:
        # Load configuration
        config = _load_config(config_file)
        
        # Validate configuration
        errors = _validate_config(config)
        
        if errors:
            console.print("[red]❌ Configuration validation failed:[/red]")
            for error in errors:
                console.print(f"  • {error}")
            raise typer.Exit(1)
        else:
            console.print("[green]✅ Configuration is valid[/green]")
            
    except Exception as e:
        console.print(f"[red]❌ Configuration validation failed: {e}[/red]")
        raise typer.Exit(1)


def _load_config(config_file: Optional[Path]) -> Dict[str, Any]:
    """Load configuration from file."""
    if config_file is None:
        config_file = Path.home() / ".pathwaylens" / "config.yml"
    
    if not config_file.exists():
        console.print(f"[red]Configuration file not found: {config_file}[/red]")
        console.print("Run 'pathwaylens config init' to create a default configuration")
        raise typer.Exit(1)
    
    try:
        with open(config_file, 'r') as f:
            if config_file.suffix.lower() in ['.yml', '.yaml']:
                return yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                return json.load(f)
            else:
                # Try YAML first, then JSON
                try:
                    f.seek(0)
                    return yaml.safe_load(f)
                except:
                    f.seek(0)
                    return json.load(f)
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        raise typer.Exit(1)


def _save_config(config: Dict[str, Any], config_file: Optional[Path]):
    """Save configuration to file."""
    if config_file is None:
        config_file = Path.home() / ".pathwaylens" / "config.yml"
    
    # Create directory if it doesn't exist
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_file, 'w') as f:
            if config_file.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif config_file.suffix.lower() == '.json':
                json.dump(config, f, indent=2)
            else:
                # Default to YAML
                yaml.dump(config, f, default_flow_style=False, indent=2)
    except Exception as e:
        console.print(f"[red]Error saving configuration: {e}[/red]")
        raise typer.Exit(1)


def _convert_value(value: str) -> Any:
    """Convert string value to appropriate type."""
    # Boolean values
    if value.lower() in ['true', 'yes', 'on', '1']:
        return True
    elif value.lower() in ['false', 'no', 'off', '0']:
        return False
    
    # Numeric values
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        pass
    
    # List values
    if value.startswith('[') and value.endswith(']'):
        try:
            return json.loads(value)
        except:
            pass
    
    # String value
    return value


def _validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration and return list of errors."""
    errors = []
    
    # Check required sections
    required_sections = ['databases', 'analysis', 'cache', 'output']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate databases section
    if 'databases' in config:
        for db_name, db_config in config['databases'].items():
            if not isinstance(db_config, dict):
                errors.append(f"Database '{db_name}' configuration must be a dictionary")
                continue
            
            required_db_fields = ['name', 'enabled', 'rate_limit', 'base_url']
            for field in required_db_fields:
                if field not in db_config:
                    errors.append(f"Database '{db_name}' missing required field: {field}")
    
    # Validate analysis section
    if 'analysis' in config:
        analysis_config = config['analysis']
        
        # Check numeric fields
        numeric_fields = ['min_pathway_size', 'max_pathway_size', 'gsea_permutations', 'gsea_min_size', 'gsea_max_size']
        for field in numeric_fields:
            if field in analysis_config:
                if not isinstance(analysis_config[field], (int, float)):
                    errors.append(f"Analysis field '{field}' must be numeric")
        
        # Check boolean fields
        boolean_fields = ['species_required', 'cross_species_allowed']
        for field in boolean_fields:
            if field in analysis_config:
                if not isinstance(analysis_config[field], bool):
                    errors.append(f"Analysis field '{field}' must be boolean")
    
    # Validate cache section
    if 'cache' in config:
        cache_config = config['cache']
        
        if 'enabled' in cache_config and not isinstance(cache_config['enabled'], bool):
            errors.append("Cache field 'enabled' must be boolean")
        
        if 'max_size_mb' in cache_config and not isinstance(cache_config['max_size_mb'], (int, float)):
            errors.append("Cache field 'max_size_mb' must be numeric")
        
        if 'ttl_days' in cache_config and not isinstance(cache_config['ttl_days'], (int, float)):
            errors.append("Cache field 'ttl_days' must be numeric")
    
    return errors


if __name__ == "__main__":
    app()
