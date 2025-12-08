"""
Analysis commands for PathwayLens CLI.
"""

import asyncio
import typer
from typing import Optional, List
from pathlib import Path
from rich.console import Console

from pathwaylens_core.analysis import AnalysisEngine, SingleCellEngine
from pathwaylens_core.analysis.schemas import AnalysisType, DatabaseType, AnalysisParameters
from pathwaylens_core.types import OmicType, DataType
from pathwaylens_core.species import Species
from pathwaylens_core.config import DEFAULT_CONFIG

app = typer.Typer(
    name="analyze",
    help="Perform pathway analysis",
    rich_markup_mode="rich"
)

console = Console()

@app.command()
def ora(
    input: str = typer.Option(..., "--input", "-i", help="Input file"),
    omic_type: OmicType = typer.Option(..., "--omic-type", help="Omic type (transcriptomics, proteomics, etc.)"),
    data_type: DataType = typer.Option(..., "--data-type", help="Specific data type (bulk, shotgun, etc.)"),
    tool: str = typer.Option("auto", "--tool", help="Tool used to generate input (default: auto-detect)"),
    databases: List[str] = typer.Option(["kegg"], "--databases", "-d", help="Databases to use"),
    species: str = typer.Option("human", "--species", "-s", help="Species"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Exact output directory path"),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Optional run name for metadata"),
    min_genes: int = typer.Option(DEFAULT_CONFIG.MIN_GENES, "--min-genes", help="Minimum pathway size"),
    max_genes: int = typer.Option(DEFAULT_CONFIG.MAX_GENES, "--max-genes", help="Maximum pathway size"),
    fdr_threshold: float = typer.Option(DEFAULT_CONFIG.FDR_THRESHOLD, "--fdr-threshold", help="FDR cutoff"),
    lfc_threshold: float = typer.Option(DEFAULT_CONFIG.LFC_THRESHOLD, "--lfc-threshold", help="Log fold change threshold"),

    background: Optional[str] = typer.Option(None, "--background", "-b", help="Background genes file or size (int)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    ctx: typer.Context = typer.Option(None, hidden=True)
):
    """Perform Over-Representation Analysis (ORA)."""
    # Load configuration if available
    from pathwaylens_cli.utils.config_loader import ConfigLoader
    from click.core import ParameterSource
    
    config_file = ctx.meta.get("config_file") if ctx else None
    if config_file:
        try:
            file_config = ConfigLoader.load_config(config_file)
            
            # Update parameters from config if not explicitly set via CLI
            # We map CLI param names to config keys (usually same, but dashes vs underscores)
            for param_name, param_value in ctx.params.items():
                # Skip the config_file param itself and ctx
                if param_name in ["ctx", "config_file"]:
                    continue
                    
                source = ctx.get_parameter_source(param_name)
                if source != ParameterSource.COMMANDLINE:
                    # Check if this parameter is in the config file
                    # Config keys usually match variable names (underscores)
                    if param_name in file_config:
                        # Update the local variable
                        locals()[param_name] = file_config[param_name]
                        # Also update ctx.params for consistency if needed, though locals is what matters for execution
                        ctx.params[param_name] = file_config[param_name]
                        
            # Re-assign variables from locals() to ensure they are updated
            # This is a bit hacky but necessary since function args are local variables
            input = locals().get('input', input)
            omic_type = locals().get('omic_type', omic_type)
            data_type = locals().get('data_type', data_type)
            tool = locals().get('tool', tool)
            databases = locals().get('databases', databases)
            species = locals().get('species', species)
            output_dir = locals().get('output_dir', output_dir)
            run_name = locals().get('run_name', run_name)
            min_genes = locals().get('min_genes', min_genes)
            max_genes = locals().get('max_genes', max_genes)
            fdr_threshold = locals().get('fdr_threshold', fdr_threshold)
            lfc_threshold = locals().get('lfc_threshold', lfc_threshold)
            background = locals().get('background', background)
            verbose = locals().get('verbose', verbose)
            
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load config file: {e}[/yellow]")

    # ... rest of the function ...
    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file '{input}' does not exist[/red]")
        raise typer.Exit(code=1)
    
    # Resolve species
    species_info = Species.get(species)
    if not species_info:
        console.print(f"[red]Error: Unknown species '{species}'[/red]")
        raise typer.Exit(code=1)

    # Create output directory (exact path)
    if output_dir.exists() and not output_dir.is_dir():
        console.print(f"[red]Error: Output path '{output_dir}' exists and is not a directory[/red]")
        raise typer.Exit(code=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold]Running ORA analysis on {input}[/bold]")
    console.print(f"Species: {species_info.common_name}")
    console.print(f"Omic Type: {omic_type.value}, Data Type: {data_type.value}")
    console.print(f"Output Directory: {output_dir}")

    # Initialize engine
    engine = AnalysisEngine()
    
    # Create parameters
    # Convert database strings to DatabaseType enum
    try:
        db_enums = [DatabaseType(db) for db in databases]
    except ValueError as e:
        console.print(f"[red]Error: Invalid database. Supported: {', '.join([d.value for d in DatabaseType])}[/red]")
        raise typer.Exit(code=1)


    # Parse background
    bg_genes = None
    bg_size = None
    if background:
        if background.isdigit():
            bg_size = int(background)
        else:
            bg_path = Path(background)
            if bg_path.exists():
                # Read file
                try:
                    with open(bg_path, 'r') as f:
                        bg_genes = [line.strip() for line in f if line.strip()]
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to read background file: {e}[/yellow]")
            else:
                console.print(f"[yellow]Warning: Background file '{background}' not found. Ignoring.[/yellow]")

    params = AnalysisParameters(
        analysis_type=AnalysisType.ORA,
        omic_type=omic_type,
        data_type=data_type,
        databases=db_enums,
        species=species_info.ensembl_name, # Use internal name
        min_pathway_size=min_genes,
        max_pathway_size=max_genes,
        significance_threshold=fdr_threshold,
        lfc_threshold=lfc_threshold,
        custom_background=bg_genes,
        background_size=bg_size,
        tool=tool
    )

    try:
        result = engine.analyze_sync(
            input_data=str(input_path),
            parameters=params,
            output_dir=str(output_dir)
        )
        
        if result.errors:
            for error in result.errors:
                console.print(f"[red]{error}[/red]")
            raise typer.Exit(code=1)
            
        console.print(f"[green]Analysis completed successfully![/green]")
        console.print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")
        raise typer.Exit(code=1)

@app.command()
def gsea(
    input: str = typer.Option(..., "--input", "-i", help="Input ranked gene list file"),
    omic_type: OmicType = typer.Option(..., "--omic-type", help="Omic type"),
    data_type: DataType = typer.Option(..., "--data-type", help="Data type"),
    tool: str = typer.Option("auto", "--tool", help="Tool used (default: auto-detect)"),
    databases: List[str] = typer.Option(["kegg"], "--databases", "-d", help="Databases to use"),
    species: str = typer.Option("human", "--species", "-s", help="Species"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Exact output directory path"),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Optional run name"),
    min_genes: int = typer.Option(DEFAULT_CONFIG.MIN_GENES, "--min-genes", help="Minimum pathway size"),
    max_genes: int = typer.Option(DEFAULT_CONFIG.MAX_GENES, "--max-genes", help="Maximum pathway size"),
    fdr_threshold: float = typer.Option(DEFAULT_CONFIG.FDR_THRESHOLD, "--fdr-threshold", help="FDR cutoff"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Perform Gene Set Enrichment Analysis (GSEA)."""
    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file '{input}' does not exist[/red]")
        raise typer.Exit(code=1)
    
    species_info = Species.get(species)
    if not species_info:
        console.print(f"[red]Error: Unknown species '{species}'[/red]")
        raise typer.Exit(code=1)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold]Running GSEA analysis on {input}[/bold]")
    console.print(f"Species: {species_info.common_name}")
    console.print(f"Output Directory: {output_dir}")

    # Initialize engine
    engine = AnalysisEngine()
    
    try:
        db_enums = [DatabaseType(db) for db in databases]
    except ValueError:
        console.print(f"[red]Error: Invalid database. Supported: {', '.join([d.value for d in DatabaseType])}[/red]")
        raise typer.Exit(code=1)

    params = AnalysisParameters(
        analysis_type=AnalysisType.GSEA,
        omic_type=omic_type,
        data_type=data_type,
        databases=db_enums,
        species=species_info.ensembl_name,
        min_pathway_size=min_genes,
        max_pathway_size=max_genes,
        significance_threshold=fdr_threshold,
        gsea_permutations=1000, # Default or add arg
        tool=tool
    )

    try:
        result = engine.analyze_sync(
            input_data=str(input_path),
            parameters=params,
            output_dir=str(output_dir)
        )
        
        if result.errors:
            for error in result.errors:
                console.print(f"[red]{error}[/red]")
            raise typer.Exit(code=1)
            
        console.print(f"[green]Analysis completed successfully![/green]")
        console.print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")
        raise typer.Exit(code=1)
@app.command()
def single_cell(
    input: str = typer.Option(..., "--input", "-i", help="Input expression matrix (CSV/TSV/H5AD)"),
    database: str = typer.Option("kegg", "--database", "-d", help="Database to use"),
    species: str = typer.Option("human", "--species", "-s", help="Species"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    min_genes: int = typer.Option(5, "--min-genes", help="Minimum pathway size"),
    max_genes: int = typer.Option(500, "--max-genes", help="Maximum pathway size"),
    method: str = typer.Option("mean_zscore", "--method", "-m", help="Scoring method (mean, mean_zscore)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Perform Single-Cell Pathway Scoring."""
    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file '{input}' does not exist[/red]")
        raise typer.Exit(code=1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize engine
    from pathwaylens_core.data import DatabaseManager
    db_manager = DatabaseManager()
    engine = SingleCellEngine(db_manager)
    
    try:
        # Load data (simplified loading for now)
        import pandas as pd
        if input_path.suffix == '.h5ad':
            from pathwaylens_core.io.r_loader import RLoader
            df = RLoader.load_h5ad(input_path)
        else:
            df = pd.read_csv(input_path, index_col=0)
            
        if df is None or df.empty:
             raise ValueError("Failed to load expression data")
             
        # Run analysis
        # We need to run async code in sync command
        result = asyncio.run(engine.score_single_cells(
            expression_matrix=df,
            database=DatabaseType(database),
            species=species,
            min_pathway_size=min_genes,
            max_pathway_size=max_genes,
            method=method
        ))
        
        # Save results
        result.pathway_scores.to_csv(output_dir / "pathway_scores.csv")
        
        console.print(f"[green]Single-cell analysis completed![/green]")
        console.print(f"Scores saved to: {output_dir}/pathway_scores.csv")
        
    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")
        raise typer.Exit(code=1)

@app.command()
def batch(
    inputs: List[str] = typer.Option(..., "--inputs", "-i", help="Input files to process"),
    method: str = typer.Option("ora", "--method", "-m", help="Analysis method (ora, gsea, etc.)"),
    omic_type: OmicType = typer.Option(..., "--omic-type", help="Omic type"),
    data_type: DataType = typer.Option(..., "--data-type", help="Data type"),
    databases: List[str] = typer.Option(["kegg"], "--databases", "-d", help="Databases to use"),
    species: str = typer.Option("human", "--species", "-s", help="Species"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Process multiple files in batch."""
    console.print(f"[bold]Processing {len(inputs)} files in batch[/bold]")
    
    # Implementation would use BatchProcessor
    # This is a placeholder