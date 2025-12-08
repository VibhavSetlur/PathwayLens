"""
Comparison commands for PathwayLens CLI.
"""

import typer
from typing import Optional, List
from pathlib import Path
from rich.console import Console

from pathwaylens_core.types import OmicType, DataType
from pathwaylens_core.species import Species
from pathwaylens_core.config import DEFAULT_CONFIG

console = Console()

def compare(
    inputs: List[str] = typer.Option(..., "--inputs", "-i", help="Input files to compare (REQUIRED)"),
    labels: List[str] = typer.Option(..., "--labels", "-l", help="Labels for each input (REQUIRED, must match number of inputs)"),
    comparison_type: str = typer.Option(..., "--comparison-type", help="sample-type, condition, or cross-omic (REQUIRED)"),
    stage: str = typer.Option(..., "--stage", help="counts, gene, or pathway (REQUIRED)"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    omic_type: Optional[OmicType] = typer.Option(None, "--omic-type", help="Omic type (required for pathway stage)"),
    data_type: Optional[DataType] = typer.Option(None, "--data-type", help="Data type (required for pathway stage)"),
    tool: str = typer.Option("auto", "--tool", help="Tool used to generate input (default: auto)"),
    databases: List[str] = typer.Option(["kegg"], "--databases", "-d", help="Databases (for pathway stage)"),
    species: str = typer.Option("human", "--species", "-s", help="Species"),
    method: str = typer.Option("deseq2", "--method", "-m", help="DE method (deseq2, edger, limma, simple)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """
    Compare multiple datasets at different stages (counts, gene, pathway).
    
    Examples:
        # Gene-stage comparison
        pathwaylens compare --inputs genes1.txt genes2.txt \\
            --labels "Treatment_A" "Control" \\
            --comparison-type condition --stage gene \\
            --output-dir comparison_results/
        
        # Pathway-stage comparison with gene lists (auto-enrichment)
        pathwaylens compare --inputs genes1.txt genes2.txt \\
            --labels "Treatment_A" "Control" \\
            --comparison-type condition --stage pathway \\
            --omic-type transcriptomics --data-type bulk \\
            --databases kegg reactome --output-dir comparison_results/
        
        # Pathway-stage comparison with existing enrichment results
        pathwaylens compare --inputs results1.json results2.json \\
            --labels "Study1" "Study2" \\
            --comparison-type condition --stage pathway \\
            --output-dir comparison_results/
    """
    
    # Validate inputs
    if len(labels) != len(inputs):
        console.print(f"[red]Error: Number of labels({len(labels)}) must match number of inputs ({len(inputs)})[/red]")
        console.print("[yellow]Example: --inputs file1.txt file2.txt --labels \"Label1\" \"Label2\"[/yellow]")
        raise typer.Exit(code=1)
    
    # Validate comparison type
    valid_comparison_types = ["sample-type", "condition", "cross-omic"]
    if comparison_type not in valid_comparison_types:
        console.print(f"[red]Error: Invalid comparison-type '{comparison_type}'[/red]")
        console.print(f"[yellow]Valid options: {', '.join(valid_comparison_types)}[/yellow]")
        raise typer.Exit(code=1)
    
    # Validate stage
    valid_stages = ["counts", "gene", "pathway"]
    if stage not in valid_stages:
        console.print(f"[red]Error: Invalid stage '{stage}'[/red]")
        console.print(f"[yellow]Valid options: {', '.join(valid_stages)}[/yellow]")
        raise typer.Exit(code=1)
    
    # Validate stage-specific requirements
    if stage == "pathway" and (not omic_type or not data_type):
        console.print("[red]Error: --omic-type and --data-type are REQUIRED for pathway-stage comparison[/red]")
        console.print("[yellow]Example: --omic-type transcriptomics --data-type bulk[/yellow]")
        raise typer.Exit(code=1)
    
    # Validate input files exist
    for input_file in inputs:
        if not Path(input_file).exists():
            console.print(f"[red]Error: Input file '{input_file}' does not exist[/red]")
            raise typer.Exit(code=1)
    
    # Map labels to files
    dataset_map = dict(zip(labels, inputs))
    
    # Resolve species
    species_info = Species.get(species)
    if not species_info:
        console.print(f"[red]Error: Unknown species '{species}'[/red]")
        raise typer.Exit(code=1)
    
    # Create output directory
    if output_dir.exists() and not output_dir.is_dir():
        console.print(f"[red]Error: Output path '{output_dir}' exists and is not a directory[/red]")
        raise typer.Exit(code=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Display configuration
    console.print(f"[bold cyan]PathwayLens Comparison Analysis[/bold cyan]")
    console.print(f"Comparison Type: {comparison_type}")
    if omic_type:
        console.print(f"Omic Type: {omic_type.value}")
    if data_type:
        console.print(f"Data Type: {data_type.value}")
    console.print(f"Stage: {stage}")
    console.print(f"Species: {species_info.common_name}")
    console.print(f"Inputs:")
    for label, file_path in dataset_map.items():
        console.print(f"  - {label}: {file_path}")
    console.print(f"Output Directory: {output_dir}")
    
    # Import comparison engine
    from pathwaylens_core.comparison.engine import ComparisonEngine
    from pathwaylens_core.comparison.schemas import (
        ComparisonParameters, ComparisonStage, ComparisonCategory,
        ComparisonType as CompType, InputType
    )
    from pathwaylens_core.analysis.schemas import DatabaseType
    
    # Map stage string to enum
    stage_map = {
        "counts": ComparisonStage.COUNTS,
        "gene": ComparisonStage.GENE,
        "pathway": ComparisonStage.PATHWAY
    }
    
    # Map comparison type string to enum
    category_map = {
        "sample-type": ComparisonCategory.SAMPLE_TYPE,
        "condition": ComparisonCategory.CONDITION,
        "cross-omic": ComparisonCategory.CROSS_OMIC
    }
    
    # Auto-detect input types
    input_types = {}
    for label, file_path in dataset_map.items():
        file_ext = Path(file_path).suffix.lower()
        if file_ext in [".json"]:
            input_types[label] = InputType.PATHWAY_RESULTS
        elif file_ext in [".csv", ".tsv", ".txt"]:
            # Could be gene list or count matrix - simple heuristic
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                if ',' in first_line or '\t' in first_line:
                    # Likely a matrix
                    input_types[label] = InputType.COUNT_MATRIX
                else:
                    # Likely a gene list
                    input_types[label] = InputType.GENE_LIST
        else:
            input_types[label] = InputType.GENE_LIST  # Default
    
    # Convert databases to enums
    if databases:
        try:
            db_enums = [DatabaseType(db) for db in databases]
        except ValueError:
            console.print(f"[red]Error: Invalid database. Supported: {', '.join([d.value for d in DatabaseType])}[/red]")
            raise typer.Exit(code=1)
    else:
        db_enums = [DatabaseType.KEGG]
    
    # Create comparison parameters
    try:
        params = ComparisonParameters(
            comparison_type=CompType.COMPREHENSIVE,  # Always use comprehensive for now
            comparison_stage=stage_map[stage],
            comparison_category=category_map[comparison_type],
            species=species_info.ensembl_name,
            input_labels=dataset_map,
            input_types=input_types,
            databases=[db.value for db in db_enums],
            run_enrichment_first=(stage == "pathway" and all(t == InputType.GENE_LIST for t in input_types.values()))
        )
    except Exception as e:
        console.print(f"[red]Error creating comparison parameters: {e}[/red]")
        raise typer.Exit(code=1)
    
    # Run comparison
    engine = ComparisonEngine()
    
    import asyncio
    try:
        console.print("[bold]Running comparison analysis...[/bold]")
        
        # Different workflows based on stage
        if stage == "gene":
            # Read gene lists
            gene_lists = {}
            for label, file_path in dataset_map.items():
                with open(file_path, 'r') as f:
                    genes = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                gene_lists[label] = genes
                console.print(f"  Loaded {len(genes)} genes from {label}")
            
            result = asyncio.run(engine.compare_gene_lists(gene_lists, params, str(output_dir)))
            
        elif stage == "pathway":
            # Check if we need to run enrichment first
            if params.run_enrichment_first:
                console.print("[yellow]Detected gene lists - will run pathway enrichment before comparison[/yellow]")
                console.print(f"  Databases: {', '.join(databases)}")
            
            result = asyncio.run(engine.compare_pathway_stage(
                dataset_map, params, str(output_dir),
                omic_type=omic_type, data_type=data_type, tool=tool
            ))
            
        elif stage == "counts":
            console.print(f"[bold]Running counts-stage comparison using {method}...[/bold]")
            result = asyncio.run(engine.compare_counts_stage(
                dataset_map, params, str(output_dir), method=method
            ))
        
        console.print("[green]✓ Comparison completed successfully![/green]")
        console.print(f"[green]Results saved to: {output_dir}[/green]")
        
    except Exception as e:
        console.print(f"[red]Comparison failed: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(code=1)