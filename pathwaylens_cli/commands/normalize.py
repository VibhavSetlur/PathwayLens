"""
Normalization commands for PathwayLens CLI.
"""

import asyncio
import typer
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import yaml
from loguru import logger
from rich.console import Console

from ..utils.api_client import APIClient
from ..utils.config import Config
from ..utils.exceptions import CLIException
from pathwaylens_core.normalization.id_converter import IDConverter
from pathwaylens_core.normalization.schemas import IDType, SpeciesType, AmbiguityPolicy


app = typer.Typer(
    name="normalize",
    help="Convert gene identifiers across formats and species",
    rich_markup_mode="rich"
)

console = Console()


@app.command()
def gene_ids(
    input: str = typer.Option(..., "--input", "-i", help="Input gene ID file or list"),
    input_format: str = typer.Option("auto", "--input-format", "-if", help="Input format (auto, entrezgene, ensembl, symbol, uniprot)"),
    output_format: str = typer.Option("entrez", "--output-format", "-of", help="Output format (entrez, ensembl, symbol, uniprot)"),
    species: str = typer.Option("human", "--species", "-s", help="Species for normalization"),
    drop_unmapped: bool = typer.Option(True, "--drop-unmapped", "-d", help="Drop unmapped genes"),
    batch_size: int = typer.Option(1000, "--batch-size", "-b", help="Batch size for processing"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json, csv, tsv)"),
    service: str = typer.Option("mygene,ensembl,ncbi", "--service", help="Comma-separated list of services to use"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Normalize gene IDs."""
    try:
        if verbose:
            logger.info(f"Starting gene ID normalization from {input_format} to {output_format}")
        
        # Load configuration
        config = Config()
        
        # Create output directory if output file is specified
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create API client
        api_client = APIClient(config.api_url, config.api_key)
        
        # Prepare input data
        input_data = _prepare_input_data(input)
        
        # Parse services
        services = [s.strip() for s in service.split(',')]
        
        # Initialize converter
        converter = IDConverter()
        
        # Map strings to enums
        try:
            in_type = IDType(input_format) if input_format != "auto" else IDType.SYMBOL # Default to symbol if auto
            out_type = IDType(output_format)
            sp_type = SpeciesType(species)
        except ValueError as e:
            raise CLIException(f"Invalid parameter: {e}")

        # Start normalization
        # Start normalization
        async def run_conversion():
            async with converter:
                return await converter.convert_identifiers(
                    identifiers=input_data,
                    input_type=in_type,
                    output_type=out_type,
                    species=sp_type,
                    services=services
                )
        
        result_list = asyncio.run(run_conversion())
            
        # Convert result list to dict format expected by _save_result/_print_result
        result = {
            "job_id": "local",
            "status": "completed",
            "results": [
                {
                    "input_id": r.input_id,
                    "output_id": r.output_id,
                    "confidence": r.confidence,
                    "source": r.source,
                    "is_ambiguous": r.is_ambiguous,
                    "alternative_mappings": r.alternative_mappings
                }
                for r in result_list
            ]
        }
        
        # Handle output
        if output:
            output_path = Path(output)
            _save_result(result, output_path, format)
            
            if verbose:
                logger.info(f"Normalization result saved to: {output_path}")
        else:
            # Print result to stdout
            _print_result(result, format)
        
        if verbose:
            logger.info("Gene ID normalization completed successfully")
        
    except Exception as e:
        logger.error(f"Error in gene ID normalization: {e}")
        raise CLIException(f"Gene ID normalization failed: {e}")


@app.command()
def pathway_ids(
    input: str = typer.Option(..., "--input", "-i", help="Input pathway ID file or list"),
    input_database: str = typer.Option("auto", "--input-database", "-id", help="Input database (auto, KEGG, Reactome, GO, WikiPathways)"),
    output_database: str = typer.Option("KEGG", "--output-database", "-od", help="Output database (KEGG, Reactome, GO, WikiPathways)"),
    species: str = typer.Option("human", "--species", "-s", help="Species for normalization"),
    drop_unmapped: bool = typer.Option(True, "--drop-unmapped", "-d", help="Drop unmapped pathways"),
    batch_size: int = typer.Option(1000, "--batch-size", "-b", help="Batch size for processing"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json, csv, tsv)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Normalize pathway IDs."""
    try:
        if verbose:
            logger.info(f"Starting pathway ID normalization from {input_database} to {output_database}")
        
        # Load configuration
        config = Config()
        
        # Create API client
        api_client = APIClient(config.api_url, config.api_key)
        
        # Prepare input data
        input_data = _prepare_input_data(input)
        
        # Prepare parameters
        parameters = {
            "input_database": input_database,
            "output_database": output_database,
            "species": species,
            "drop_unmapped": drop_unmapped,
            "batch_size": batch_size,
            "output_format": format
        }
        
        # Start normalization
        result = asyncio.run(_start_normalization(api_client, "pathway-ids", input_data, parameters))
        
        # Handle output
        if output:
            output_path = Path(output)
            _save_result(result, output_path, format)
            
            if verbose:
                logger.info(f"Normalization result saved to: {output_path}")
        else:
            # Print result to stdout
            _print_result(result, format)
        
        if verbose:
            logger.info("Pathway ID normalization completed successfully")
        
    except Exception as e:
        logger.error(f"Error in pathway ID normalization: {e}")
        raise CLIException(f"Pathway ID normalization failed: {e}")


@app.command()
def omics_data(
    input: str = typer.Option(..., "--input", "-i", help="Input omics data file"),
    data_type: str = typer.Option("auto", "--data-type", "-dt", help="Data type (auto, genomics, transcriptomics, proteomics, metabolomics)"),
    normalization_method: str = typer.Option("zscore", "--normalization-method", "-nm", help="Normalization method (zscore, minmax, quantile, log2)"),
    species: str = typer.Option("human", "--species", "-s", help="Species for normalization"),
    batch_size: int = typer.Option(1000, "--batch-size", "-b", help="Batch size for processing"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json, csv, tsv)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Normalize omics data."""
    try:
        if verbose:
            logger.info(f"Starting omics data normalization with method: {normalization_method}")
        
        # Load configuration
        config = Config()
        
        # Create API client
        api_client = APIClient(config.api_url, config.api_key)
        
        # Prepare input data
        input_data = _prepare_input_data(input)
        
        # Prepare parameters
        parameters = {
            "data_type": data_type,
            "normalization_method": normalization_method,
            "species": species,
            "batch_size": batch_size,
            "output_format": format
        }
        
        # Start normalization
        result = asyncio.run(_start_normalization(api_client, "omics-data", input_data, parameters))
        
        # Handle output
        if output:
            output_path = Path(output)
            _save_result(result, output_path, format)
            
            if verbose:
                logger.info(f"Normalization result saved to: {output_path}")
        else:
            # Print result to stdout
            _print_result(result, format)
        
        if verbose:
            logger.info("Omics data normalization completed successfully")
        
    except Exception as e:
        logger.error(f"Error in omics data normalization: {e}")
        raise CLIException(f"Omics data normalization failed: {e}")


@app.command()
def status(
    job_id: str = typer.Option(..., "--job-id", "-j", help="Normalization job ID"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Get normalization job status."""
    try:
        if verbose:
            logger.info(f"Getting status for normalization job: {job_id}")
        
        # Load configuration
        config = Config()
        
        # Create API client
        api_client = APIClient(config.api_url, config.api_key)
        
        # Get job status
        status = asyncio.run(_get_job_status(api_client, "normalize", job_id))
        
        # Print status
        _print_status(status)
        
        if verbose:
            logger.info("Job status retrieved successfully")
        
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise CLIException(f"Failed to get job status: {e}")


@app.command()
def result(
    job_id: str = typer.Option(..., "--job-id", "-j", help="Normalization job ID"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json, csv, tsv)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Get normalization job result."""
    try:
        if verbose:
            logger.info(f"Getting result for normalization job: {job_id}")
        
        # Load configuration
        config = Config()
        
        # Create API client
        api_client = APIClient(config.api_url, config.api_key)
        
        # Get job result
        result = asyncio.run(_get_job_result(api_client, "normalize", job_id))
        
        # Handle output
        if output:
            output_path = Path(output)
            _save_result(result, output_path, format)
            
            if verbose:
                logger.info(f"Normalization result saved to: {output_path}")
        else:
            # Print result to stdout
            _print_result(result, format)
        
        if verbose:
            logger.info("Normalization result retrieved successfully")
        
    except Exception as e:
        logger.error(f"Error getting job result: {e}")
        raise CLIException(f"Failed to get job result: {e}")


@app.command()
def list_jobs(
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of jobs to return"),
    offset: int = typer.Option(0, "--offset", "-o", help="Number of jobs to skip"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """List normalization jobs."""
    try:
        if verbose:
            logger.info("Listing normalization jobs")
        
        # Load configuration
        config = Config()
        
        # Create API client
        api_client = APIClient(config.api_url, config.api_key)
        
        # List jobs
        jobs = asyncio.run(_list_jobs(api_client, "normalize", limit, offset))
        
        # Print jobs
        _print_jobs(jobs)
        
        if verbose:
            logger.info("Normalization jobs listed successfully")
        
    except Exception as e:
        logger.error(f"Error listing normalization jobs: {e}")
        raise CLIException(f"Failed to list normalization jobs: {e}")


@app.command()
def delete(
    job_id: str = typer.Option(..., "--job-id", "-j", help="Normalization job ID"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Delete normalization job."""
    try:
        if verbose:
            logger.info(f"Deleting normalization job: {job_id}")
        
        # Load configuration
        config = Config()
        
        # Create API client
        api_client = APIClient(config.api_url, config.api_key)
        
        # Delete job
        asyncio.run(_delete_job(api_client, "normalize", job_id))
        
        if verbose:
            logger.info("Normalization job deleted successfully")
        
    except Exception as e:
        logger.error(f"Error deleting normalization job: {e}")
        raise CLIException(f"Failed to delete normalization job: {e}")


def _prepare_input_data(input_path: str) -> Any:
    """Prepare input data for normalization."""
    input_file = Path(input_path)
    
    if not input_file.exists():
        raise CLIException(f"Input file not found: {input_path}")
    
    # Read input data based on file extension
    if input_file.suffix.lower() == '.json':
        with open(input_file, 'r') as f:
            return json.load(f)
    elif input_file.suffix.lower() in ['.yaml', '.yml']:
        with open(input_file, 'r') as f:
            return yaml.safe_load(f)
    elif input_file.suffix.lower() in ['.csv', '.tsv']:
        import pandas as pd
        return pd.read_csv(input_file).to_dict(orient='records')
    else:
        # Assume it's a text file with IDs
        with open(input_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]


async def _start_normalization(api_client: APIClient, endpoint: str, input_data: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Start normalization job."""
    request_data = {
        "input_data": input_data,
        "parameters": parameters
    }
    
    response = await api_client.post(f"/normalize/{endpoint}", request_data)
    return response


async def _get_job_status(api_client: APIClient, job_type: str, job_id: str) -> Dict[str, Any]:
    """Get job status."""
    response = await api_client.get(f"/{job_type}/status/{job_id}")
    return response


async def _get_job_result(api_client: APIClient, job_type: str, job_id: str) -> Dict[str, Any]:
    """Get job result."""
    response = await api_client.get(f"/{job_type}/result/{job_id}")
    return response


async def _list_jobs(api_client: APIClient, job_type: str, limit: int, offset: int) -> List[Dict[str, Any]]:
    """List jobs."""
    params = {"limit": limit, "offset": offset}
    response = await api_client.get(f"/{job_type}/jobs", params=params)
    return response


async def _delete_job(api_client: APIClient, job_type: str, job_id: str) -> None:
    """Delete job."""
    await api_client.delete(f"/{job_type}/job/{job_id}")


def _save_result(result: Dict[str, Any], output_path: Path, format: str) -> None:
    """Save normalization result to file."""
    if format.lower() == 'json':
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
    elif format.lower() == 'yaml':
        with open(output_path, 'w') as f:
            yaml.dump(result, f, default_flow_style=False)
    elif format.lower() in ['csv', 'tsv']:
        import pandas as pd
        # Convert result to DataFrame and save
        if 'results' in result:
            df = pd.DataFrame(result['results'])
            df.to_csv(output_path, index=False, sep='\t' if format.lower() == 'tsv' else ',')
        else:
            # Save as single row
            df = pd.DataFrame([result])
            df.to_csv(output_path, index=False, sep='\t' if format.lower() == 'tsv' else ',')
    else:
        raise CLIException(f"Unsupported output format: {format}")


def _print_result(result: Dict[str, Any], format: str) -> None:
    """Print normalization result to stdout."""
    if format.lower() == 'json':
        print(json.dumps(result, indent=2))
    elif format.lower() == 'yaml':
        print(yaml.dump(result, default_flow_style=False))
    else:
        # Print summary
        print(f"Normalization completed successfully")
        print(f"Job ID: {result.get('job_id', 'N/A')}")
        print(f"Status: {result.get('status', 'N/A')}")
        if 'results' in result:
            print(f"Results: {len(result['results'])} items")


def _print_status(status: Dict[str, Any]) -> None:
    """Print job status."""
    print(f"Job ID: {status.get('job_id', 'N/A')}")
    print(f"Status: {status.get('status', 'N/A')}")
    print(f"Progress: {status.get('progress', 0)}%")
    print(f"Message: {status.get('message', 'N/A')}")
    print(f"Created: {status.get('created_at', 'N/A')}")
    print(f"Updated: {status.get('updated_at', 'N/A')}")


def _print_jobs(jobs: List[Dict[str, Any]]) -> None:
    """Print list of jobs."""
    if not jobs:
        print("No normalization jobs found")
        return
    
    print(f"Found {len(jobs)} normalization jobs:")
    print("-" * 80)
    for job in jobs:
        print(f"Job ID: {job.get('job_id', 'N/A')}")
        print(f"Status: {job.get('status', 'N/A')}")
        print(f"Progress: {job.get('progress', 0)}%")
        print(f"Created: {job.get('created_at', 'N/A')}")
        print("-" * 80)