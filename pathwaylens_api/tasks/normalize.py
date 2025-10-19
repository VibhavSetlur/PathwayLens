"""
Normalization tasks for PathwayLens API.

This module provides Celery tasks for gene normalization.
"""

from celery import current_task
from pathwaylens_api.celery_app import celery_app
from pathwaylens_core.normalization.normalizer import Normalizer
from pathwaylens_core.normalization.schemas import InputData


@celery_app.task(bind=True, name="pathwaylens_api.tasks.normalize.normalize_genes")
def normalize_genes_task(self, job_id: str, data: list, species: str, target_type: str, 
                        target_species: str = None, ambiguity_policy: str = "expand"):
    """Normalize gene identifiers task."""
    return normalize_gene_ids(job_id, data, species, target_type, target_species, ambiguity_policy)


@celery_app.task(bind=True, name="pathwaylens_api.tasks.normalize.normalize_pathways")
def normalize_pathways_task(self, job_id: str, data: list, species: str, target_type: str, 
                           target_species: str = None, ambiguity_policy: str = "expand"):
    """Normalize pathway identifiers task."""
    return normalize_pathway_ids(job_id, data, species, target_type, target_species, ambiguity_policy)


@celery_app.task(bind=True, name="pathwaylens_api.tasks.normalize.normalize_omics")
def normalize_omics_task(self, job_id: str, data: list, species: str, target_type: str, 
                        target_species: str = None, ambiguity_policy: str = "expand"):
    """Normalize omics data task."""
    return normalize_omics_data(job_id, data, species, target_type, target_species, ambiguity_policy)


def normalize_gene_ids(job_id: str, data: list, species: str, target_type: str, 
                      target_species: str = None, ambiguity_policy: str = "expand"):
    """Normalize gene identifiers task."""
    try:
        # Update task progress
        current_task.update_state(state="PROGRESS", meta={"progress": 10, "message": "Starting normalization"})
        
        # Create input data object
        input_data = InputData(
            data=data,
            species=species,
            target_type=target_type,
            target_species=target_species,
            ambiguity_policy=ambiguity_policy
        )
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 30, "message": "Processing genes"})
        
        # Initialize normalizer
        normalizer = Normalizer()
        
        # Perform normalization
        normalized_table = normalizer.normalize(input_data)
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 80, "message": "Finalizing results"})
        
        # Convert to response format
        normalized_data = []
        for row in normalized_table.data:
            normalized_data.append({
                "original_id": row.get("original_id"),
                "normalized_id": row.get("normalized_id"),
                "confidence": row.get("confidence", 1.0),
                "metadata": row.get("metadata", {})
            })
        
        # Calculate conversion stats
        total_genes = len(data)
        converted_genes = len(normalized_data)
        conversion_rate = converted_genes / total_genes if total_genes > 0 else 0
        
        conversion_stats = {
            "total_genes": total_genes,
            "converted_genes": converted_genes,
            "conversion_rate": conversion_rate,
            "species": species,
            "target_type": target_type
        }
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 100, "message": "Completed"})
        
        return {
            "job_id": job_id,
            "status": "completed",
            "normalized_data": normalized_data,
            "conversion_stats": conversion_stats
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e), "message": "Normalization failed"}
        )
        raise


@celery_app.task(bind=True, name="pathwaylens_api.tasks.normalize.batch_normalize")
def batch_normalize_task(self, job_id: str, datasets: list, species: str, target_type: str, 
                        ambiguity_policy: str = "expand"):
    """Batch normalize multiple datasets task."""
    try:
        # Update task progress
        current_task.update_state(state="PROGRESS", meta={"progress": 5, "message": "Starting batch normalization"})
        
        results = []
        total_datasets = len(datasets)
        
        # Initialize normalizer
        normalizer = Normalizer()
        
        for i, dataset in enumerate(datasets):
            dataset_name = dataset.get("name", f"dataset_{i+1}")
            dataset_data = dataset.get("data", [])
            
            # Update progress
            progress = int(5 + (i / total_datasets) * 90)
            current_task.update_state(
                state="PROGRESS", 
                meta={"progress": progress, "message": f"Processing {dataset_name}"}
            )
            
            if not dataset_data:
                results.append({
                    "name": dataset_name,
                    "error": "Empty dataset",
                    "normalized_data": [],
                    "conversion_stats": {}
                })
                continue
            
            try:
                # Create input data object
                input_data = InputData(
                    data=dataset_data,
                    species=species,
                    target_type=target_type,
                    ambiguity_policy=ambiguity_policy
                )
                
                # Perform normalization
                normalized_table = normalizer.normalize(input_data)
                
                # Convert to response format
                normalized_data = []
                for row in normalized_table.data:
                    normalized_data.append({
                        "original_id": row.get("original_id"),
                        "normalized_id": row.get("normalized_id"),
                        "confidence": row.get("confidence", 1.0),
                        "metadata": row.get("metadata", {})
                    })
                
                # Calculate conversion stats
                total_genes = len(dataset_data)
                converted_genes = len(normalized_data)
                conversion_rate = converted_genes / total_genes if total_genes > 0 else 0
                
                conversion_stats = {
                    "total_genes": total_genes,
                    "converted_genes": converted_genes,
                    "conversion_rate": conversion_rate
                }
                
                results.append({
                    "name": dataset_name,
                    "normalized_data": normalized_data,
                    "conversion_stats": conversion_stats
                })
                
            except Exception as e:
                results.append({
                    "name": dataset_name,
                    "error": str(e),
                    "normalized_data": [],
                    "conversion_stats": {}
                })
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 100, "message": "Completed"})
        
        return {
            "job_id": job_id,
            "status": "completed",
            "results": results
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e), "message": "Batch normalization failed"}
        )
        raise


def normalize_pathway_ids(job_id: str, data: list, species: str, target_type: str, 
                         target_species: str = None, ambiguity_policy: str = "expand"):
    """Normalize pathway identifiers."""
    # For now, just call the gene normalization function
    return normalize_gene_ids(job_id, data, species, target_type, target_species, ambiguity_policy)


def normalize_omics_data(job_id: str, data: list, species: str, target_type: str, 
                        target_species: str = None, ambiguity_policy: str = "expand"):
    """Normalize omics data."""
    # For now, just call the gene normalization function
    return normalize_gene_ids(job_id, data, species, target_type, target_species, ambiguity_policy)
