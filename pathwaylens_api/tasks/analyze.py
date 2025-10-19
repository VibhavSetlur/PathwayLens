"""
Analysis tasks for PathwayLens API.

This module provides Celery tasks for pathway analysis.
"""

from celery import current_task
from pathwaylens_api.celery_app import celery_app
from pathwaylens_core.analysis.engine import AnalysisEngine


@celery_app.task(bind=True, name="pathwaylens_api.tasks.analyze.analyze_ora")
def analyze_ora_task(self, job_id: str, genes: list, species: str, databases: list, parameters: dict):
    """Perform Over-Representation Analysis task."""
    try:
        # Update task progress
        current_task.update_state(state="PROGRESS", meta={"progress": 10, "message": "Starting ORA analysis"})
        
        # Initialize analysis engine
        engine = AnalysisEngine()
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 30, "message": "Querying databases"})
        
        # Perform ORA analysis
        result = engine.analyze_ora(
            genes=genes,
            species=species,
            databases=databases,
            parameters=parameters
        )
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 80, "message": "Processing results"})
        
        # Convert results to response format
        database_results = {}
        for db_name, pathways in result.database_results.items():
            database_results[db_name] = []
            for pathway in pathways:
                database_results[db_name].append({
                    "pathway_id": pathway.pathway_id,
                    "pathway_name": pathway.pathway_name,
                    "p_value": pathway.p_value,
                    "adjusted_p_value": pathway.adjusted_p_value,
                    "genes": pathway.genes,
                    "gene_ratio": pathway.gene_ratio,
                    "background_ratio": pathway.background_ratio
                })
        
        consensus_results = []
        for pathway in result.consensus_results:
            consensus_results.append({
                "pathway_id": pathway.pathway_id,
                "pathway_name": pathway.pathway_name,
                "consensus_score": pathway.consensus_score,
                "databases": pathway.databases,
                "p_value": pathway.p_value,
                "adjusted_p_value": pathway.adjusted_p_value
            })
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 100, "message": "Completed"})
        
        return {
            "job_id": job_id,
            "status": "completed",
            "analysis_type": "ora",
            "species": species,
            "input_gene_count": len(genes),
            "total_pathways": result.total_pathways,
            "significant_pathways": result.significant_pathways,
            "database_results": database_results,
            "consensus_results": consensus_results
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e), "message": "ORA analysis failed"}
        )
        raise


@celery_app.task(bind=True, name="pathwaylens_api.tasks.analyze.analyze_gsea")
def analyze_gsea_task(self, job_id: str, ranked_genes: list, species: str, databases: list, parameters: dict):
    """Perform Gene Set Enrichment Analysis task."""
    try:
        # Update task progress
        current_task.update_state(state="PROGRESS", meta={"progress": 10, "message": "Starting GSEA analysis"})
        
        # Initialize analysis engine
        engine = AnalysisEngine()
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 30, "message": "Querying databases"})
        
        # Perform GSEA analysis
        result = engine.analyze_gsea(
            ranked_genes=ranked_genes,
            species=species,
            databases=databases,
            parameters=parameters
        )
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 80, "message": "Processing results"})
        
        # Convert results to response format
        database_results = {}
        for db_name, pathways in result.database_results.items():
            database_results[db_name] = []
            for pathway in pathways:
                database_results[db_name].append({
                    "pathway_id": pathway.pathway_id,
                    "pathway_name": pathway.pathway_name,
                    "enrichment_score": pathway.enrichment_score,
                    "normalized_enrichment_score": pathway.normalized_enrichment_score,
                    "p_value": pathway.p_value,
                    "adjusted_p_value": pathway.adjusted_p_value,
                    "leading_edge": pathway.leading_edge
                })
        
        consensus_results = []
        for pathway in result.consensus_results:
            consensus_results.append({
                "pathway_id": pathway.pathway_id,
                "pathway_name": pathway.pathway_name,
                "consensus_score": pathway.consensus_score,
                "databases": pathway.databases,
                "enrichment_score": pathway.enrichment_score,
                "p_value": pathway.p_value,
                "adjusted_p_value": pathway.adjusted_p_value
            })
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 100, "message": "Completed"})
        
        return {
            "job_id": job_id,
            "status": "completed",
            "analysis_type": "gsea",
            "species": species,
            "input_gene_count": len(ranked_genes),
            "total_pathways": result.total_pathways,
            "significant_pathways": result.significant_pathways,
            "database_results": database_results,
            "consensus_results": consensus_results
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e), "message": "GSEA analysis failed"}
        )
        raise


@celery_app.task(bind=True, name="pathwaylens_api.tasks.analyze.batch_analyze")
def batch_analyze_task(self, job_id: str, datasets: list, species: str, analysis_type: str, 
                      databases: list, parameters: dict):
    """Batch analyze multiple datasets task."""
    try:
        # Update task progress
        current_task.update_state(state="PROGRESS", meta={"progress": 5, "message": "Starting batch analysis"})
        
        results = []
        total_datasets = len(datasets)
        
        # Initialize analysis engine
        engine = AnalysisEngine()
        
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
                    "results": {}
                })
                continue
            
            try:
                if analysis_type == "ora":
                    # Extract genes from dataset
                    genes = [row.get("gene", row.get("gene_id", row.get("symbol", ""))) for row in dataset_data]
                    genes = [g for g in genes if g]  # Remove empty genes
                    
                    # Perform ORA analysis
                    result = engine.analyze_ora(
                        genes=genes,
                        species=species,
                        databases=databases,
                        parameters=parameters
                    )
                    
                elif analysis_type == "gsea":
                    # Perform GSEA analysis
                    result = engine.analyze_gsea(
                        ranked_genes=dataset_data,
                        species=species,
                        databases=databases,
                        parameters=parameters
                    )
                
                else:
                    raise ValueError(f"Unsupported analysis type: {analysis_type}")
                
                # Convert results to response format
                database_results = {}
                for db_name, pathways in result.database_results.items():
                    database_results[db_name] = []
                    for pathway in pathways:
                        pathway_data = {
                            "pathway_id": pathway.pathway_id,
                            "pathway_name": pathway.pathway_name,
                            "p_value": pathway.p_value,
                            "adjusted_p_value": pathway.adjusted_p_value
                        }
                        
                        if analysis_type == "gsea":
                            pathway_data.update({
                                "enrichment_score": pathway.enrichment_score,
                                "normalized_enrichment_score": pathway.normalized_enrichment_score,
                                "leading_edge": pathway.leading_edge
                            })
                        else:
                            pathway_data.update({
                                "genes": pathway.genes,
                                "gene_ratio": pathway.gene_ratio,
                                "background_ratio": pathway.background_ratio
                            })
                        
                        database_results[db_name].append(pathway_data)
                
                consensus_results = []
                for pathway in result.consensus_results:
                    consensus_data = {
                        "pathway_id": pathway.pathway_id,
                        "pathway_name": pathway.pathway_name,
                        "consensus_score": pathway.consensus_score,
                        "databases": pathway.databases,
                        "p_value": pathway.p_value,
                        "adjusted_p_value": pathway.adjusted_p_value
                    }
                    
                    if analysis_type == "gsea":
                        consensus_data["enrichment_score"] = pathway.enrichment_score
                    
                    consensus_results.append(consensus_data)
                
                results.append({
                    "name": dataset_name,
                    "analysis_type": analysis_type,
                    "species": species,
                    "input_gene_count": len(dataset_data),
                    "total_pathways": result.total_pathways,
                    "significant_pathways": result.significant_pathways,
                    "database_results": database_results,
                    "consensus_results": consensus_results
                })
                
            except Exception as e:
                results.append({
                    "name": dataset_name,
                    "error": str(e),
                    "results": {}
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
            meta={"error": str(e), "message": "Batch analysis failed"}
        )
        raise


def analyze_pathway_enrichment(job_id: str, genes: list, species: str, databases: list, parameters: dict):
    """Analyze pathway enrichment."""
    # For now, just call the ORA analysis function
    return analyze_ora_task(job_id, genes, species, databases, parameters)


def analyze_multi_omics(job_id: str, datasets: list, species: str, databases: list, parameters: dict):
    """Analyze multi-omics data."""
    # For now, just call the batch analysis function
    return batch_analyze_task(job_id, datasets, species, databases, parameters)


def analyze_statistical(job_id: str, data: list, species: str, databases: list, parameters: dict):
    """Perform statistical analysis."""
    # For now, just call the ORA analysis function
    return analyze_ora_task(job_id, data, species, databases, parameters)
