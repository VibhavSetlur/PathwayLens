"""
Workflow templates for PathwayLens.
"""

from typing import Dict, Optional

WORKFLOW_TEMPLATES: Dict[str, str] = {
    "default": """workflow_name: default_workflow
steps:
  - step_id: norm1
    type: normalization
    input: input_file.txt
    species: human
    input_type: auto
    target_type: symbol
    ambiguity_policy: expand
  
  - step_id: enrich1
    type: analysis
    input: norm1
    method: ORA
    databases: [kegg, reactome]
    species: human
    fdr: 0.05
""",
    
    "gsea": """workflow_name: gsea_workflow
steps:
  - step_id: norm1
    type: normalization
    input: input_file.txt
    species: human
    input_type: auto
    target_type: symbol
  
  - step_id: gsea1
    type: analysis
    input: norm1
    method: GSEA
    databases: [go, kegg]
    species: human
    fdr: 0.05
    permutations: 1000
""",
    
    "multi_omics": """workflow_name: multi_omics_workflow
steps:
  - step_id: norm1
    type: normalization
    input: genomics_data.txt
    species: human
    input_type: auto
    target_type: symbol
  
  - step_id: norm2
    type: normalization
    input: proteomics_data.txt
    species: human
    input_type: auto
    target_type: symbol
  
  - step_id: multi_omics1
    type: analysis
    inputs: [norm1, norm2]
    method: MULTI_OMICS
    integration_method: multigsea
    databases: [kegg, reactome]
    species: human
    fdr: 0.05
""",
    
    "batch": """workflow_name: batch_workflow
steps:
  - step_id: batch_norm
    type: normalization
    input: batch_input_dir/
    species: human
    input_type: auto
    target_type: symbol
    batch: true
  
  - step_id: batch_enrich
    type: analysis
    input: batch_norm
    method: ORA
    databases: [kegg]
    species: human
    fdr: 0.05
"""
}


def get_template(name: str = "default") -> str:
    """
    Get a workflow template by name.
    
    Args:
        name: Template name (default, gsea, multi_omics, batch)
        
    Returns:
        Template YAML string
        
    Raises:
        ValueError: If template not found
    """
    if name not in WORKFLOW_TEMPLATES:
        available = ", ".join(WORKFLOW_TEMPLATES.keys())
        raise ValueError(
            f"Template '{name}' not found. Available templates: {available}"
        )
    
    return WORKFLOW_TEMPLATES[name]


def list_templates() -> list:
    """List all available workflow templates."""
    return list(WORKFLOW_TEMPLATES.keys())



