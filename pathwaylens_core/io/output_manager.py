"""
Output directory management for structured pathway analysis results.

This module provides professional output organization with:
- Structured directory hierarchies
- Automatic file naming
- Metadata preservation
- Publication-ready organization
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd

from ..analysis.schemas import AnalysisResult, DatabaseResult, PathwayResult
from ..schemas.provenance import AnalysisManifest
from ..utils.manifest_generator import save_manifest


class AnalysisOutputManager:
    """
    Manages structured output directories for pathway analysis.
    
    Creates publication-ready directory structure with organized results,
    figures, data, and methods documentation.
    """
    
    def __init__(self, base_dir: Union[str, Path], analysis_id: str):
        """
        Initialize output manager.
        
        Args:
            base_dir: Base directory for outputs
            analysis_id: Unique analysis identifier
        """
        self.base_dir = Path(base_dir)
        self.analysis_id = analysis_id
        
        # Create timestamped directory name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.base_dir / f"pathway_analysis_{timestamp}_{analysis_id}"
        
        self.directories = {}
    
    def create_directory_structure(self) -> Dict[str, Path]:
        """
        Create standardized directory structure.
        
        Returns:
            Dictionary mapping directory names to paths
        """
        # Define directory structure
        subdirs = {
            "root": self.output_dir,
            "results": self.output_dir / "results",
            "figures": self.output_dir / "figures",
            "data": self.output_dir / "data",
            "data_normalized": self.output_dir / "data" / "normalized",
            "data_intermediate": self.output_dir / "data" / "intermediate",
            "data_database": self.output_dir / "data" / "database_metadata",
            "methods": self.output_dir / "methods",
            "diagnostics": self.output_dir / "diagnostics"
        }
        
        # Create all directories
        for name, path in subdirs.items():
            path.mkdir(parents=True, exist_ok=True)
            self.directories[name] = path
        
        return self.directories
    
    def save_results(
        self,
        results: Union[AnalysisResult, DatabaseResult],
        format: str = "tsv"
    ) -> List[Path]:
        """
        Save analysis results in structured format.
        
        Args:
            results: AnalysisResult or DatabaseResult to save
            format: Output format ('tsv', 'csv', 'json')
            
        Returns:
            List of created file paths
        """
        created_files = []
        results_dir = self.directories["results"]
        
        if isinstance(results, AnalysisResult):
            # Save comprehensive results
            for db_name, db_result in results.database_results.items():
                files = self._save_database_result(db_result, db_name, format)
                created_files.extend(files)
            
            # Save summary statistics
            summary_path = results_dir / "summary_statistics.json"
            summary = {
                "analysis_id": results.job_id,
                "analysis_type": results.analysis_type.value,
                "total_pathways": results.total_pathways,
                "significant_pathways": results.significant_pathways,
                "significant_databases": results.significant_databases,
                "overall_quality": results.overall_quality,
                "reproducibility": results.reproducibility,
                "processing_time": results.processing_time
            }
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            created_files.append(summary_path)
            
        elif isinstance(results, DatabaseResult):
            files = self._save_database_result(results, results.database.value, format)
            created_files.extend(files)
        
        return created_files
    
    def _save_database_result(
        self,
        db_result: DatabaseResult,
        db_name: str,
        format: str
    ) -> List[Path]:
        """Save database-specific results."""
        created_files = []
        results_dir = self.directories["results"]
        
        # Convert pathways to DataFrame
        pathway_data = []
        for pathway in db_result.pathways:
            row = {
                "pathway_id": pathway.pathway_id,
                "pathway_name": pathway.pathway_name,
                "database": pathway.database.value,
                "p_value": pathway.p_value,
                "adjusted_p_value": pathway.adjusted_p_value,
                "enrichment_score": pathway.enrichment_score,
                # Research-grade statistics
                "odds_ratio": pathway.odds_ratio,
                "odds_ratio_ci_lower": pathway.odds_ratio_ci_lower,
                "odds_ratio_ci_upper": pathway.odds_ratio_ci_upper,
                "fold_enrichment": pathway.fold_enrichment,
                "effect_size": pathway.effect_size,
                "genes_expected": pathway.genes_expected,
                "statistical_power": pathway.statistical_power,
                # Gene counts
                "overlap_count": pathway.overlap_count,
                "pathway_count": pathway.pathway_count,
                "input_count": pathway.input_count,
                "overlapping_genes": ";".join(pathway.overlapping_genes),
                "pathway_description": pathway.pathway_description,
                "pathway_url": pathway.pathway_url
            }
            pathway_data.append(row)
        
        df = pd.DataFrame(pathway_data)
        
        # Save significant pathways
        df_significant = df[df['adjusted_p_value'] <= 0.05]
        if format == "tsv":
            sig_path = results_dir / f"{db_name}_enrichment.tsv"
            df_significant.to_csv(sig_path, sep='\t', index=False)
        elif format == "csv":
            sig_path = results_dir / f"{db_name}_enrichment.csv"
            df_significant.to_csv(sig_path, index=False)
        else:  # json
            sig_path = results_dir / f"{db_name}_enrichment.json"
            df_significant.to_json(sig_path, orient='records', indent=2)
        created_files.append(sig_path)
        
        # Save all pathways (full results)
        if format == "tsv":
            full_path = results_dir / f"{db_name}_enrichment_full.tsv"
            df.to_csv(full_path, sep='\t', index=False)
        elif format == "csv":
            full_path = results_dir / f"{db_name}_enrichment_full.csv"
            df.to_csv(full_path, index=False)
        else:
            full_path = results_dir / f"{db_name}_enrichment_full.json"
            df.to_json(full_path, orient='records', indent=2)
        created_files.append(full_path)
        
        # Save gene-pathway mapping
        gene_mapping = []
        for pathway in db_result.pathways:
            for gene in pathway.overlapping_genes:
                gene_mapping.append({
                    "gene": gene,
                    "pathway_id": pathway.pathway_id,
                    "pathway_name": pathway.pathway_name,
                    "p_value": pathway.p_value,
                    "adjusted_p_value": pathway.adjusted_p_value
                })
        
        if gene_mapping:
            gene_df = pd.DataFrame(gene_mapping)
            gene_path = results_dir / f"{db_name}_gene_pathway_mapping.tsv"
            gene_df.to_csv(gene_path, sep='\t', index=False)
            created_files.append(gene_path)
        
        return created_files
    
    def save_figures(
        self,
        figures: Dict[str, Any],
        format: str = "svg"
    ) -> List[Path]:
        """
        Save visualization figures.
        
        Args:
            figures: Dictionary mapping figure names to figure objects
            format: Output format ('svg', 'pdf', 'png')
            
        Returns:
            List of created file paths
        """
        created_files = []
        figures_dir = self.directories["figures"]
        
        for fig_name, fig in figures.items():
            fig_path = figures_dir / f"{fig_name}.{format}"
            
            # Save based on figure type
            if hasattr(fig, 'write_image'):
                # Plotly figure
                fig.write_image(str(fig_path))
            elif hasattr(fig, 'savefig'):
                # Matplotlib figure
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            
            created_files.append(fig_path)
        
        return created_files
    
    def save_manifest(self, manifest: AnalysisManifest) -> Path:
        """
        Save analysis manifest.
        
        Args:
            manifest: AnalysisManifest to save
            
        Returns:
            Path to saved manifest
        """
        return save_manifest(manifest, self.output_dir)
    
    def generate_methods_text(
        self,
        analysis_type: str,
        parameters: Dict[str, Any],
        databases: List[str]
    ) -> str:
        """
        Generate methods section text for publication.
        
        Args:
            analysis_type: Type of analysis performed
            parameters: Analysis parameters
            databases: List of databases used
            
        Returns:
            Formatted methods text
        """
        methods = []
        
        methods.append("## Pathway Enrichment Analysis\n")
        
        # Analysis method
        if analysis_type == "ora":
            methods.append(
                "Over-representation analysis (ORA) was performed using PathwayLens "
                f"(version {parameters.get('pathwaylens_version', 'X.X.X')}). "
                "Enrichment significance was assessed using the hypergeometric test, "
                "with odds ratios and 95% confidence intervals calculated using the "
                "Wilson score method. "
            )
        elif analysis_type == "gsea":
            methods.append(
                "Gene Set Enrichment Analysis (GSEA) was performed using PathwayLens "
                f"(version {parameters.get('pathwaylens_version', 'X.X.X')}). "
                f"Enrichment scores were calculated with {parameters.get('permutations', 1000)} "
                "permutations for significance testing. "
            )
        
        # Multiple testing correction
        correction = parameters.get('correction_method', 'fdr_bh')
        methods.append(
            f"Multiple testing correction was applied using the {correction.upper()} method. "
            f"Pathways with adjusted p-value < {parameters.get('significance_threshold', 0.05)} "
            "were considered significantly enriched. "
        )
        
        # Databases
        db_str = ", ".join(databases)
        methods.append(
            f"\n\nPathway annotations were obtained from {db_str}. "
        )
        
        # Effect sizes
        methods.append(
            "Effect sizes were calculated as Cohen's h to quantify enrichment magnitude. "
            "Statistical power was estimated post-hoc for significant pathways. "
        )
        
        # Quality control
        methods.append(
            "\n\nQuality control included analysis of p-value distributions, "
            "detection of pathway size bias, and assessment of gene coverage metrics. "
        )
        
        return "".join(methods)
    
    def generate_citations(self, databases: List[str]) -> str:
        """
        Generate BibTeX citations.
        
        Args:
            databases: List of databases used
            
        Returns:
            BibTeX formatted citations
        """
        citations = []
        
        # Database citations
        db_citations = {
            "kegg": """@article{kanehisa2021kegg,
  title={KEGG: integrating viruses and cellular organisms},
  author={Kanehisa, Minoru and Furumichi, Miho and Sato, Yoko and Ishiguro-Watanabe, Mari and Tanabe, Mao},
  journal={Nucleic acids research},
  volume={49},
  number={D1},
  pages={D545--D551},
  year={2021},
  publisher={Oxford University Press}
}""",
            "reactome": """@article{jassal2020reactome,
  title={The reactome pathway knowledgebase},
  author={Jassal, Bijay and Matthews, Lisa and Viteri, Guilherme and Gong, Chuqiao and Lorente, Pascual and Fabregat, Antonio and Sidiropoulos, Konstantinos and Cook, Justin and Gillespie, Marc and Haw, Robin and others},
  journal={Nucleic acids research},
  volume={48},
  number={D1},
  pages={D498--D503},
  year={2020},
  publisher={Oxford University Press}
}""",
            "go": """@article{gene2021gene,
  title={The Gene Ontology resource: enriching a GOld mine},
  author={Gene Ontology Consortium},
  journal={Nucleic acids research},
  volume={49},
  number={D1},
  pages={D325--D334},
  year={2021},
  publisher={Oxford University Press}
}"""
        }
        
        for db in databases:
            db_lower = db.lower()
            if db_lower in db_citations:
                citations.append(db_citations[db_lower])
        
        return "\n\n".join(citations)
    
    def create_readme(
        self,
        analysis_type: str,
        parameters: Dict[str, Any]
    ) -> Path:
        """
        Create README file for output directory.
        
        Args:
            analysis_type: Type of analysis
            parameters: Analysis parameters
            
        Returns:
            Path to README file
        """
        readme_path = self.output_dir / "README.txt"
        
        content = []
        content.append(f"PathwayLens Analysis Results")
        content.append("=" * 50)
        content.append(f"\nAnalysis ID: {self.analysis_id}")
        content.append(f"Analysis Type: {analysis_type}")
        content.append(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("\n\nDirectory Structure:")
        content.append("-" * 50)
        content.append("\nresults/")
        content.append("  - pathway_enrichment.tsv: Significant pathways")
        content.append("  - pathway_enrichment_full.tsv: All tested pathways")
        content.append("  - gene_pathway_mapping.tsv: Gene-to-pathway mapping")
        content.append("  - summary_statistics.json: Analysis summary")
        content.append("\nfigures/")
        content.append("  - Visualization plots (SVG/PDF format)")
        content.append("\ndata/")
        content.append("  - normalized/: Normalized input data")
        content.append("  - database_metadata/: Database version information")
        content.append("\nmethods/")
        content.append("  - analysis_methods.txt: Methods section text")
        content.append("  - citations.bib: BibTeX citations")
        content.append("\ndiagnostics/")
        content.append("  - Quality control and diagnostic plots")
        content.append("\nmanifest.json")
        content.append("  - Complete analysis provenance for reproducibility")
        
        with open(readme_path, 'w') as f:
            f.write("\n".join(content))
        
        return readme_path
    
    def save_methods_and_citations(
        self,
        analysis_type: str,
        parameters: Dict[str, Any],
        databases: List[str]
    ) -> Dict[str, Path]:
        """
        Save methods text and citations.
        
        Args:
            analysis_type: Type of analysis
            parameters: Analysis parameters
            databases: List of databases used
            
        Returns:
            Dictionary with paths to created files
        """
        methods_dir = self.directories["methods"]
        
        # Generate and save methods text
        methods_text = self.generate_methods_text(analysis_type, parameters, databases)
        methods_path = methods_dir / "analysis_methods.txt"
        with open(methods_path, 'w') as f:
            f.write(methods_text)
        
        # Generate and save citations
        citations = self.generate_citations(databases)
        citations_path = methods_dir / "citations.bib"
        with open(citations_path, 'w') as f:
            f.write(citations)
        
        return {
            "methods": methods_path,
            "citations": citations_path
        }
