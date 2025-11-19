"""
Basic ORA Analysis Example

This example demonstrates a simple pathway enrichment analysis
with all new research-grade features.
"""

import asyncio
from pathways_core.analysis import ORAEngine
from pathwaylens_core.data import DatabaseManager
from pathwaylens_core.analysis.schemas import DatabaseType
from pathwaylens_core.io import AnalysisOutputManager
from pathwaylens_core.utils.manifest_generator import generate_manifest
from pathwaylens_core.visualization.diagnostic_plots import create_diagnostic_panel


async def run_basic_ora_analysis():
    """Run a basic ORA analysis with publication-ready output."""
    
    # Sample gene list (cancer-related genes)
    gene_list = [
        "BRCA1", "BRCA2", "TP53", "EGFR", "MYC",
        "AKT1", "PTEN", "RB1", "KRAS", "PIK3CA",
        "ERBB2", "MDM2", "CDKN2A", "ATM", "CHEK2"
    ]
    
    print("=" * 60)
    print("PathwayLens Basic ORA Analysis Example")
    print("=" * 60)
    print(f"\nAnalyzing {len(gene_list)} genes...")
    
    # Initialize database manager and ORA engine
    db_manager = DatabaseManager()
    ora_engine = ORAEngine(db_manager)
    
    # Run ORA analysis
    print("\nRunning ORA analysis...")
    result = await ora_engine.analyze(
        gene_list=gene_list,
        database=DatabaseType.GO,  # Using Gene Ontology
        species="human",
        significance_threshold=0.05
    )
    
    # Display results with new statistical metrics
    print(f"\n✓ Analysis complete!")
    print(f"  Total pathways tested: {result.total_pathways}")
    print(f"  Significant pathways: {result.significant_pathways}")
    print(f"  Gene coverage: {result.coverage:.1%}")
    
    # Show top 5 significant pathways with comprehensive statistics
    print("\nTop 5 Significant Pathways:")
    print("-" * 60)
    
    for i, pathway in enumerate(result.pathways[:5], 1):
        if pathway.adjusted_p_value < 0.05:
            print(f"\n{i}. {pathway.pathway_name}")
            print(f"   P-value: {pathway.p_value:.2e}")
            print(f"   Adj. P-value: {pathway.adjusted_p_value:.2e}")
            print(f"   Odds Ratio: {pathway.odds_ratio:.2f} "
                  f"(95% CI: {pathway.odds_ratio_ci_lower:.2f}-{pathway.odds_ratio_ci_upper:.2f})")
            print(f"   Fold Enrichment: {pathway.fold_enrichment:.2f}x")
            print(f"   Effect Size (Cohen's h): {pathway.effect_size:.3f}")
            print(f"   Statistical Power: {pathway.statistical_power:.2f}")
            print(f"   Overlapping Genes: {pathway.overlap_count}/{pathway.input_count}")
            print(f"   Expected by Chance: {pathway.genes_expected:.1f}")
    
    # Create structured output directory
    print("\n\nCreating publication-ready output...")
    output_mgr = AnalysisOutputManager(
        base_dir="./example_results",
        analysis_id="example_ora_001"
    )
    
    # Create directory structure
    dirs = output_mgr.create_directory_structure()
    print(f"✓ Created output directory: {output_mgr.output_dir}")
    
    # Save results in TSV format
    result_files = output_mgr.save_results(result, format="tsv")
    print(f"✓ Saved {len(result_files)} result files")
    
    # Generate and save manifest for reproducibility
    manifest = generate_manifest(
        analysis_id="example_ora_001",
        analysis_type="ora",
        parameters={
            "database": "GO",
            "species": "human",
            "significance_threshold": 0.05,
            "correction_method": "fdr_bh"
        },
        input_files=[],  # Would include actual input files
        database_versions={},  # Would include actual DB versions
        random_seed=None
    )
    manifest_path = output_mgr.save_manifest(manifest)
    print(f"✓ Saved manifest: {manifest_path.name}")
    
    # Generate methods text and citations
    methods_files = output_mgr.save_methods_and_citations(
        analysis_type="ora",
        parameters={
            "pathwaylens_version": "1.0.0",
            "correction_method": "fdr_bh",
            "significance_threshold": 0.05
        },
        databases=["GO"]
    )
    print(f"✓ Generated methods text and citations")
    
    # Create diagnostic plots
    print("\nGenerating diagnostic plots...")
    diagnostic_fig = create_diagnostic_panel(result)
    fig_files = output_mgr.save_figures(
        {"diagnostic_panel": diagnostic_fig},
        format="svg"
    )
    print(f"✓ Saved {len(fig_files)} diagnostic plots")
    
    # Create README
    readme_path = output_mgr.create_readme(
        analysis_type="ora",
        parameters={"database": "GO", "species": "human"}
    )
    print(f"✓ Created README: {readme_path.name}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_mgr.output_dir}")
    print("\nOutput includes:")
    print("  - Enrichment results (TSV)")
    print("  - Analysis manifest (JSON)")
    print("  - Methods text (ready for manuscript)")
    print("  - BibTeX citations")
    print("  - Diagnostic plots (SVG)")
    print("  - README documentation")
    print("\n✅ Ready for publication!")


if __name__ == "__main__":
    # Run the async analysis
    asyncio.run(run_basic_ora_analysis())
