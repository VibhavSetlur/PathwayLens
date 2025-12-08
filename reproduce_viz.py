import asyncio
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from pathwaylens_core.visualization.engine import VisualizationEngine
from pathwaylens_core.visualization.schemas import (
    VisualizationParameters, PlotType, VisualizationResult
)
from pathwaylens_core.analysis.schemas import (
    AnalysisResult, PathwayResult, DatabaseResult, DatabaseType, AnalysisType,
    AnalysisParameters, CorrectionMethod, ConsensusMethod
)
from pathwaylens_core.types import OmicType, DataType

async def run_reproduction():
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"tests/outputs/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating visualizations in: {output_dir}")

    # Create sample data
    pathway_results = [
        PathwayResult(
            pathway_id="PATH:00010",
            pathway_name="Glycolysis / Gluconeogenesis",
            database=DatabaseType.KEGG,
            p_value=0.0001,
            adjusted_p_value=0.001,
            enrichment_score=2.5,
            overlapping_genes=["HK1", "GPI", "PFK", "ALDO", "GAPDH"],
            overlap_count=5,
            pathway_count=50,
            input_count=1000,
            analysis_method="ORA"
        ),
        PathwayResult(
            pathway_id="PATH:00020",
            pathway_name="Citrate cycle (TCA cycle)",
            database=DatabaseType.KEGG,
            p_value=0.005,
            adjusted_p_value=0.02,
            enrichment_score=1.8,
            overlapping_genes=["CS", "ACO", "IDH", "OGDH"],
            overlap_count=4,
            pathway_count=40,
            input_count=1000,
            analysis_method="ORA"
        ),
        PathwayResult(
            pathway_id="R-HSA-12345",
            pathway_name="Apoptosis",
            database=DatabaseType.REACTOME,
            p_value=0.00001,
            adjusted_p_value=0.0005,
            enrichment_score=3.0,
            overlapping_genes=["CASP3", "CASP8", "BAX", "BCL2", "CYCS", "FAS"],
            overlap_count=6,
            pathway_count=60,
            input_count=1000,
            analysis_method="ORA"
        )
    ]

    analysis_result = AnalysisResult(
        job_id="repro_job",
        analysis_id="analysis_repro",
        analysis_name="Reproduction Analysis",
        analysis_type=AnalysisType.ORA,
        parameters=AnalysisParameters(
            analysis_type=AnalysisType.ORA,
            databases=[DatabaseType.KEGG, DatabaseType.REACTOME],
            species="human",
            omic_type=OmicType.TRANSCRIPTOMICS,
            data_type=DataType.BULK,
            significance_threshold=0.05,
            correction_method=CorrectionMethod.FDR_BH,
            min_pathway_size=5,
            max_pathway_size=500,
            consensus_method=ConsensusMethod.STOUFFER
        ),
        input_file="test_input.txt",
        input_gene_count=1000,
        input_species="human",
        database_results={
            "KEGG": DatabaseResult(
                database=DatabaseType.KEGG,
                total_pathways=200,
                significant_pathways=2,
                pathways=[p for p in pathway_results if p.database == DatabaseType.KEGG],
                species="human",
                coverage=0.5,
                redundancy=0.1
            ),
            "REACTOME": DatabaseResult(
                database=DatabaseType.REACTOME,
                total_pathways=300,
                significant_pathways=1,
                pathways=[p for p in pathway_results if p.database == DatabaseType.REACTOME],
                species="human",
                coverage=0.6,
                redundancy=0.2
            )
        },
        total_pathways=500,
        significant_pathways=3,
        significant_databases=2,
        overall_quality=0.95,
        reproducibility=0.98,
        created_at=datetime.now().isoformat(),
        completed_at=datetime.now().isoformat(),
        processing_time=1.5
    )

    # Initialize engine
    engine = VisualizationEngine()

    # Define plot types to generate
    plot_types = [
        PlotType.DOT_PLOT,
        PlotType.BAR_CHART,
        PlotType.VOLCANO_PLOT,
        PlotType.HEATMAP,
        PlotType.SCATTER_PLOT
    ]

    # Create visualization parameters
    parameters = VisualizationParameters(
        plot_types=[
            PlotType.DOT_PLOT,
            PlotType.BAR_CHART,
            PlotType.VOLCANO_PLOT,
            PlotType.HEATMAP,
            PlotType.SCATTER_PLOT
        ],
        interactive=True,
        output_formats=["html", "svg"],  # Generate both HTML and SVG
        theme="publication",
        figure_size=[1000, 800],
        dpi=300
    )
    
    # Generate visualizations
    print(f"Generating visualizations in: {output_dir}")
    print("Generating visualizations...")
    try:
        result = await engine.visualize(
            data=analysis_result,
            parameters=parameters,
            output_dir=str(output_dir)
        )
        
        print(f"Generated {result.total_plots} plots.")
        for plot_type, plot_file in result.generated_plots.items():
            print(f"- {plot_type.value}: {plot_file}")
            # Check for SVG existence
            svg_file = str(plot_file).replace(".html", ".svg")
            if Path(svg_file).exists():
                print(f"  (SVG generated: {svg_file})")
            
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_reproduction())
