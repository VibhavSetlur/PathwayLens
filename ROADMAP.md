# PathwayLens Roadmap

This document outlines the future development goals for PathwayLens. We aim to evolve from a CLI tool into a comprehensive platform for multi-omics pathway analysis.

## Short-Term Goals (v1.1 - v1.2)

- [ ] **Enhanced Single-Cell Support**: Implement true GSVA/ssGSEA with full statistical testing.
- [ ] **Cloud Database Integration**: Cache pathway databases locally to reduce API dependency.
- [ ] **Interactive Visualizations**: Add more interactive plots (e.g., Plotly-based volcano plots, heatmaps) to the HTML report.

## Mid-Term Goals (v2.0)

- [ ] **Pathway Topology Analysis**: Move beyond gene sets to true topology-based analysis (e.g., SPIA, Impact Analysis) that considers gene interactions and directionality.
- [ ] **Native Bioconductor Integration**: Seamlessly read/write Bioconductor objects (`SingleCellExperiment`, `DESeqDataSet`) via `rpy2`.
- [ ] **Multi-Omics Integration**: True multi-omics pathway scoring (e.g., combining RNA-seq and Proteomics data).


