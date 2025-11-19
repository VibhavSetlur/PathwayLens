# PathwayLens - Research-Grade Pathway Enrichment Analysis

Professional computational biology tool for pathway enrichment analysis across bulk RNA-seq, single-cell RNA-seq, ATAC-seq, proteomics, and multi-omics datasets.

## Installation

```bash
pip install pathwaylens
```

See [INSTALL.md](INSTALL.md) for detailed instructions.

## Quick Start

### Analyze with Automatic Visualizations

```bash
# Over-representation analysis
pathwaylens analyze ora \
  --input deseq2_results.csv \
  --databases kegg reactome go \
  --species human \
  --output-dir results/
```

All visualizations generated automatically in database-specific subdirectories.

### Compare Multiple Datasets

```bash
# Intelligent comparison with type-aware processing
pathwaylens compare \
  --inputs ctrl.csv treat1.csv treat2.csv \
  --labels Control Treatment_A Treatment_B \
  --type rnaseq \
  --databases kegg \
  --species human \
  --output-dir comparison/
```

Automatically performs: normalization → DE analysis → enrichment → RRHO2/UpSet → visualizations

### Visualize Experimental Data

```bash
# Create publication-quality figures from raw data
pathwaylens visualize \
  --input deseq2_results.csv \
  --type deseq \
  --species human \
  --output-dir figures/
```

Produces MA plots, volcano plots, PCA, heatmaps automatically.

## Core Commands

- `analyze` - Pathway enrichment (ORA/GSEA) with automatic visualization
- `compare` - Type-aware multi-dataset comparison (RRHO2, UpSet, concordance)
- `visualize` - Publication-quality figures from experimental data
- `normalize` - Gene ID conversion and normalization

## Supported Data Types

| Type | Input | Auto-Processing | Visualizations |
|------|-------|----------------|----------------|
| Bulk RNA-seq | DESeq2, edgeR, limma | ID normalization, filtering | MA, volcano, PCA, heatmap |
| scRNA-seq | Seurat, Scanpy | Pseudo-bulk, clustering | UMAP, feature, violin, dotplot |
| Raw counts | Count matrices | Normalization, DE, enrichment | PCA, correlation, dispersion |
| ATAC-seq | Peak genes | Gene extraction | Genomic distribution |
| Proteomics | MaxQuant, Perseus | Protein ID mapping | Abundance, volcano |

## Output Structure

```
results/
├── kegg/
│   ├── enrichment.tsv
│   ├── gene_pathway_mapping.tsv
│   └── figures/
│       ├── barplot.svg
│       ├── volcano.svg
│       ├── network.svg
│       └── heatmap.svg
├── reactome/
│   └── ...
└── go/
    └── ...
```

## Key Features

**Automatic Processing**
- Intelligent format detection
- Auto ID normalization
- Type-aware preprocessing

**Comprehensive Statistics**
- Odds ratios with 95% CI
- Effect sizes (Cohen's h)
- Statistical power
- P-value diagnostics

**Publication Quality**
- All visualizations auto-generated
- Colorblind-safe palettes
- 300+ DPI resolution
- SVG/PDF formats

**Type-Aware Comparison**
- RRHO2 for RNA-seq
- UpSet for gene overlaps
- Concordance analysis
- Metadata integration

## Example Workflows

### RNA-seq Analysis
```bash
# Full pipeline: normalize → analyze → visualize
pathwaylens analyze ora \
  --input raw_counts.csv \
  --databases kegg reactome \
  --species human
```

### Multi-Condition Comparison
```bash
# Compare with RRHO2 and UpSet plots
pathwaylens compare \
  --inputs cond1.csv cond2.csv cond3.csv \
  --labels A B C \
  --type deseq \
  --databases kegg \
  --species human
```

### Seurat Object Visualization
```bash
# All scRNA-seq visualizations
pathwaylens visualize \
  --input seurat_object.rds \
  --type seurat \
  --species human
```

## Documentation

- **Installation**: [INSTALL.md](INSTALL.md)
- **CLI Reference**: [docs/user_guide/CLI_REFERENCE.md](docs/user_guide/CLI_REFERENCE.md)
- **Input Formats**: [docs/user_guide/](docs/user_guide/)
- **Examples**: `examples/`

## Validation

```bash
# Verify installation
python scripts/validation/validate_installation.py

# Run tests
pytest tests/
```

## Citation

```bibtex
@software{pathwaylens2025,
  title={PathwayLens: Research-Grade Pathway Enrichment Analysis},
  author={PathwayLens Contributors},
  year={2025},
  version={1.0.0},
  url={https://github.com/VibhavSetlur/PathwayLens}
}
```

## License

MIT License - see [LICENSE](LICENSE)

## Support

- **Issues**: [GitHub Issues](https://github.com/VibhavSetlur/PathwayLens/issues)
- **Documentation**: Complete guides in `docs/`