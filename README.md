# PathwayLens

Research-grade pathway enrichment analysis for bulk RNA-seq, single-cell RNA-seq, ATAC-seq, proteomics, and multi-omics datasets.

## Installation

```bash
pip install pathwaylens
```

See [INSTALL.md](INSTALL.md) for detailed instructions.

## Quick Start

### Command-Line Interface

```bash
# Get help
pathwaylens --help

# Normalize gene identifiers
pathwaylens normalize genes.txt --species human --target-type symbol

# Over-representation analysis
pathwaylens analyze ora \
  --input gene_list.txt \
  --databases kegg reactome \
  --species human \
  --output results/

# Gene set enrichment analysis
pathwaylens analyze gsea \
  --input ranked_genes.txt \
  --databases go \
  --species human

# Compare datasets
pathwaylens compare dataset1.csv dataset2.csv \
  --species human \
  --output comparison/
```

### Python API

```python
import asyncio
from pathwaylens_core.analysis import ORAEngine
from pathwaylens_core.data import DatabaseManager
from pathwaylens_core.analysis.schemas import DatabaseType

# Initialize
db_manager = DatabaseManager()
ora = ORAEngine(db_manager)

# Run analysis
async def analyze():
    result = await ora.analyze(
        gene_list=["BRCA1", "TP53", "EGFR", "MYC"],
        database=DatabaseType.KEGG,
        species="human"
    )
    
    for pathway in result.pathways[:5]:
        print(f"{pathway.pathway_name}:")
        print(f"  P-value: {pathway.p_value:.2e}")
        print(f"  Odds Ratio: {pathway.odds_ratio:.2f}")
        print(f"  95% CI: [{pathway.odds_ratio_ci_lower:.2f}, "
              f"{pathway.odds_ratio_ci_upper:.2f}]")

asyncio.run(analyze())
```

## Supported Data Types

PathwayLens handles diverse genomics and proteomics datasets:

| Data Type | Input Format | Analysis Type |
|-----------|--------------|---------------|
| Bulk RNA-seq | DESeq2, edgeR, limma output | ORA, GSEA |
| Single-cell RNA-seq | Seurat, Scanpy markers | ORA, GSEA |
| ATAC-seq | Peak-associated genes | ORA |
| Proteomics | Differential protein lists | ORA, GSEA |
| Multi-omics | Integrated gene lists | ORA, GSEA, Comparison |
| Gene lists | TXT, CSV, TSV | All methods |

**Supported File Formats**:
- CSV, TSV, TXT (gene lists)
- AnnData (.h5ad) for single-cell
- 10X Genomics outputs
- GMT/GCT pathway databases
- DESeq2/edgeR/limma results tables

## Key Features

**Research-Grade Statistics**
- Odds ratios with 95% confidence intervals (Wilson score method)
- Effect sizes (Cohen's h)
- Fold enrichment ratios
- Statistical power estimation
- P-value distribution diagnostics

**Complete Reproducibility**
- Analysis manifests with environment snapshots
- Database version tracking with checksums
- Input file verification (MD5)
- Reproducibility hashes (SHA256)

**Publication-Ready Outputs**
- Structured directory organization
- Auto-generated methods sections
- BibTeX citations
- High-resolution figures (SVG/PDF, 300+ DPI)
- Colorblind-safe palettes

## Analysis Methods

- **ORA** (Over-Representation Analysis): Fast hypergeometric enrichment
- **GSEA** (Gene Set Enrichment Analysis): Rank-based enrichment
- **Consensus**: Combine multiple methods
- **Comparison**: Compare enrichment across conditions
- **Multi-Omics**: Integrated pathway analysis

## Available Databases

- KEGG: Metabolic and signaling pathways
- Reactome: Biological reactions
- GO: Gene Ontology (BP, MF, CC)
- WikiPathways: Community pathways
- BioCarta: Disease pathways

## Output Structure

```
pathway_analysis_<timestamp>/
├── manifest.json
├── results/
│   ├── <database>_enrichment.tsv
│   ├── gene_pathway_mapping.tsv
│   └── summary_statistics.json
├── figures/
│   └── diagnostic_plots.svg
├── methods/
│   ├── analysis_methods.txt
│   └── citations.bib
└── README.txt
```

## Documentation

- Installation Guide: [INSTALL.md](INSTALL.md)
- Contributing Guidelines: [CONTRIBUTING.md](CONTRIBUTING.md)
- Example Scripts: `examples/`
- Sample Data: `data/examples/`

## Validation

```bash
# Validate installation
python scripts/validation/validate_installation.py

# Run test suite
pytest tests/
```

## Citation

```bibtex
@software{pathwaylens2025,
  title={PathwayLens: Research-Grade Pathway Enrichment Analysis},
  author={PathwayLens Contributors},
  year={2025},
  url={https://github.com/VibhavSetlur/PathwayLens}
}
```

## License

MIT License - see [LICENSE](LICENSE)

## Support

- GitHub Issues: [Report bugs](https://github.com/VibhavSetlur/PathwayLens/issues)
- Documentation: Complete guides in `docs/`