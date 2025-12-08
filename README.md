# PathwayLens - Research-Grade Pathway Enrichment Analysis

[![CI](https://github.com/VibhavSetlur/PathwayLens/workflows/CI/badge.svg)](https://github.com/VibhavSetlur/PathwayLens/actions)
[![codecov](https://codecov.io/gh/VibhavSetlur/PathwayLens/branch/main/graph/badge.svg)](https://codecov.io/gh/VibhavSetlur/PathwayLens)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Professional computational biology tool for pathway enrichment analysis across bulk RNA-seq, single-cell RNA-seq, ATAC-seq, proteomics, and multi-omics datasets.

## Installation

```bash
pip install pathwaylens
```

See [INSTALL.md](INSTALL.md) for detailed instructions.

## Quick Start

### Basic Usage

**1. Analyze a gene list (ORA):**
```bash
pathwaylens analyze ora \
    --input genes.txt \
    --omic-type transcriptomics \
    --data-type bulk \
    --databases kegg \
    --species human \
    --output-dir results_ora
```

**2. Analyze ranked genes (GSEA):**
```bash
pathwaylens analyze gsea \
    --input ranked_genes.rnk \
    --omic-type transcriptomics \
    --data-type bulk \
    --databases kegg \
    --species human \
    --output-dir results_gsea
```

**3. Compare datasets:**
```bash
pathwaylens compare \
    --inputs list1.txt \
    --inputs list2.txt \
    --mode genes \
    --omic-type transcriptomics \
    --data-type bulk \
    --output-dir comparison_results
```

For detailed documentation, see [CLI Reference](docs/CLI_REFERENCE.md).

## Core Commands

- `analyze ora` - Over-Representation Analysis with hypergeometric test
- `analyze gsea` - Gene Set Enrichment Analysis for ranked lists
- `compare` - Compare multiple gene lists or pathway results
- `normalize` - Gene ID conversion across formats

## Key Features

- **Intelligent Tool Detection**: Automatically detects DESeq2, edgeR, limma, MaxQuant formats
- **Multiple Databases**: KEGG, Reactome, GO, WikiPathways, MSigDB support
- **Research-Grade Statistics**: Odds ratios, effect sizes, confidence intervals
- **Publication-Quality Plots**: Interactive and static visualizations
- **Cross-Species Support**: Human, mouse, rat, and more
- **Single-Cell Ready**: Native sparse matrix support

> [!IMPORTANT]
> **Single-Cell Normalization**: PathwayLens assumes input scRNA-seq data (`.h5ad`, `.csv`) is already normalized for sequencing depth (e.g., LogNormalize, SCTransform). It does **not** perform library size normalization internally. Using raw counts will lead to invalid results.

## Documentation

- **Installation**: [INSTALL.md](INSTALL.md)
- **CLI Reference**: [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md)
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