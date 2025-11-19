# PathwayLens - Research-Grade Pathway Enrichment Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Publication-ready pathway enrichment analysis with comprehensive statistical rigor, complete reproducibility, and automated manuscript generation.**

## 🎯 Key Features

### Research-Grade Statistics
- **Odds Ratios** with 95% Wilson score confidence intervals
- **Effect Sizes** (Cohen's h) for enrichment magnitude
- **Statistical Power** estimation (post-hoc)
- **Fold Enrichment** calculations
- **Diagnostic Plots** for quality control

### Complete Reproducibility
- **Analysis Manifests** capturing all provenance
- **Database Version** tracking with checksums
- **Environment Snapshots** (Python, OS, dependencies)
- **Reproducibility Hashes** for validation
- **Random Seed Management** for deterministic execution

### Publication-Ready Outputs
- **Structured Directories** with organized results
- **Auto-Generated Methods** sections for manuscripts
- **BibTeX Citations** for all databases and tools
- **TSV/CSV Exports** with full statistical metrics
- **SVG/PDF Figures** for publication

### Professional Visualizations
- **Colorblind-Safe Palettes** (Wong, Tol, journal-specific)
- **Diagnostic Plots** (p-value distributions, Q-Q plots, size bias)
- **Publication Themes** (Nature, Science standards)
- **High-Resolution Export** (300+ DPI)

## 📦 Installation

### Quick Install
```bash
pip install pathwaylens
```

### From Source
```bash
git clone https://github.com/yourusername/PathwayLens.git
cd PathwayLens
pip install -e .
```

### Verify Installation
```bash
python validate_installation.py
```

## 🚀 Quick Start

### Basic ORA Analysis
```python
from pathwaylens_core.analysis import ORAEngine
from pathwaylens_core.data import DatabaseManager
from pathwaylens_core.analysis.schemas import DatabaseType

# Initialize
db_manager = DatabaseManager()
ora_engine = ORAEngine(db_manager)

# Run analysis
result = await ora_engine.analyze(
    gene_list=["BRCA1", "TP53", "EGFR", "MYC"],
    database=DatabaseType.KEGG,
    species="human",
    significance_threshold=0.05
)

# Results include comprehensive statistics
for pathway in result.pathways[:5]:
    print(f"{pathway.pathway_name}:")
    print(f"  P-value: {pathway.p_value:.2e}")
    print(f"  Odds Ratio: {pathway.odds_ratio:.2f} "
          f"(95% CI: {pathway.odds_ratio_ci_lower:.2f}-{pathway.odds_ratio_ci_upper:.2f})")
    print(f"  Effect Size: {pathway.effect_size:.3f}")
    print(f"  Statistical Power: {pathway.statistical_power:.2f}")
```

### Publication-Ready Workflow
```python
from pathwaylens_core.io import AnalysisOutputManager
from pathwaylens_core.utils import generate_manifest
from pathwaylens_core.visualization import create_diagnostic_panel

# Create structured output
output_mgr = AnalysisOutputManager(
    base_dir="./results",
    analysis_id="study_001"
)
output_mgr.create_directory_structure()

# Save results with all statistics
output_mgr.save_results(result, format="tsv")

# Generate manifest for reproducibility
manifest = generate_manifest(
    analysis_id="study_001",
    analysis_type="ora",
    parameters={"significance_threshold": 0.05},
    input_files=["gene_list.txt"],
    database_versions=db_versions
)
output_mgr.save_manifest(manifest)

# Auto-generate methods section and citations
output_mgr.save_methods_and_citations(
    analysis_type="ora",
    parameters={...},
    databases=["KEGG", "Reactome"]
)

# Create diagnostic plots
diagnostic_fig = create_diagnostic_panel(result)
output_mgr.save_figures({"diagnostics": diagnostic_fig}, format="svg")
```

**Output Directory**:
```
pathway_analysis_20250119_143022/
├── manifest.json                    # Complete provenance
├── README.txt                       # Directory guide
├── results/
│   ├── KEGG_enrichment.tsv         # Significant pathways (with OR, CI, effect sizes)
│   ├── KEGG_enrichment_full.tsv    # All tested pathways
│   ├── KEGG_gene_pathway_mapping.tsv
│   └── summary_statistics.json
├── figures/
│   └── diagnostics.svg             # QC plots
├── methods/
│   ├── analysis_methods.txt        # Ready for manuscript
│   └── citations.bib               # BibTeX citations
└── diagnostics/
```

## 📊 Analysis Types

- **ORA** (Over-Representation Analysis) - Fast, hypergeometric test
- **GSEA** (Gene Set Enrichment Analysis) - Rank-based enrichment
- **Consensus** - Combine multiple methods
- **Comparison** - Compare enrichment across conditions
- **Multi-Omics** - Integrated analysis

## 🎨 Visualization Features

### Colorblind-Safe Palettes
```python
from pathwaylens_core.visualization.palettes import ColorPalette

# Get colorblind-safe palette
colors = ColorPalette.get_colorblind_safe_palette(8)

# Journal-specific palettes
nature_colors = ColorPalette.get_publication_palette("nature")
```

### Diagnostic Plots
```python
from pathwaylens_core.visualization.diagnostic_plots import (
    plot_pvalue_histogram,
    plot_qq_plot,
    create_diagnostic_panel
)

# Individual diagnostic plots
pvalue_fig = plot_pvalue_histogram(pvalues)
qq_fig = plot_qq_plot(pvalues)

# Comprehensive 2x2 diagnostic panel
panel = create_diagnostic_panel(result, output_path="diagnostics.svg")
```

## 🔬 Scientific Rigor

### Statistical Methods
- **Hypergeometric Test** for ORA
- **Wilson Score Method** for confidence intervals
- **Cohen's h** for effect sizes
- **Spearman Correlation** for bias detection
- **Kolmogorov-Smirnov** for p-value distribution testing

### Quality Control
- P-value distribution analysis
- Pathway size bias detection
- Gene coverage metrics
- Multiple testing correction (11 methods available)
- Statistical power estimation

### References
All statistical methods are properly cited with academic references in the auto-generated methods text.

## 📚 Documentation

- **Quick Start Guide**: Get started in 5 minutes
- **API Reference**: Complete function documentation
- **Tutorial Notebooks**: Step-by-step examples
- **Statistical Methods**: Detailed methodology
- **Best Practices**: Publication guidelines

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Validate installation:
```bash
python validate_installation.py
```

## 📋 Requirements

- Python 3.8+
- numpy >= 1.20.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- pydantic >= 2.0.0
- plotly >= 5.15.0
- psutil >= 5.8.0

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📮 Citation

If you use PathwayLens in your research, please cite:

```bibtex
@software{pathwaylens2025,
  title={PathwayLens: Research-Grade Pathway Enrichment Analysis},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/PathwayLens}
}
```

## 🌟 Acknowledgments

- Statistical methods based on established bioinformatics approaches
- Colorblind-safe palettes from Wong (2011) and Tol (2021)
- Database annotations from KEGG, Reactome, GO Consortium

## 📊 Comparison with Other Tools

| Feature | PathwayLens | DAVID | GOseq | clusterProfiler |
|---------|-------------|-------|-------|-----------------|
| Odds Ratios + CI | ✅ | ⚠️ | ⚠️ | ⚠️ |
| Effect Sizes | ✅ | ❌ | ❌ | ❌ |
| Statistical Power | ✅ | ❌ | ❌ | ❌ |
| Complete Provenance | ✅ | ❌ | ❌ | ❌ |
| Auto Methods Text | ✅ | ❌ | ❌ | ❌ |
| Diagnostic Plots | ✅ | ⚠️ | ⚠️ | ✅ |
| Colorblind-Safe | ✅ | ❌ | ❌ | ⚠️ |

## 🔗 Links

- **Documentation**: [docs.pathwaylens.org](https://docs.pathwaylens.org)
- **GitHub**: [github.com/yourusername/PathwayLens](https://github.com/yourusername/PathwayLens)
- **Issues**: [github.com/yourusername/PathwayLens/issues](https://github.com/yourusername/PathwayLens/issues)
- **PyPI**: [pypi.org/project/pathwaylens](https://pypi.org/project/pathwaylens)

---

**Made with ❤️ for researchers, by researchers.**