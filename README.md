# 🧬 PathwayLens 2.0

**Next-generation computational biology platform for multi-omics pathway analysis**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/pathwaylens.svg)](https://badge.fury.io/py/pathwaylens)

## 🚀 Quick Start

### 📦 Easy Installation

```bash
# Install core features (lightweight, ~50MB)
pip install pathwaylens

# Install all features (full functionality, ~500MB)
pip install pathwaylens[all]

# Install specific features
pip install pathwaylens[analysis]  # Core analysis tools
pip install pathwaylens[viz]       # Visualization tools
pip install pathwaylens[pathways]  # Pathway databases
pip install pathwaylens[api]       # Web API
```

### 🔧 Basic Usage

```bash
# Check installation
pathwaylens --help

# Normalize gene IDs
pathwaylens normalize gene-ids --input genes.csv --species human

# Perform pathway analysis
pathwaylens analyze enrichment --input deseq2_results.csv --databases kegg,reactome

# Compare datasets
pathwaylens compare --input dataset1.csv dataset2.csv --species human
```

## 📦 Installation Options

### 🎯 Minimal Installation (Recommended for CLI users)
```bash
pip install pathwaylens
```
**Includes:** Basic CLI, gene ID normalization, simple analysis tools

### 🔬 Full Installation (Recommended for researchers)
```bash
pip install pathwaylens[all]
```
**Includes:** All analysis tools, visualization, pathway databases, web API

### 🔧 Feature-Specific Installation
```bash
pip install pathwaylens[analysis,viz,pathways]
```
**Includes:** Analysis tools, visualization, and pathway databases

## 🧬 Features

### ✅ Core Features (Always Included)
- **Gene ID Normalization**: Convert between Entrez, Ensembl, Symbol, UniProt
- **Basic Analysis**: Simple pathway enrichment and statistical tests
- **Cross-Platform CLI**: Works on Windows, macOS, and Linux
- **Rich Interface**: Beautiful command-line interface with progress bars

### 🔬 Extended Features (Optional)
- **Advanced Analysis**: Multi-omics integration, topology analysis, GSVA
- **Visualization**: Interactive plots, network diagrams, dashboards
- **Pathway Databases**: KEGG, Reactome, GO, WikiPathways
- **Web Interface**: Next.js frontend with modern UI
- **Background Jobs**: Celery-based job processing
- **Database Support**: PostgreSQL with full schema

## 🛠️ Development Installation

```bash
# Clone repository
git clone https://github.com/pathwaylens/pathwaylens.git
cd pathwaylens

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run system test
python test_system.py
```

## 🐳 Docker Deployment

```bash
# Quick start with Docker Compose
docker compose -f infra/docker/docker-compose.yml up -d

# Access web interface
open http://localhost:3000
```

## 📚 Documentation

- **User Guide**: [docs/user-guide.md](docs/user-guide.md)
- **API Documentation**: [docs/api.md](docs/api.md)
- **Deployment Guide**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Blueprint**: [PATHWAYLENS_2.0_UPGRADE_BLUEPRINT.md](PATHWAYLENS_2.0_UPGRADE_BLUEPRINT.md)

## 🎯 Use Cases

### 🔬 Bioinformatics Researchers
```bash
# Install full suite
pip install pathwaylens[all]

# Analyze RNA-seq data
pathwaylens analyze enrichment --input deseq2_results.csv --databases kegg,reactome

# Generate publication-ready plots
pathwaylens visualize --input analysis_results.json --type enrichment_plot
```

### 📊 Data Scientists
```bash
# Install analysis tools
pip install pathwaylens[analysis,viz]

# Normalize gene identifiers
pathwaylens normalize gene-ids --input genes.csv --species human

# Perform multi-omics analysis
pathwaylens analyze multi-omics --input rna.csv protein.csv --method integration
```

### 💻 Software Developers
```bash
# Install API components
pip install pathwaylens[api,database,jobs]

# Start web API
pathwaylens api start --host 0.0.0.0 --port 8000
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with ❤️ for the bioinformatics community using:
- [Typer](https://typer.tiangolo.com/) for CLI
- [FastAPI](https://fastapi.tiangolo.com/) for web API
- [Next.js](https://nextjs.org/) for frontend
- [Plotly](https://plotly.com/) for visualizations

---

**PathwayLens 2.0** - Making pathway analysis accessible to everyone! 🧬