# PathwayLens 2.0: Next-Generation Computational Biology Platform
## Comprehensive Upgrade Blueprint

### Executive Summary

This document outlines the complete transformation of PathwayLens from a CLI-only tool to a next-generation, robust, and scalable computational biology platform. The upgrade addresses backend design, computational biology workflows, modular architecture, future-proofing, edge cases, and automation while maintaining backward compatibility.

---

## 1. Current State Analysis

### 1.1 Existing Capabilities
- **CLI Tool**: `python3 -m pathwaylens.cli` with subcommands `normalize`, `analyze`, `compare`
- **Gene Identifier Conversion**: Ensembl, NCBI, MyGene for different species
- **Pathway Analysis**: ORA/GSEA across KEGG, Reactome, Gene Ontology
- **Multi-Dataset Comparison**: Pairwise dataset comparison for overlap/consistency
- **Intelligent Visualizations**: Automatic plot type detection via R integration
- **Structured Output**: Cross-species compatibility (human, mouse, model organisms)
- **Project Structure**: Python package with CLI, tests, and documentation

### 1.2 CLI Enhancement Goals
- **Standalone Executable**: Transform from `python3 -m pathwaylens.cli` to direct `pathwaylens` command
- **Unix-like Interface**: Commands like `pathwaylens normalize`, `pathwaylens analyze`, `pathwaylens compare`
- **Cross-Platform**: Compatible across Windows, macOS, and Linux
- **Easy Installation**: `pip install pathwaylens` adds command to system PATH

### 1.3 Key Limitations & Opportunities
- **CLI Only**: No web UI or interactive dashboard
- **Limited Comparison**: Only pairwise dataset comparison
- **Static Visualizations**: R-based, potentially non-interactive
- **Limited Multi-Omics**: Primarily gene lists/expression matrices
- **No Reproducibility**: Missing pipelines, job tracking, collaboration features
- **Monolithic Structure**: Lacks modular architecture for scalability

---

## 2. Vision for PathwayLens 2.0

### 2.1 Core Goals
- **Dual Interface**: Zero-code web UI + standalone CLI for power users
- **Standalone CLI**: Direct `pathwaylens` command with subcommands (normalize, analyze, compare, visualize)
- **Multi-Omics Support**: Genomics, transcriptomics, proteomics, metabolomics, phosphoproteomics, epigenomics
- **Advanced Workflows**: Batch processing, multi-dataset comparison, cross-species analysis
- **Interactive Visualizations**: Fully interactive, linked dashboards with publication-ready exports
- **Reproducibility**: Job tracking, configuration files, versioning, API integration
- **Scalability**: Job queues, parallelism, data caching, cloud/cluster readiness
- **Modular Architecture**: Plugin system, AI modules, custom analyses
- **Design System**: Globally scoped, themeable, maintainable frontend

---

## 3. Detailed Backend Architecture

### 3.1 Core Modules

#### 3.1.1 Data Ingestion & Normalization Module
**Location**: `pathwaylens_core/normalization/`

**Responsibilities**:
- Accept varied input formats: gene lists, DE tables, expression matrices, proteomics/metabolomics tables
- Validate input: detect format (CSV, TSV, Excel, HDF5), check required columns
- Identifier conversion: map input IDs to canonical format using MyGene.info, UniProt REST, Ensembl
- Cross-species mapping: mouse→human orthologs via Ensembl/OrthoDB
- Preview mapping outcomes: unmapped, duplicates, ambiguous IDs
- Standardized representation: `NormalizedTable` object with schema
- Logging/tracking: job_id, user, timestamp, conversion stats
- Edge cases: missing values, duplicate IDs, multi-mapping, large datasets

**Key Files**:
```
normalization/
├── __init__.py
├── normalizer.py              # Main normalization engine
├── format_detector.py         # Input format detection
├── id_converter.py           # Gene ID conversion logic
├── species_mapper.py         # Cross-species mapping
├── validation.py             # Input validation
├── schemas.py                # Pydantic schemas
└── utils.py                  # Helper functions
```

#### 3.1.2 Analysis Engine Module
**Location**: `pathwaylens_core/analysis/`

**Responsibilities**:
- Multiple analysis types: ORA, GSEA, GSVA, pathway topology methods
- Unified database interface: KEGG, Reactome, GO, BioCyc, Pathway Commons
- Parameterization: species, thresholds, corrections, gene set limits
- Multi-dataset mode: joint enrichment, differential enrichment, pathway concordance
- Output: `AnalysisResult` object with standardized schema
- Logging/provenance: input versions, code versions, database versions
- Performance: streaming, parallel execution, caching
- Plugins: custom analysis modules via plugin interface

**Key Files**:
```
analysis/
├── __init__.py
├── engine.py                 # Main analysis engine
├── ora_engine.py            # Over-representation analysis
├── gsea_engine.py           # Gene set enrichment analysis
├── gsva_engine.py           # Gene set variation analysis
├── topology_engine.py       # Pathway topology methods
├── multi_omics_engine.py    # Multi-omics integration
├── consensus_engine.py      # Consensus analysis
├── schemas.py               # Analysis result schemas
└── utils.py                 # Statistical utilities
```

#### 3.1.3 Visualization & Reporting Module
**Location**: `pathwaylens_core/visualization/`

**Responsibilities**:
- Multiple visual outputs: volcano plots, dot plots, networks, heatmaps, PCA/UMAP
- Interactive dashboards: filtering, zooming, tooltips, linked views
- Export reports: HTML, Markdown, Jupyter notebooks, configuration files
- Customization: color schemes, labels, annotations, publication-ready figures
- Performance: pagination, lazy loading, efficient rendering
- Theme support: dark/light theme, accessibility, colorblind-safe palettes

**Key Files**:
```
visualization/
├── __init__.py
├── engine.py                # Main visualization engine
├── plotly_renderer.py       # Interactive Plotly visualizations
├── static_renderer.py       # Static matplotlib/seaborn plots
├── network_renderer.py      # Network visualizations
├── dashboard_builder.py     # Interactive dashboard creation
├── export_manager.py        # Report export functionality
├── themes.py                # Theme management
└── utils.py                 # Visualization utilities
```

#### 3.1.4 Comparison & Multi-Dataset Module
**Location**: `pathwaylens_core/comparison/`

**Responsibilities**:
- Multi-dataset analysis: overlap statistics, enrichment profile correlation
- Clustering: hierarchical clustering, heatmaps based on enrichment signatures
- Visualization: Venn/Euler diagrams, upset plots, pathway concordance plots
- Parameters: pathway inclusion, thresholds, normalization across datasets
- Output: `ComparisonResult` with comprehensive schema
- Edge cases: species/feature type harmonization, missing pathways

**Key Files**:
```
comparison/
├── __init__.py
├── engine.py                # Main comparison engine
├── overlap_analyzer.py      # Gene/pathway overlap analysis
├── correlation_analyzer.py  # Enrichment profile correlation
├── clustering_analyzer.py   # Dataset clustering
├── visualization.py         # Comparison visualizations
├── schemas.py               # Comparison result schemas
└── utils.py                 # Comparison utilities
```

#### 3.1.5 Backend API & Job Management Module
**Location**: `pathwaylens_api/`

**Responsibilities**:
- FastAPI endpoints: `/upload`, `/normalize`, `/analyze`, `/compare`, `/results/{job_id}`
- Authentication/authorization: JWT tokens, OAuth2, role-based access
- Job queue: Celery + Redis for asynchronous execution
- Metadata store: PostgreSQL for job metadata and results
- Storage: cloud storage or local filesystem with versioning
- API documentation: OpenAPI/Swagger UI
- Rate limiting/quotas: resource management for shared servers

**Key Files**:
```
api/
├── __init__.py
├── main.py                  # FastAPI application
├── routes/
│   ├── __init__.py
│   ├── upload.py            # File upload endpoints
│   ├── normalize.py         # Normalization endpoints
│   ├── analyze.py           # Analysis endpoints
│   ├── compare.py           # Comparison endpoints
│   ├── results.py           # Results retrieval
│   └── jobs.py              # Job management
├── schemas/
│   ├── __init__.py
│   ├── job_schema.py        # Job-related schemas
│   ├── analysis_schema.py   # Analysis schemas
│   └── comparison_schema.py # Comparison schemas
├── middleware/
│   ├── __init__.py
│   ├── auth.py              # Authentication middleware
│   ├── rate_limit.py        # Rate limiting
│   └── logging.py           # Request logging
└── utils/
    ├── __init__.py
    ├── database.py          # Database utilities
    └── storage.py           # Storage utilities
```

#### 3.1.6 Data/Database Module
**Location**: `pathwaylens_core/data/`

**Responsibilities**:
- Database adapters: KEGG, Reactome, GO, BioCyc, Pathway Commons
- Local cache: SQLite/on-disk store with versioning
- Annotation mapping: gene↔pathway, gene-set definitions
- Ortholog mapping: cross-species via Ensembl Compara
- Custom gene sets: user-uploaded GMT files
- Update mechanism: refresh databases while maintaining versions

**Key Files**:
```
data/
├── __init__.py
├── adapters/
│   ├── __init__.py
│   ├── kegg_adapter.py      # KEGG database adapter
│   ├── reactome_adapter.py  # Reactome adapter
│   ├── go_adapter.py        # Gene Ontology adapter
│   ├── biocyc_adapter.py    # BioCyc adapter
│   └── custom_adapter.py    # Custom gene sets
├── mapping/
│   ├── __init__.py
│   ├── gene_mapper.py       # Gene ID mapping
│   ├── ortholog_mapper.py   # Cross-species mapping
│   └── pathway_mapper.py    # Pathway mapping
├── cache/
│   ├── __init__.py
│   ├── cache_manager.py     # Cache management
│   └── version_manager.py   # Version control
└── utils/
    ├── __init__.py
    └── database_utils.py    # Database utilities
```

#### 3.1.7 Standalone CLI Module
**Location**: `pathwaylens_cli/`

**Responsibilities**:
- Standalone executable: Direct `pathwaylens` command without `python3 -m` prefix
- Unix-like interface: Subcommands (normalize, analyze, compare, visualize, config)
- Cross-platform compatibility: Windows, macOS, Linux support
- Entry point management: Automatic PATH addition via pip installation
- Command-line argument parsing: Rich help messages and validation
- Integration with core modules: Seamless access to all backend functionality
- Configuration management: CLI-specific settings and user preferences

**Key Files**:
```
cli/
├── __init__.py
├── main.py                  # Main CLI entry point
├── commands/
│   ├── __init__.py
│   ├── normalize.py         # Gene normalization command
│   ├── analyze.py           # Pathway analysis command
│   ├── compare.py           # Dataset comparison command
│   ├── visualize.py         # Visualization command
│   ├── config.py            # Configuration management
│   └── info.py              # System information
├── utils/
│   ├── __init__.py
│   ├── formatters.py        # Output formatting
│   ├── validators.py        # Input validation
│   ├── progress.py          # Progress indicators
│   └── colors.py            # Terminal colors
├── templates/
│   ├── help_templates/      # Help message templates
│   └── output_templates/    # Output formatting templates
└── config/
    ├── __init__.py
    ├── cli_config.py        # CLI-specific configuration
    └── user_preferences.py  # User preference management
```

#### 3.1.8 Infrastructure Module
**Location**: `pathwaylens_infra/`

**Responsibilities**:
- Containerization: Docker, docker-compose, Kubernetes manifests
- Scalability: autoscaling, distributed processing
- Logging/monitoring: Prometheus + Grafana integration
- Deployment: CI/CD with GitHub Actions
- Security: secrets management, TLS, CORS, input validation
- Performance: async I/O, caching, efficient data structures

**Key Files**:
```
infra/
├── __init__.py
├── docker/
│   ├── Dockerfile.backend   # Backend container
│   ├── Dockerfile.frontend  # Frontend container
│   └── docker-compose.yml   # Local development
├── kubernetes/
│   ├── backend-deployment.yaml
│   ├── frontend-deployment.yaml
│   ├── redis-deployment.yaml
│   └── postgres-deployment.yaml
├── monitoring/
│   ├── prometheus.yml       # Prometheus config
│   ├── grafana-dashboards/  # Grafana dashboards
│   └── alerting.yml         # Alert rules
├── ci-cd/
│   ├── .github/workflows/
│   │   ├── test.yml         # Test pipeline
│   │   ├── build.yml        # Build pipeline
│   │   └── deploy.yml       # Deployment pipeline
│   └── scripts/
│       ├── setup.sh         # Environment setup
│       └── deploy.sh        # Deployment script
└── config/
    ├── production.yml       # Production config
    ├── staging.yml          # Staging config
    └── development.yml      # Development config
```

---

## 4. Frontend Architecture

### 4.1 Technology Stack
- **Framework**: Next.js 14 with TypeScript
- **Styling**: Tailwind CSS + custom design system
- **State Management**: Zustand + React Query
- **Visualizations**: Plotly.js + D3.js
- **UI Components**: Radix UI + custom components
- **Testing**: Jest + React Testing Library

### 4.2 Design System
**Location**: `frontend/design-system/`

**Components**:
```
design-system/
├── tokens/
│   ├── colors.json          # Color palette
│   ├── typography.json      # Typography scale
│   ├── spacing.json         # Spacing scale
│   └── breakpoints.json     # Responsive breakpoints
├── components/
│   ├── atoms/               # Basic components
│   ├── molecules/           # Composite components
│   ├── organisms/           # Complex components
│   └── templates/           # Page templates
├── themes/
│   ├── light.json           # Light theme
│   ├── dark.json            # Dark theme
│   └── high-contrast.json   # Accessibility theme
└── utils/
    ├── theme-provider.tsx   # Theme context
    └── style-utils.ts       # Style utilities
```

### 4.3 Application Structure
**Location**: `frontend/src/`

```
src/
├── app/                     # Next.js app router
│   ├── layout.tsx           # Root layout
│   ├── page.tsx             # Home page
│   ├── upload/              # File upload page
│   ├── jobs/                # Job management
│   ├── job/[id]/            # Job details
│   ├── compare/             # Comparison interface
│   └── api/                 # API routes
├── components/              # Reusable components
│   ├── ui/                  # Base UI components
│   ├── charts/              # Visualization components
│   ├── forms/               # Form components
│   └── layout/              # Layout components
├── hooks/                   # Custom React hooks
├── lib/                     # Utility libraries
├── stores/                  # Zustand stores
├── types/                   # TypeScript types
└── utils/                   # Helper functions
```

---

## 5. Advanced Features

### 5.1 Multi-Omics Integration
- **Proteomics Support**: UniProt IDs, protein modifications
- **Metabolomics Support**: Metabolite IDs, concentration data
- **Phosphoproteomics**: Phosphorylation site analysis
- **Epigenomics**: ChIP-seq, ATAC-seq integration
- **Joint Analysis**: Multi-layer pathway activity inference
- **Time-Course Analysis**: Longitudinal data processing

### 5.2 Custom Gene Sets & Community
- **User Uploads**: GMT files, custom pathway definitions
- **Community Repository**: Shared gene sets with versioning
- **Annotation Editing**: Custom metadata mapping
- **Gene Set Enrichment**: Custom pathway analysis

### 5.3 Collaboration & Lab Features
- **Multi-User Support**: User accounts, project sharing
- **Project Workspaces**: Team collaboration, version history
- **Annotation System**: Notes, tags, bookmarks
- **LIMS Integration**: API endpoints for lab systems
- **Audit Trail**: Complete action logging

### 5.4 Automation & Workflows
- **Pipeline Definition**: YAML/JSON workflow configuration
- **Scheduling**: Cron jobs for recurring analyses
- **Batch Processing**: Multiple dataset processing
- **Script Generation**: Reproducible analysis scripts

### 5.5 Advanced Visual Analytics
- **Network Visualization**: Interactive pathway networks
- **Multi-Layer Dashboards**: Cross-omics visualization
- **3D Plots**: Large dataset visualization
- **Story Reports**: Presentation-ready slide decks

### 5.6 Plugin Architecture
- **Plugin Interface**: Extensible analysis modules
- **Hook Points**: Custom processing stages
- **Plugin Registry**: Enable/disable plugins
- **Versioned API**: Plugin development support

---

## 6. Implementation Roadmap

### Phase 1: Core Backend + Standalone CLI (Weeks 1-4)
**Priority**: High
**Deliverables**:
- [ ] Modular package structure setup
- [ ] Core normalization engine
- [ ] Basic analysis engines (ORA, GSEA)
- [ ] Database adapters (KEGG, Reactome, GO)
- [ ] Standalone CLI with entry points
- [ ] Job management system
- [ ] Basic API endpoints
- [ ] Unit tests for core modules

**Files to Create**:
```
pathwaylens_core/
├── __init__.py
├── normalization/
│   ├── __init__.py
│   ├── normalizer.py
│   ├── format_detector.py
│   ├── id_converter.py
│   ├── species_mapper.py
│   ├── validation.py
│   ├── schemas.py
│   └── utils.py
├── analysis/
│   ├── __init__.py
│   ├── engine.py
│   ├── ora_engine.py
│   ├── gsea_engine.py
│   ├── schemas.py
│   └── utils.py
├── data/
│   ├── __init__.py
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── kegg_adapter.py
│   │   ├── reactome_adapter.py
│   │   └── go_adapter.py
│   ├── mapping/
│   │   ├── __init__.py
│   │   ├── gene_mapper.py
│   │   └── ortholog_mapper.py
│   └── cache/
│       ├── __init__.py
│       └── cache_manager.py
└── utils/
    ├── __init__.py
    └── common.py

pathwaylens_cli/
├── __init__.py
├── main.py                  # Main CLI entry point
├── commands/
│   ├── __init__.py
│   ├── normalize.py         # pathwaylens normalize
│   ├── analyze.py           # pathwaylens analyze
│   ├── compare.py           # pathwaylens compare
│   ├── visualize.py         # pathwaylens visualize
│   ├── config.py            # pathwaylens config
│   └── info.py              # pathwaylens info
├── utils/
│   ├── __init__.py
│   ├── formatters.py        # Output formatting
│   ├── validators.py        # Input validation
│   ├── progress.py          # Progress indicators
│   └── colors.py            # Terminal colors
└── config/
    ├── __init__.py
    ├── cli_config.py        # CLI-specific configuration
    └── user_preferences.py  # User preference management

pyproject.toml               # Updated with entry points
setup.py                     # Alternative setup script
```

### Phase 2: API & Job Management (Weeks 5-6)
**Priority**: High
**Deliverables**:
- [ ] FastAPI application setup
- [ ] Job queue with Celery + Redis
- [ ] PostgreSQL database schema
- [ ] Authentication system
- [ ] API documentation
- [ ] Docker containerization

**Files to Create**:
```
pathwaylens_api/
├── __init__.py
├── main.py
├── routes/
│   ├── __init__.py
│   ├── upload.py
│   ├── normalize.py
│   ├── analyze.py
│   ├── compare.py
│   ├── results.py
│   └── jobs.py
├── schemas/
│   ├── __init__.py
│   ├── job_schema.py
│   ├── analysis_schema.py
│   └── comparison_schema.py
├── middleware/
│   ├── __init__.py
│   ├── auth.py
│   ├── rate_limit.py
│   └── logging.py
└── utils/
    ├── __init__.py
    ├── database.py
    └── storage.py
```

### Phase 3: Frontend Foundation (Weeks 7-9)
**Priority**: High
**Deliverables**:
- [ ] Next.js application setup
- [ ] Design system implementation
- [ ] Basic UI components
- [ ] File upload interface
- [ ] Job management interface
- [ ] Basic visualizations

**Files to Create**:
```
frontend/
├── package.json
├── next.config.js
├── tailwind.config.js
├── tsconfig.json
├── design-system/
│   ├── tokens/
│   │   ├── colors.json
│   │   ├── typography.json
│   │   ├── spacing.json
│   │   └── breakpoints.json
│   ├── components/
│   │   ├── atoms/
│   │   ├── molecules/
│   │   ├── organisms/
│   │   └── templates/
│   └── themes/
│       ├── light.json
│       ├── dark.json
│       └── high-contrast.json
├── src/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   ├── upload/
│   │   ├── jobs/
│   │   └── job/[id]/
│   ├── components/
│   │   ├── ui/
│   │   ├── charts/
│   │   ├── forms/
│   │   └── layout/
│   ├── hooks/
│   ├── lib/
│   ├── stores/
│   ├── types/
│   └── utils/
└── public/
    └── assets/
```

### Phase 4: Advanced Features (Weeks 10-12)
**Priority**: Medium
**Deliverables**:
- [ ] Multi-omics support
- [ ] Advanced visualizations
- [ ] Comparison engine
- [ ] Plugin system
- [ ] Custom gene sets
- [ ] Export functionality

**Files to Create**:
```
pathwaylens_core/
├── comparison/
│   ├── __init__.py
│   ├── engine.py
│   ├── overlap_analyzer.py
│   ├── correlation_analyzer.py
│   ├── clustering_analyzer.py
│   ├── visualization.py
│   ├── schemas.py
│   └── utils.py
├── visualization/
│   ├── __init__.py
│   ├── engine.py
│   ├── plotly_renderer.py
│   ├── static_renderer.py
│   ├── network_renderer.py
│   ├── dashboard_builder.py
│   ├── export_manager.py
│   ├── themes.py
│   └── utils.py
├── multi_omics/
│   ├── __init__.py
│   ├── proteomics.py
│   ├── metabolomics.py
│   ├── phosphoproteomics.py
│   ├── epigenomics.py
│   ├── joint_analysis.py
│   └── time_course.py
└── plugins/
    ├── __init__.py
    ├── base.py
    ├── registry.py
    └── examples/
        ├── network_enrichment.py
        └── causal_inference.py
```

### Phase 5: Production & DevOps (Weeks 13-14)
**Priority**: Medium
**Deliverables**:
- [ ] Kubernetes deployment
- [ ] Monitoring setup
- [ ] CI/CD pipeline
- [ ] Security hardening
- [ ] Performance optimization
- [ ] Documentation

**Files to Create**:
```
infra/
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── kubernetes/
│   ├── backend-deployment.yaml
│   ├── frontend-deployment.yaml
│   ├── redis-deployment.yaml
│   └── postgres-deployment.yaml
├── monitoring/
│   ├── prometheus.yml
│   ├── grafana-dashboards/
│   └── alerting.yml
├── ci-cd/
│   ├── .github/workflows/
│   │   ├── test.yml
│   │   ├── build.yml
│   │   └── deploy.yml
│   └── scripts/
│       ├── setup.sh
│       └── deploy.sh
└── config/
    ├── production.yml
    ├── staging.yml
    └── development.yml
```

### Phase 6: Advanced Analytics (Weeks 15-16)
**Priority**: Low
**Deliverables**:
- [ ] Machine learning integration
- [ ] Network analysis
- [ ] Time-series analysis
- [ ] Advanced statistics
- [ ] AI-powered insights
- [ ] Performance benchmarking

---

## 7. Configuration & Environment

### 7.1 Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/pathwaylens
REDIS_URL=redis://localhost:6379/0

# API
SECRET_KEY=your-secret-key
API_HOST=0.0.0.0
API_PORT=8000

# External APIs
NCBI_API_KEY=your-ncbi-key
STRING_API_TOKEN=your-string-token
MSIGDB_PATH=/path/to/msigdb

# Storage
STORAGE_BACKEND=local  # or s3, gcs, azure
STORAGE_PATH=/path/to/storage

# Monitoring
PROMETHEUS_ENDPOINT=http://localhost:9090
GRAFANA_ENDPOINT=http://localhost:3000

# Development
DEBUG=false
LOG_LEVEL=INFO
```

### 7.2 CLI Configuration & Packaging

#### 7.2.1 Entry Points Configuration
**File**: `pyproject.toml`
```toml
[project]
name = "pathwaylens"
version = "2.0.0"
description = "Next-generation computational biology platform for multi-omics pathway analysis"
authors = [
    {name = "PathwayLens Team", email = "pathwaylens@example.com"}
]
readme = "README.md"
requires-python = ">=3.8"
dependencies = [
    "typer>=0.9.0",
    "rich>=13.0.0",
    "click>=8.0.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.15.0",
    "requests>=2.31.0",
    "aiohttp>=3.8.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "loguru>=0.7.0",
    "biopython>=1.79",
    "mygene>=3.2.2",
    "openpyxl>=3.1.0",
    "xlrd>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=22.0.0",
    "flake8>=5.0.0",
    "mypy>=0.991",
]

# CLI Entry Points - This creates the standalone 'pathwaylens' command
[project.scripts]
pathwaylens = "pathwaylens_cli.main:app"

# Alternative entry point for backward compatibility
[project.gui-scripts]
pathwaylens-gui = "pathwaylens_cli.main:gui_app"

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["pathwaylens_*"]

[tool.setuptools.package-data]
"pathwaylens_cli" = ["templates/**/*", "config/**/*"]
```

#### 7.2.2 CLI Main Entry Point
**File**: `pathwaylens_cli/main.py`
```python
#!/usr/bin/env python3
"""
PathwayLens CLI - Standalone command-line interface.

This module provides the main entry point for the PathwayLens CLI,
enabling direct invocation as 'pathwaylens' command.
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pathwaylens_cli.commands import (
    normalize,
    analyze, 
    compare,
    visualize,
    config,
    info
)

# Initialize the main CLI app
app = typer.Typer(
    name="pathwaylens",
    help="🧬 PathwayLens: Next-generation computational biology platform",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Add subcommands
app.add_typer(normalize.app, name="normalize", help="Convert gene identifiers across formats")
app.add_typer(analyze.app, name="analyze", help="Perform pathway analysis")
app.add_typer(compare.app, name="compare", help="Compare multiple datasets")
app.add_typer(visualize.app, name="visualize", help="Generate visualizations")
app.add_typer(config.app, name="config", help="Manage configuration")
app.add_typer(info.app, name="info", help="Display system information")

# Global options
@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, 
        "--version", 
        "-v", 
        help="Show version and exit"
    ),
    verbose: bool = typer.Option(
        False, 
        "--verbose", 
        help="Enable verbose output"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    )
):
    """
    🧬 PathwayLens: Next-generation computational biology platform
    
    A comprehensive tool for pathway analysis across bulk RNA-seq, scRNA-seq/snRNA-seq,
    ATAC-seq, proteomics, and arbitrary gene lists with support for multiple pathway
    databases and robust ID conversion.
    
    Examples:
        pathwaylens normalize genes.csv --species human --target-type symbol
        pathwaylens analyze deseq2_results.csv --databases kegg,reactome
        pathwaylens compare dataset1.csv dataset2.csv --species human
    """
    if version:
        console = Console()
        console.print(Panel.fit(
            "[bold blue]PathwayLens v2.0.0[/bold blue]\n"
            "Next-generation computational biology platform\n"
            "Built with ❤️ for the bioinformatics community",
            title="🧬 PathwayLens"
        ))
        raise typer.Exit()

if __name__ == "__main__":
    app()
```

#### 7.2.3 CLI Command Examples
```bash
# Installation (creates 'pathwaylens' command in PATH)
pip install pathwaylens

# Basic usage
pathwaylens --help
pathwaylens --version

# Gene normalization
pathwaylens normalize genes.csv --species human --target-type symbol
pathwaylens normalize expression_matrix.csv --species mouse --target-type ensembl

# Pathway analysis
pathwaylens analyze deseq2_results.csv --databases kegg,reactome,go
pathwaylens analyze proteomics_data.csv --analysis-type gsea --species human

# Dataset comparison
pathwaylens compare dataset1.csv dataset2.csv --species human
pathwaylens compare *.csv --comparison-type gene_overlap

# Visualization
pathwaylens visualize results.json --output-format html
pathwaylens visualize analysis_results/ --interactive

# Configuration
pathwaylens config show
pathwaylens config set database.kegg.enabled true
pathwaylens config init

# System information
pathwaylens info
pathwaylens info databases
pathwaylens info version
```

### 7.3 Configuration Files

#### 7.3.1 Backend Configuration
**File**: `config/pathwaylens.yml`
```yaml
version: "2.0.0"
debug: false
verbose: false

databases:
  ensembl:
    name: ensembl
    enabled: true
    rate_limit: 15
    base_url: "https://rest.ensembl.org"
  ncbi:
    name: ncbi
    enabled: true
    rate_limit: 3
    base_url: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
  mygene:
    name: mygene
    enabled: true
    rate_limit: 10
    base_url: "https://mygene.info/v3"
  kegg:
    name: kegg
    enabled: true
    rate_limit: 10
    base_url: "https://rest.kegg.jp"
  reactome:
    name: reactome
    enabled: true
    rate_limit: 10
    base_url: "https://reactome.org/AnalysisService"
  go:
    name: go
    enabled: true
    rate_limit: 5
    base_url: "http://current.geneontology.org"

analysis:
  ambiguity_policy: "expand"
  species_required: true
  cross_species_allowed: false
  multiple_testing_correction: "fdr_bh"
  min_pathway_size: 5
  max_pathway_size: 500
  consensus_method: "stouffer"
  background_source: "database"
  gsea_permutations: 1000
  gsea_min_size: 15
  gsea_max_size: 500

cache:
  enabled: true
  base_dir: ".pathwaylens/cache"
  max_size_mb: 1000
  ttl_days: 90
  compression: true

output:
  base_dir: ".pathwaylens/results"
  formats: ["json", "markdown", "html", "graphml"]
  include_plots: true
  include_tables: true
  include_graphs: true
  interactive_plots: true
```

#### 7.2.2 Frontend Configuration
**File**: `frontend/design-system/tokens/colors.json`
```json
{
  "primary": {
    "50": "#eff6ff",
    "100": "#dbeafe",
    "200": "#bfdbfe",
    "300": "#93c5fd",
    "400": "#60a5fa",
    "500": "#3b82f6",
    "600": "#2563eb",
    "700": "#1d4ed8",
    "800": "#1e40af",
    "900": "#1e3a8a"
  },
  "secondary": {
    "50": "#fefce8",
    "100": "#fef9c3",
    "200": "#fef08a",
    "300": "#fde047",
    "400": "#facc15",
    "500": "#eab308",
    "600": "#ca8a04",
    "700": "#a16207",
    "800": "#854d0e",
    "900": "#713f12"
  },
  "success": {
    "50": "#f0fdf4",
    "100": "#dcfce7",
    "200": "#bbf7d0",
    "300": "#86efac",
    "400": "#4ade80",
    "500": "#22c55e",
    "600": "#16a34a",
    "700": "#15803d",
    "800": "#166534",
    "900": "#14532d"
  },
  "error": {
    "50": "#fef2f2",
    "100": "#fee2e2",
    "200": "#fecaca",
    "300": "#fca5a5",
    "400": "#f87171",
    "500": "#ef4444",
    "600": "#dc2626",
    "700": "#b91c1c",
    "800": "#991b1b",
    "900": "#7f1d1d"
  },
  "warning": {
    "50": "#fffbeb",
    "100": "#fef3c7",
    "200": "#fde68a",
    "300": "#fcd34d",
    "400": "#fbbf24",
    "500": "#f59e0b",
    "600": "#d97706",
    "700": "#b45309",
    "800": "#92400e",
    "900": "#78350f"
  },
  "neutral": {
    "50": "#fafafa",
    "100": "#f5f5f5",
    "200": "#e5e5e5",
    "300": "#d4d4d4",
    "400": "#a3a3a3",
    "500": "#737373",
    "600": "#525252",
    "700": "#404040",
    "800": "#262626",
    "900": "#171717"
  }
}
```

---

## 8. Database Schema

### 8.1 Job Management Tables
```sql
-- Jobs table
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'queued',
    parameters JSONB NOT NULL,
    input_files JSONB,
    output_files JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Job results table
CREATE TABLE job_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    result_type VARCHAR(50) NOT NULL,
    result_data JSONB NOT NULL,
    file_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Users table (for multi-user support)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Projects table
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Project jobs relationship
CREATE TABLE project_jobs (
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    PRIMARY KEY (project_id, job_id)
);
```

### 8.2 Analysis Results Tables
```sql
-- Analysis results table
CREATE TABLE analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL,
    species VARCHAR(50) NOT NULL,
    input_gene_count INTEGER,
    total_pathways INTEGER,
    significant_pathways INTEGER,
    database_results JSONB,
    consensus_results JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Pathway results table
CREATE TABLE pathway_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_result_id UUID REFERENCES analysis_results(id) ON DELETE CASCADE,
    pathway_id VARCHAR(255) NOT NULL,
    pathway_name TEXT NOT NULL,
    database VARCHAR(50) NOT NULL,
    p_value DECIMAL(10, 8),
    adjusted_p_value DECIMAL(10, 8),
    enrichment_score DECIMAL(10, 6),
    normalized_enrichment_score DECIMAL(10, 6),
    overlap_count INTEGER,
    pathway_count INTEGER,
    input_count INTEGER,
    overlapping_genes JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Comparison results table
CREATE TABLE comparison_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    comparison_type VARCHAR(50) NOT NULL,
    input_analysis_ids JSONB NOT NULL,
    overlap_statistics JSONB,
    correlation_results JSONB,
    clustering_results JSONB,
    visualization_data JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

---

## 9. API Endpoints

### 9.1 Core Endpoints
```python
# File Upload
POST /api/v1/upload
Content-Type: multipart/form-data
{
    "file": <file>,
    "species": "human",
    "feature_type": "gene"
}

# Normalization
POST /api/v1/normalize
{
    "job_id": "uuid",
    "target_type": "symbol",
    "species": "human",
    "ambiguity_resolution": "expand"
}

# Analysis
POST /api/v1/analyze
{
    "job_id": "uuid",
    "analysis_type": "ora",
    "databases": ["kegg", "reactome", "go"],
    "parameters": {
        "significance_threshold": 0.05,
        "min_pathway_size": 5,
        "max_pathway_size": 500
    }
}

# Comparison
POST /api/v1/compare
{
    "analysis_job_ids": ["uuid1", "uuid2"],
    "comparison_type": "gene_overlap",
    "aggregation_method": "robust_rank"
}

# Results
GET /api/v1/results/{job_id}
GET /api/v1/results/{job_id}/download
GET /api/v1/results/{job_id}/visualizations

# Job Management
GET /api/v1/jobs
GET /api/v1/jobs/{job_id}
DELETE /api/v1/jobs/{job_id}
```

### 9.2 Response Schemas
```python
# Job Status Response
{
    "job_id": "uuid",
    "status": "completed",
    "progress": 100,
    "created_at": "2024-01-01T00:00:00Z",
    "started_at": "2024-01-01T00:01:00Z",
    "completed_at": "2024-01-01T00:05:00Z",
    "result_url": "/api/v1/results/uuid"
}

# Analysis Result Response
{
    "job_id": "uuid",
    "analysis_type": "ora",
    "species": "human",
    "input_gene_count": 1500,
    "total_pathways": 250,
    "significant_pathways": 45,
    "database_results": {
        "kegg": [...],
        "reactome": [...],
        "go": [...]
    },
    "consensus_results": [...],
    "visualizations": {
        "dot_plot": "/api/v1/results/uuid/visualizations/dot_plot",
        "network": "/api/v1/results/uuid/visualizations/network",
        "heatmap": "/api/v1/results/uuid/visualizations/heatmap"
    }
}
```

---

## 10. Testing Strategy

### 10.1 Backend Testing
```python
# Unit Tests
tests/
├── unit/
│   ├── test_normalization.py
│   ├── test_analysis.py
│   ├── test_comparison.py
│   ├── test_visualization.py
│   └── test_data_adapters.py
├── integration/
│   ├── test_api_endpoints.py
│   ├── test_job_workflow.py
│   └── test_database_integration.py
├── e2e/
│   ├── test_complete_workflow.py
│   └── test_multi_user_scenarios.py
└── fixtures/
    ├── sample_data/
    └── mock_responses/
```

### 10.2 Frontend Testing
```typescript
// Component Tests
src/
├── __tests__/
│   ├── components/
│   │   ├── UploadWidget.test.tsx
│   │   ├── JobCard.test.tsx
│   │   ├── ChartContainer.test.tsx
│   │   └── FilterPanel.test.tsx
│   ├── pages/
│   │   ├── upload.test.tsx
│   │   ├── jobs.test.tsx
│   │   └── job-details.test.tsx
│   ├── hooks/
│   │   ├── useJobs.test.ts
│   │   └── useVisualizations.test.ts
│   └── utils/
│       ├── api.test.ts
│       └── formatters.test.ts
└── e2e/
    ├── upload-workflow.spec.ts
    ├── analysis-workflow.spec.ts
    └── comparison-workflow.spec.ts
```

---

## 11. Deployment & DevOps

### 11.1 Docker Configuration
```dockerfile
# Backend Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pathwaylens_core/ ./pathwaylens_core/
COPY pathwaylens_api/ ./pathwaylens_api/

EXPOSE 8000

CMD ["uvicorn", "pathwaylens_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# Frontend Dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
```

### 11.2 Kubernetes Manifests
```yaml
# Backend Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pathwaylens-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pathwaylens-backend
  template:
    metadata:
      labels:
        app: pathwaylens-backend
    spec:
      containers:
      - name: backend
        image: pathwaylens/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: pathwaylens-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: pathwaylens-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### 11.3 CI/CD Pipeline
```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:6
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest tests/ --cov=pathwaylens_core --cov=pathwaylens_api
        pytest tests/ --html=report.html --self-contained-html
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker images
      run: |
        docker build -t pathwaylens/backend:${{ github.sha }} -f infra/docker/Dockerfile.backend .
        docker build -t pathwaylens/frontend:${{ github.sha }} -f infra/docker/Dockerfile.frontend ./frontend
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push pathwaylens/backend:${{ github.sha }}
        docker push pathwaylens/frontend:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/pathwaylens-backend backend=pathwaylens/backend:${{ github.sha }}
        kubectl set image deployment/pathwaylens-frontend frontend=pathwaylens/frontend:${{ github.sha }}
        kubectl rollout status deployment/pathwaylens-backend
        kubectl rollout status deployment/pathwaylens-frontend
```

---

## 12. Migration Strategy

### 12.1 Backward Compatibility & CLI Migration
- **New CLI Interface**: Transform from `python3 -m pathwaylens.cli` to standalone `pathwaylens` command
- **Command Mapping**: 
  - `python3 -m pathwaylens.cli normalize` → `pathwaylens normalize`
  - `python3 -m pathwaylens.cli analyze` → `pathwaylens analyze`
  - `python3 -m pathwaylens.cli compare` → `pathwaylens compare`
- **Legacy Support**: Provide compatibility wrapper for existing scripts
- **Preserve Options**: Maintain all current command-line options and behavior
- **Keep Output Formats**: Ensure existing output formats remain unchanged
- **Migration Guide**: Comprehensive documentation for users transitioning to new CLI

### 12.2 Legacy Compatibility Wrapper
**File**: `pathwaylens_cli/legacy_wrapper.py`
```python
#!/usr/bin/env python3
"""
Legacy compatibility wrapper for PathwayLens CLI.

This module provides backward compatibility for existing scripts that use
'python3 -m pathwaylens.cli' by redirecting to the new standalone CLI.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pathwaylens_cli.main import app

if __name__ == "__main__":
    # Remove the module name from sys.argv to match new CLI format
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        sys.argv.pop(1)
    
    app()
```

**File**: `pathwaylens/__main__.py` (for backward compatibility)
```python
"""
Backward compatibility entry point for 'python3 -m pathwaylens.cli'
"""

import sys
from pathlib import Path

# Redirect to new CLI
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pathwaylens_cli.main import app

if __name__ == "__main__":
    app()
```

### 12.3 Data Migration
- Convert existing results to new schema
- Migrate user configurations
- Preserve analysis history
- Update database schemas

### 12.4 User Migration
- Provide migration scripts
- Update documentation
- Training materials for new features
- Support for existing workflows

---

## 13. Success Metrics

### 13.1 Performance Metrics
- **Response Time**: < 2s for API endpoints
- **Throughput**: 100+ concurrent users
- **Scalability**: Auto-scaling to 10+ instances
- **Availability**: 99.9% uptime

### 13.2 User Experience Metrics
- **Time to First Result**: < 30s for simple analysis
- **User Satisfaction**: > 4.5/5 rating
- **Feature Adoption**: > 80% of users try new features
- **Support Tickets**: < 5% of users need support

### 13.3 Technical Metrics
- **Test Coverage**: > 90%
- **Code Quality**: A grade on SonarQube
- **Security**: No critical vulnerabilities
- **Documentation**: 100% API coverage

---

## 14. Risk Mitigation

### 14.1 Technical Risks
- **Database Performance**: Implement caching and indexing
- **API Rate Limits**: Implement queuing and retry logic
- **Memory Usage**: Use streaming and chunking for large datasets
- **Security**: Implement proper authentication and input validation

### 14.2 Business Risks
- **User Adoption**: Provide training and migration support
- **Feature Complexity**: Start with core features, add advanced features gradually
- **Performance Issues**: Implement monitoring and alerting
- **Data Loss**: Implement backup and recovery procedures

---

## 15. Conclusion

This comprehensive upgrade blueprint transforms PathwayLens from a CLI-only tool into a next-generation computational biology platform. The modular architecture, advanced features, and robust infrastructure provide a solid foundation for future growth and innovation.

The phased implementation approach ensures that core functionality is delivered quickly while advanced features are added incrementally. The emphasis on testing, documentation, and user experience ensures a high-quality product that meets the needs of computational biologists.

**Next Steps**:
1. Review and approve this blueprint
2. Set up development environment
3. Begin Phase 1 implementation
4. Establish CI/CD pipeline
5. Start user testing and feedback collection

This upgrade positions PathwayLens as a leading platform in computational biology, ready to handle the complex multi-omics analyses of the future.
