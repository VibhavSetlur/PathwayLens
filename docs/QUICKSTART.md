# PathwayLens 2.0 Quick Start Guide

## Overview

This guide will get you up and running with PathwayLens 2.0 in minutes. You'll learn how to perform basic pathway analysis operations using the command-line interface.

## Prerequisites

- PathwayLens 2.0 installed (see [Installation Guide](INSTALLATION.md))
- Basic familiarity with command-line tools
- Sample data files (provided in `tests/data/`)

## Basic Operations

### 1. Check Installation

First, verify that PathwayLens is properly installed:

```bash
# Check version
pathwaylens --version

# View system information
pathwaylens info

# Check available databases
pathwaylens info databases
```

### 2. Gene Identifier Normalization

Convert gene identifiers to a standard format:

```bash
# Normalize gene symbols to Ensembl IDs
pathwaylens normalize tests/data/input_data/gene_list.txt \
  --species human \
  --target-type ensembl \
  --output normalized_genes.csv

# View results
pathwaylens normalize --help
```

### 3. Pathway Enrichment Analysis

Perform pathway enrichment analysis on your gene list:

```bash
# Basic ORA analysis
pathwaylens analyze tests/data/input_data/gene_list.txt \
  --databases kegg,reactome,go \
  --species human \
  --output-dir analysis_results

# GSEA analysis with custom parameters
pathwaylens analyze tests/data/input_data/ranked_genes.csv \
  --analysis-type gsea \
  --databases kegg,reactome \
  --significance-threshold 0.01 \
  --output-dir gsea_results
```

### 4. Dataset Comparison

Compare multiple datasets:

```bash
# Compare two gene lists
pathwaylens compare tests/data/input_data/gene_list.txt tests/data/input_data/mouse_genes.csv \
  --comparison-type gene_overlap \
  --species human \
  --output-dir comparison_results

# Compare analysis results
pathwaylens compare analysis_results/ gsea_results/ \
  --comparison-type pathway_concordance
```

### 5. Generate Visualizations

Create publication-ready plots:

```bash
# Generate interactive visualizations
pathwaylens visualize analysis_results/results.json \
  --plot-types dot_plot,volcano_plot,heatmap \
  --interactive \
  --output-dir visualizations

# Generate static plots
pathwaylens visualize analysis_results/results.json \
  --plot-types dot_plot,network \
  --interactive=false \
  --format png
```

## Configuration

### Set Up Configuration

```bash
# Initialize configuration
pathwaylens config init

# View current settings
pathwaylens config show

# Modify settings
pathwaylens config set databases.kegg.enabled true
pathwaylens config set analysis.significance_threshold 0.01
```

### Database Configuration

```bash
# Enable/disable databases
pathwaylens config set databases.reactome.enabled true
pathwaylens config set databases.go.enabled false

# Set rate limits
pathwaylens config set databases.kegg.rate_limit 5
```

## Batch Processing

### Process Multiple Files

```bash
# Batch normalize files
pathwaylens normalize batch tests/data/input_data/ \
  --pattern "*.csv" \
  --species human \
  --target-type ensembl

# Batch analyze files
pathwaylens analyze batch tests/data/input_data/ \
  --pattern "*.txt" \
  --databases kegg,reactome,go \
  --output-dir batch_results
```

## Advanced Usage

### Custom Analysis Parameters

```bash
# Advanced ORA analysis
pathwaylens analyze genes.csv \
  --databases kegg,reactome,go,msigdb \
  --significance-threshold 0.001 \
  --min-pathway-size 10 \
  --max-pathway-size 200 \
  --correction-method fdr_bh

# Advanced GSEA analysis
pathwaylens analyze ranked_genes.csv \
  --analysis-type gsea \
  --gsea-permutations 2000 \
  --gsea-min-size 20 \
  --gsea-max-size 300
```

### Cross-Species Analysis

```bash
# Convert mouse genes to human orthologs
pathwaylens normalize mouse_genes.csv \
  --species mouse \
  --target-species human \
  --target-type symbol

# Analyze with cross-species mapping
pathwaylens analyze mouse_genes.csv \
  --species mouse \
  --target-species human \
  --databases kegg,reactome
```

## Output Files

### Understanding Results

PathwayLens generates several types of output files:

- **JSON files**: Machine-readable results with complete metadata
- **CSV files**: Tabular data for spreadsheet analysis
- **HTML files**: Interactive visualizations and reports
- **PNG/SVG files**: Static plots for publications

### Example Output Structure

```
analysis_results/
├── results.json              # Complete analysis results
├── results.csv               # Tabular results
├── summary.html              # Interactive summary
├── visualizations/           # Generated plots
│   ├── dot_plot.html
│   ├── volcano_plot.html
│   └── heatmap.html
└── metadata/                 # Analysis metadata
    ├── parameters.json
    └── log.txt
```

## Tips and Best Practices

### 1. Data Preparation
- Ensure gene identifiers are consistent
- Remove duplicate entries
- Check for missing values

### 2. Analysis Parameters
- Start with default parameters
- Adjust significance thresholds based on your needs
- Use appropriate pathway size limits

### 3. Database Selection
- Use multiple databases for comprehensive analysis
- Consider database-specific strengths
- Balance between coverage and specificity

### 4. Visualization
- Generate both interactive and static plots
- Use appropriate plot types for your data
- Customize themes for publication

## Troubleshooting

### Common Issues

#### 1. No Significant Results
```bash
# Try different parameters
pathwaylens analyze genes.csv \
  --significance-threshold 0.1 \
  --min-pathway-size 3 \
  --max-pathway-size 1000
```

#### 2. Low Mapping Rates
```bash
# Check gene identifier format
pathwaylens normalize genes.csv --validate

# Try different target types
pathwaylens normalize genes.csv --target-type symbol
```

#### 3. Database Connection Issues
```bash
# Check database status
pathwaylens info databases

# Test specific database
pathwaylens info databases --verbose
```

## Next Steps

Now that you're familiar with the basics:

1. **Explore Advanced Features**: Try multi-omics analysis and custom workflows
2. **Read the Full Documentation**: Check other guides in the `docs/` directory
3. **Join the Community**: Participate in discussions and share your experiences
4. **Contribute**: Help improve PathwayLens by reporting issues or contributing code

## Getting Help

- **Documentation**: Browse the `docs/` directory
- **CLI Help**: Use `pathwaylens --help` or `pathwaylens <command> --help`
- **Examples**: Check the `tests/data/` directory for sample files
- **Community**: Join our discussion forum for support
