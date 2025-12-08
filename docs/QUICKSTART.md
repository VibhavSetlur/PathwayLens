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
pathwaylens analyze ora \
  --input tests/data/input_data/gene_list.txt \
  --databases kegg,reactome,go \
  --species human \
  --output-dir analysis_results

# GSEA analysis with custom parameters
pathwaylens analyze gsea \
  --input tests/data/input_data/ranked_genes.csv \
  --databases kegg,reactome \
  --output-dir gsea_results
```

### 4. Dataset Comparison

Compare multiple datasets:

```bash
# Compare two gene lists
pathwaylens compare \
  --inputs tests/data/input_data/gene_list.txt \
  --inputs tests/data/input_data/mouse_genes.csv \
  --mode genes \
  --omic-type transcriptomics \
  --data-type bulk \
  --output-dir comparison_results

# Compare analysis results (pathways)
pathwaylens compare \
  --inputs analysis_results/run.json \
  --inputs gsea_results/run.json \
  --mode pathways \
  --omic-type transcriptomics \
  --data-type bulk \
  --databases kegg \
  --species human \
  --output-dir comparison_results
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

# Batch analyze files (Note: batch mode not yet implemented)
# Use shell loop instead:
for file in tests/data/input_data/*.txt; do
  pathwaylens analyze ora --input "$file" \
    --databases kegg,reactome,go \
    --omic-type transcriptomics \
    --data-type bulk \
    --species human \
    --output-dir "batch_results/$(basename "$file" .txt)"
done
```

## Advanced Usage

### Custom Analysis Parameters

```bash
# Advanced ORA analysis
pathwaylens analyze ora \
  --input genes.csv \
  --databases kegg,reactome,go,msigdb \
  --omic-type transcriptomics \
  --data-type bulk \
  --species human \
  --fdr-threshold 0.001 \
  --min-genes 10 \
  --max-genes 200 \
  --output-dir advanced_ora

# Advanced GSEA analysis
pathwaylens analyze gsea \
  --input ranked_genes.csv \
  --databases kegg,reactome \
  --omic-type transcriptomics \
  --data-type bulk \
  --species human \
  --output-dir advanced_gsea
```

### Cross-Species Analysis

```bash
# Convert mouse genes to human orthologs
pathwaylens normalize mouse_genes.csv \
  --species mouse \
  --target-species human \
  --target-type symbol

# Analyze mouse genes
pathwaylens analyze ora \
  --input mouse_genes.csv \
  --species mouse \
  --databases kegg,reactome \
  --omic-type transcriptomics \
  --data-type bulk \
  --output-dir mouse_analysis
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

## Workflows and Plugins

### Run a Workflow

```bash
pathwaylens workflow validate pathwaylens_cli/templates/workflows/example.yaml
pathwaylens workflow run pathwaylens_cli/templates/workflows/example.yaml
```

### Manage Plugins

```bash
pathwaylens plugin list
pathwaylens plugin exec example-plugin --input data.json
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
