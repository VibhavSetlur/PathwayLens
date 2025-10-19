# PathwayLens 2.0 User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

## Introduction

PathwayLens 2.0 is a next-generation computational biology platform for multi-omics pathway analysis. It provides both a command-line interface (CLI) and a web-based user interface for comprehensive pathway analysis workflows.

### Key Features

- **Multi-omics Support**: Genomics, transcriptomics, proteomics, metabolomics, phosphoproteomics, epigenomics
- **Cross-species Analysis**: Convert and compare data across different species
- **Interactive Visualizations**: Dynamic plots and network visualizations
- **Reproducible Analysis**: Job tracking, configuration files, and versioning
- **Scalable Processing**: Support for large datasets and batch processing

## Getting Started

### Installation

See [INSTALLATION.md](INSTALLATION.md) for detailed installation instructions.

### Quick Start

1. **Initialize Configuration**:
   ```bash
   pathwaylens config init
   ```

2. **Check System Status**:
   ```bash
   pathwaylens info
   ```

3. **Run Your First Analysis**:
   ```bash
   pathwaylens analyze tests/data/gene_list.txt --databases kegg,reactome,go
   ```

## Basic Usage

### 1. Gene Identifier Normalization

Before performing pathway analysis, you often need to normalize gene identifiers to a consistent format.

#### Example: Normalize Gene Symbols to Ensembl IDs

```bash
# Input file: genes.csv
# Format: gene_symbol,expression_value
# TP53,2.5
# BRCA1,1.8
# MYC,3.2

pathwaylens normalize genes.csv --species human --target-type ensembl --output normalized_genes.csv
```

#### Example: Cross-species Mapping

```bash
# Convert mouse genes to human orthologs
pathwaylens normalize mouse_genes.csv --species mouse --target-species human --output human_orthologs.csv
```

### 2. Pathway Enrichment Analysis

#### Over-Representation Analysis (ORA)

```bash
# Basic ORA analysis
pathwaylens analyze genes.csv --databases kegg,reactome,go --output-dir ora_results/

# Custom parameters
pathwaylens analyze genes.csv \
  --databases kegg,reactome,go,msigdb \
  --significance-threshold 0.01 \
  --min-pathway-size 10 \
  --max-pathway-size 200 \
  --correction-method fdr_bh \
  --output-dir custom_ora_results/
```

#### Gene Set Enrichment Analysis (GSEA)

```bash
# GSEA requires ranked gene list
# Input format: gene_id,rank_score
# ENSG00000012048,2.5
# ENSG00000012049,1.8

pathwaylens analyze ranked_genes.csv --analysis-type gsea --databases kegg,reactome --output-dir gsea_results/
```

### 3. Visualization

Generate interactive and static visualizations from your analysis results.

```bash
# Interactive visualizations
pathwaylens visualize ora_results/results.json --plot-types dot_plot,volcano_plot,network_plot --interactive

# Static visualizations
pathwaylens visualize ora_results/results.json --plot-types dot_plot,volcano_plot --interactive=false --format png
```

### 4. Dataset Comparison

Compare multiple datasets or analysis results.

```bash
# Compare gene lists
pathwaylens compare dataset1.csv dataset2.csv --comparison-type gene_overlap --output-dir comparison_results/

# Compare analysis results
pathwaylens compare ora_results/ gsea_results/ --comparison-type pathway_concordance --output-dir comparison_results/
```

## Advanced Features

### 1. Multi-omics Analysis

PathwayLens 2.0 supports various omics data types:

#### Genomics Data

```bash
# SNP analysis
pathwaylens analyze snps.csv --databases kegg,reactome,go --analysis-type genomic

# Copy number variation
pathwaylens analyze cnv_data.csv --databases kegg,reactome,go --analysis-type genomic
```

#### Transcriptomics Data

```bash
# RNA-seq differential expression
pathwaylens analyze rnaseq_results.csv --databases kegg,reactome,go --analysis-type transcriptomic

# Microarray data
pathwaylens analyze microarray_data.csv --databases kegg,reactome,go --analysis-type transcriptomic
```

#### Proteomics Data

```bash
# Protein expression
pathwaylens analyze proteomics_data.csv --databases kegg,reactome,go --analysis-type proteomic

# Phosphoproteomics
pathwaylens analyze phospho_data.csv --databases kegg,reactome,go --analysis-type phosphoproteomic
```

#### Metabolomics Data

```bash
# Metabolite analysis
pathwaylens analyze metabolites.csv --databases kegg,reactome,go --analysis-type metabolomic
```

#### Epigenomics Data

```bash
# ChIP-seq peaks
pathwaylens analyze chipseq_peaks.csv --databases kegg,reactome,go --analysis-type epigenomic

# Methylation data
pathwaylens analyze methylation_data.csv --databases kegg,reactome,go --analysis-type epigenomic
```

### 2. Batch Processing

Process multiple files or directories efficiently:

```bash
# Batch normalization
pathwaylens normalize batch data/ --pattern "*.csv" --species human --target-type ensembl

# Batch analysis
pathwaylens analyze batch data/ --pattern "*.csv" --databases kegg,reactome,go

# Batch visualization
pathwaylens visualize batch results/ --pattern "*.json" --plot-types dot_plot,volcano_plot
```

### 3. Custom Configuration

Create and manage custom configurations for different analysis types:

```bash
# Initialize configuration
pathwaylens config init

# Set custom parameters
pathwaylens config set analysis.significance_threshold 0.001
pathwaylens config set analysis.min_pathway_size 15
pathwaylens config set databases.kegg.enabled true
pathwaylens config set databases.reactome.enabled true

# Use configuration
pathwaylens analyze genes.csv --config custom_config.yml
```

### 4. Job Management

Track and manage long-running analyses:

```bash
# Start analysis in background
pathwaylens analyze large_dataset.csv --databases kegg,reactome,go --background

# Check job status
pathwaylens jobs status

# Cancel job
pathwaylens jobs cancel <job_id>
```

## Best Practices

### 1. Data Preparation

- **Validate Input**: Always validate your input data before analysis
- **Check Gene IDs**: Ensure gene identifiers are in the correct format
- **Species Consistency**: Verify species information is accurate
- **Quality Control**: Remove low-quality or ambiguous data

### 2. Analysis Design

- **Appropriate Databases**: Choose databases relevant to your research question
- **Significance Thresholds**: Use appropriate significance thresholds (typically 0.05 or 0.01)
- **Pathway Size Limits**: Set reasonable min/max pathway sizes (e.g., 5-500 genes)
- **Multiple Testing**: Always apply multiple testing correction

### 3. Interpretation

- **Biological Relevance**: Consider biological context when interpreting results
- **Pathway Overlap**: Look for overlapping pathways across different analyses
- **Cross-validation**: Validate findings with independent datasets
- **Literature Review**: Compare results with published literature

### 4. Reproducibility

- **Configuration Files**: Use configuration files for reproducible analyses
- **Version Control**: Track analysis versions and parameters
- **Documentation**: Document analysis steps and decisions
- **Data Sharing**: Share analysis code and configuration files

## Troubleshooting

### Common Issues

#### 1. File Format Errors

**Problem**: "Invalid file format" error

**Solution**:
- Check file format (CSV, TSV, Excel, JSON)
- Verify column headers
- Ensure proper encoding (UTF-8)
- Check for special characters

#### 2. Gene ID Conversion Issues

**Problem**: Low conversion rates or missing genes

**Solution**:
- Verify gene identifier type
- Check species information
- Use appropriate ambiguity policies
- Consider alternative identifier types

#### 3. Database Connection Issues

**Problem**: "Database unavailable" error

**Solution**:
- Check internet connection
- Verify database URLs
- Check rate limits
- Try alternative databases

#### 4. Memory Issues

**Problem**: "Out of memory" error

**Solution**:
- Reduce dataset size
- Increase system memory
- Use batch processing
- Optimize analysis parameters

#### 5. Performance Issues

**Problem**: Slow analysis or timeouts

**Solution**:
- Enable caching
- Use parallel processing
- Optimize database selection
- Reduce pathway size limits

### Debug Mode

Enable debug mode for detailed error information:

```bash
# Global debug mode
pathwaylens --verbose <command>

# Command-specific debug
pathwaylens analyze genes.csv --verbose

# Configuration debug
pathwaylens config set debug true
```

### Log Files

Check log files for detailed error information:

```bash
# View recent logs
tail -f ~/.pathwaylens/logs/pathwaylens.log

# Search for errors
grep -i error ~/.pathwaylens/logs/pathwaylens.log
```

## FAQ

### General Questions

**Q: What file formats does PathwayLens support?**

A: PathwayLens supports CSV, TSV, Excel (.xlsx, .xls), JSON, and plain text files.

**Q: Which species are supported?**

A: PathwayLens supports human, mouse, rat, Drosophila, zebrafish, C. elegans, and yeast.

**Q: How do I convert between different gene identifier types?**

A: Use the `normalize` command with appropriate `--target-type` parameter.

**Q: Can I analyze multiple datasets simultaneously?**

A: Yes, use the `compare` command or batch processing features.

### Analysis Questions

**Q: What's the difference between ORA and GSEA?**

A: ORA tests for over-representation of genes in pathways, while GSEA tests for enrichment of gene sets in ranked gene lists.

**Q: Which databases should I use?**

A: Choose databases relevant to your research. KEGG, Reactome, and GO are commonly used.

**Q: How do I set appropriate significance thresholds?**

A: Use 0.05 for general analysis, 0.01 for more stringent analysis, and 0.001 for very stringent analysis.

**Q: What are pathway size limits?**

A: Typically 5-500 genes. Smaller pathways may be too specific, larger pathways may be too general.

### Technical Questions

**Q: How do I enable caching?**

A: Set `cache.enabled: true` in your configuration file.

**Q: Can I run analyses in parallel?**

A: Yes, PathwayLens automatically uses multiple cores when available.

**Q: How do I manage large datasets?**

A: Use batch processing, increase system memory, or split datasets.

**Q: How do I update PathwayLens?**

A: Use `pip install --upgrade pathwaylens` or update your conda environment.

### Output Questions

**Q: What output formats are available?**

A: JSON, CSV, HTML, Markdown, PNG, SVG, and PDF.

**Q: How do I generate publication-ready figures?**

A: Use the `visualize` command with `--format pdf` or `--format svg`.

**Q: Can I customize visualizations?**

A: Yes, use configuration files or command-line options to customize plots.

**Q: How do I export results for further analysis?**

A: Use JSON or CSV output formats for programmatic access.

## Getting Help

- **Documentation**: Check the `docs/` directory
- **CLI Help**: Use `pathwaylens <command> --help`
- **Examples**: Use sample data in `tests/data/`
- **Community**: Join our discussion forum
- **Support**: Contact our support team

## Additional Resources

- [CLI Reference](CLI_REFERENCE.md)
- [Installation Guide](INSTALLATION.md)
- [Quick Start Guide](QUICKSTART.md)
- [API Reference](API_REFERENCE.md)
- [Contributing Guide](CONTRIBUTING.md)
