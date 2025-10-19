# PathwayLens 2.0 CLI Reference

## Overview

PathwayLens 2.0 provides a comprehensive command-line interface for pathway analysis. This reference covers all available commands, options, and usage examples.

## Global Options

All commands support these global options:

```bash
pathwaylens [GLOBAL_OPTIONS] <COMMAND> [COMMAND_OPTIONS]
```

### Global Options

- `--version`, `-v`: Show version and exit
- `--verbose`: Enable verbose output
- `--config`, `-c`: Path to configuration file
- `--help`, `-h`: Show help message

## Commands

### 1. normalize

Convert gene identifiers across formats and species.

```bash
pathwaylens normalize [OPTIONS] INPUT_FILE
```

#### Options

- `--output`, `-o`: Output file path
- `--species`, `-s`: Species of input data (human, mouse, rat, etc.)
- `--target-species`, `-t`: Target species for cross-species mapping
- `--target-type`: Target identifier type (symbol, ensembl, entrez, etc.)
- `--ambiguity-policy`: How to handle ambiguous mappings (expand, collapse, skip, error)
- `--validate/--no-validate`: Validate input before processing
- `--verbose`, `-v`: Enable verbose output

#### Examples

```bash
# Basic normalization
pathwaylens normalize genes.csv --species human --target-type symbol

# Cross-species mapping
pathwaylens normalize mouse_genes.csv --species mouse --target-species human

# Batch processing
pathwaylens normalize batch data/ --pattern "*.csv" --species human
```

### 2. analyze

Perform pathway enrichment analysis.

```bash
pathwaylens analyze [OPTIONS] INPUT_FILE
```

#### Options

- `--output-dir`, `-o`: Output directory for results
- `--analysis-type`, `-t`: Type of analysis (ora, gsea)
- `--databases`, `-d`: Databases to use (kegg, reactome, go, etc.)
- `--species`, `-s`: Species for analysis
- `--significance-threshold`, `-p`: Significance threshold
- `--correction-method`: Multiple testing correction method
- `--min-pathway-size`: Minimum pathway size
- `--max-pathway-size`: Maximum pathway size
- `--verbose`, `-v`: Enable verbose output

#### Examples

```bash
# Basic ORA analysis
pathwaylens analyze genes.csv --databases kegg,reactome,go

# GSEA analysis
pathwaylens analyze ranked_genes.csv --analysis-type gsea --databases kegg

# Custom parameters
pathwaylens analyze genes.csv --significance-threshold 0.01 --min-pathway-size 10
```

### 3. compare

Compare multiple datasets and analysis results.

```bash
pathwaylens compare [OPTIONS] INPUT_FILES...
```

#### Options

- `--output-dir`, `-o`: Output directory for results
- `--comparison-type`, `-t`: Type of comparison (gene_overlap, pathway_concordance, etc.)
- `--species`, `-s`: Species for comparison
- `--significance-threshold`, `-p`: Significance threshold
- `--databases`, `-d`: Databases to use
- `--verbose`, `-v`: Enable verbose output

#### Examples

```bash
# Compare gene lists
pathwaylens compare dataset1.csv dataset2.csv --comparison-type gene_overlap

# Compare analysis results
pathwaylens compare results1/ results2/ --comparison-type pathway_concordance

# Compare directories
pathwaylens compare directories condition1/ condition2/ condition3/
```

### 4. visualize

Generate visualizations from analysis results.

```bash
pathwaylens visualize [OPTIONS] INPUT_FILE
```

#### Options

- `--output-dir`, `-o`: Output directory for visualizations
- `--plot-types`, `-p`: Types of plots to generate
- `--interactive/--static`: Generate interactive or static plots
- `--format`, `-f`: Output format (html, png, svg, pdf)
- `--theme`: Plot theme (light, dark, high_contrast)
- `--verbose`, `-v`: Enable verbose output

#### Examples

```bash
# Generate interactive plots
pathwaylens visualize results.json --plot-types dot_plot,volcano_plot

# Generate static plots
pathwaylens visualize results.json --interactive=false --format png

# Batch visualization
pathwaylens visualize batch results/ --pattern "*.json"
```

### 5. config

Manage PathwayLens configuration.

```bash
pathwaylens config <SUBCOMMAND> [OPTIONS]
```

#### Subcommands

- `show`: Show current configuration
- `set KEY VALUE`: Set configuration value
- `get KEY`: Get configuration value
- `init`: Initialize configuration file
- `validate`: Validate configuration file

#### Examples

```bash
# Show configuration
pathwaylens config show

# Set values
pathwaylens config set databases.kegg.enabled true
pathwaylens config set analysis.significance_threshold 0.01

# Get values
pathwaylens config get databases.kegg.enabled

# Initialize configuration
pathwaylens config init
```

### 6. info

Display system information and status.

```bash
pathwaylens info [SUBCOMMAND] [OPTIONS]
```

#### Subcommands

- `main`: Show general system information
- `databases`: Show database information
- `version`: Show version information

#### Options

- `--verbose`, `-v`: Show detailed information

#### Examples

```bash
# General information
pathwaylens info

# Database information
pathwaylens info databases

# Version information
pathwaylens info version

# Verbose output
pathwaylens info --verbose
```

## Configuration

### Configuration File

PathwayLens uses a YAML configuration file located at `~/.pathwaylens/config.yml`.

#### Example Configuration

```yaml
version: "2.0.0"
debug: false
verbose: false

databases:
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

analysis:
  significance_threshold: 0.05
  correction_method: "fdr_bh"
  min_pathway_size: 5
  max_pathway_size: 500

cache:
  enabled: true
  base_dir: ".pathwaylens/cache"
  max_size_mb: 1000
  ttl_days: 90

output:
  base_dir: ".pathwaylens/results"
  formats: ["json", "markdown", "html"]
  include_plots: true
  interactive_plots: true
```

### Environment Variables

PathwayLens also supports environment variables:

```bash
export PATHWAYLENS_CONFIG_FILE="/path/to/config.yml"
export PATHWAYLENS_DEBUG=true
export PATHWAYLENS_VERBOSE=true
```

## Input Formats

### Supported File Formats

- **CSV**: Comma-separated values
- **TSV**: Tab-separated values
- **Excel**: .xlsx and .xls files
- **JSON**: JavaScript Object Notation
- **Text**: Plain text files with one gene per line

### Gene Identifier Types

- **Symbol**: Gene symbols (e.g., TP53, BRCA1)
- **Ensembl**: Ensembl gene IDs (e.g., ENSG00000012048)
- **Entrez**: NCBI Entrez gene IDs (e.g., 7157)
- **UniProt**: UniProt protein IDs (e.g., P04637)
- **RefSeq**: RefSeq gene IDs (e.g., NM_000546)

### Species Support

- **Human**: Homo sapiens
- **Mouse**: Mus musculus
- **Rat**: Rattus norvegicus
- **Drosophila**: Drosophila melanogaster
- **Zebrafish**: Danio rerio
- **C. elegans**: Caenorhabditis elegans
- **Yeast**: Saccharomyces cerevisiae

## Output Formats

### Analysis Results

- **JSON**: Complete results with metadata
- **CSV**: Tabular data for spreadsheet analysis
- **HTML**: Interactive reports and visualizations
- **Markdown**: Human-readable reports

### Visualizations

- **HTML**: Interactive plots (Plotly)
- **PNG**: Static raster images
- **SVG**: Vector graphics
- **PDF**: Publication-ready figures

## Error Handling

### Common Error Messages

1. **File not found**: Check file path and permissions
2. **Invalid format**: Ensure file format is supported
3. **Database unavailable**: Check network connection and database status
4. **Configuration error**: Validate configuration file
5. **Memory error**: Reduce dataset size or increase system memory

### Debug Mode

Enable debug mode for detailed error information:

```bash
pathwaylens --verbose <command>
```

Or set in configuration:

```bash
pathwaylens config set debug true
```

## Performance Tips

### Optimization

1. **Use caching**: Enable cache for repeated analyses
2. **Batch processing**: Process multiple files together
3. **Parallel processing**: Use multiple cores when available
4. **Database selection**: Use only necessary databases
5. **Pathway size limits**: Set appropriate min/max pathway sizes

### Resource Management

```bash
# Monitor resource usage
pathwaylens info --verbose

# Clear cache if needed
rm -rf ~/.pathwaylens/cache/*

# Check disk space
df -h ~/.pathwaylens/
```

## Examples

### Complete Workflow

```bash
# 1. Normalize gene identifiers
pathwaylens normalize genes.csv --species human --target-type ensembl

# 2. Perform pathway analysis
pathwaylens analyze normalized_genes.csv --databases kegg,reactome,go

# 3. Generate visualizations
pathwaylens visualize analysis_results/results.json --plot-types dot_plot,volcano_plot

# 4. Compare with another dataset
pathwaylens compare analysis_results/ other_results/ --comparison-type pathway_concordance
```

### Advanced Usage

```bash
# Cross-species analysis
pathwaylens normalize mouse_genes.csv --species mouse --target-species human
pathwaylens analyze mouse_genes.csv --species mouse --target-species human

# Custom analysis parameters
pathwaylens analyze genes.csv \
  --databases kegg,reactome,go,msigdb \
  --significance-threshold 0.001 \
  --min-pathway-size 10 \
  --max-pathway-size 200 \
  --correction-method fdr_bh

# Batch processing
pathwaylens analyze batch data/ --pattern "*.csv" --databases kegg,reactome
```

## Getting Help

- **Command help**: `pathwaylens <command> --help`
- **Global help**: `pathwaylens --help`
- **Documentation**: Check the `docs/` directory
- **Examples**: Use sample data in `tests/data/`
- **Community**: Join our discussion forum
