# PathwayLens CLI Reference

This document provides a detailed reference for the PathwayLens Command Line Interface (CLI).

## Global Options

- `--version`: Show the version and exit.
- `--verbose`, `-v`: Enable verbose output for debugging.
- `--config`, `-c`: Path to a custom configuration file.

## Commands

### `analyze`

Perform pathway analysis (ORA, GSEA) on gene lists or expression data.

#### `analyze ora`

Perform Over-Representation Analysis (ORA).

**Usage:**
```bash
pathwaylens analyze ora [OPTIONS]
```

**Options:**
- `--input`, `-i`: Input file path (required).
- `--omic-type`: Omic type (e.g., `transcriptomics`, `proteomics`) (required).
- `--data-type`: Data type (e.g., `bulk`, `scrna`, `shotgun`) (required).
- `--tool`: Tool used to generate input (default: `auto`).
- `--databases`, `-d`: Databases to use (default: `kegg`).
- `--species`, `-s`: Species name (default: `human`).
- `--output-dir`, `-o`: Output directory (required).
- `--min-genes`: Minimum pathway size (default: 10).
- `--max-genes`: Maximum pathway size (default: 500).
- `--fdr-threshold`: FDR cutoff (default: 0.05).

#### `analyze gsea`

Perform Gene Set Enrichment Analysis (GSEA).

**Usage:**
```bash
pathwaylens analyze gsea [OPTIONS]
```

**Options:**
- `--input`, `-i`: Input ranked gene list file (required).
- `--omic-type`: Omic type (required).
- `--data-type`: Data type (required).
- `--tool`: Tool used (default: `auto`).
- `--databases`, `-d`: Databases to use (default: `kegg`).
- `--species`, `-s`: Species name (default: `human`).
- `--output-dir`, `-o`: Output directory (required).

### `compare`

Compare multiple datasets or analysis results.

**Usage:**
```bash
pathwaylens compare [OPTIONS]
```

**Options:**
- `--inputs`, `-i`: Input files to compare (can be used multiple times) (required).
- `--mode`, `-m`: Comparison mode: `genes` or `pathways` (default: `genes`).
- `--labels`, `-l`: Labels for each input file (optional).
- `--output-dir`, `-o`: Output directory (required).
- `--omic-type`: Omic type (e.g., `transcriptomics`, `proteomics`) (required).
- `--data-type`: Data type (e.g., `bulk`, `singlecell`) (required).
- `--tool`: Tool used to generate input (default: `auto`).
- `--databases`, `-d`: Databases (for `pathways` mode).
- `--species`, `-s`: Species (default: `human`).

### `normalize`

Convert gene identifiers across formats.

**Usage:**
```bash
pathwaylens normalize [OPTIONS]
```

**Options:**
- `--input`, `-i`: Input file (required).
- `--target-type`, `-t`: Target identifier type (e.g., `symbol`, `ensembl`) (required).
- `--species`, `-s`: Species name (default: `human`).
- `--output`, `-o`: Output file path.

### `visualize`

Generate standalone visualizations.

**Usage:**
```bash
pathwaylens visualize [OPTIONS] COMMAND [ARGS]...
```

**Subcommands:**
- `volcano`: Create a volcano plot.
- `dotplot`: Create a dot plot.
- `heatmap`: Create a heatmap.
- `network`: Create a pathway network plot.

### `workflow`

Run and validate workflows.

**Usage:**
```bash
pathwaylens workflow [OPTIONS] COMMAND [ARGS]...
```

**Subcommands:**
- `run`: Execute a workflow file.
- `validate`: Validate a workflow file.
