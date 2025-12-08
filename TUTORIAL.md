# PathwayLens Tutorial

Welcome to PathwayLens! This tutorial will guide you through the process of analyzing your omics data, from installation to visualization.

## 1. Installation

PathwayLens is a Python package. You can install it via pip:

```bash
pip install pathwaylens
```

For R object support (optional), install `rpy2`:
```bash
pip install rpy2
```

## 2. Preparing Your Data

PathwayLens accepts various input formats:
- **CSV/TSV:** A list of genes (one per line) or a table with columns like `gene`, `logFC`, `p_value`.
- **R Objects:** `.rds` files (requires `rpy2`).
- **AnnData:** `.h5ad` files (for single-cell data).

### Example Input (genes.csv)
```csv
gene,logFC,p_value
TP53,2.5,0.001
BRCA1,-1.8,0.005
EGFR,1.2,0.04
```

## 3. Running Analysis

### 3.1 Over-Representation Analysis (ORA)

Run ORA on a gene list against KEGG and Reactome:

```bash
pathwaylens analyze ora \
  --input genes.csv \
  --omic-type transcriptomics \
  --data-type bulk \
  --databases kegg,reactome \
  --species human \
  --output-dir results_ora
```

**Key Parameters:**
- `--lfc-threshold`: Filter genes by Log Fold Change (default: 1.0).
- `--fdr-threshold`: Significance cutoff (default: 0.05).
- `--background`: Specify background genes (file or size).

### 3.2 Comparison Analysis

Compare two datasets (e.g., Control vs. Treated):

```bash
pathwaylens compare \
  --inputs control.csv \
  --inputs treated.csv \
  --labels "Control" \
  --labels "Treated" \
  --comparison-type condition \
  --stage counts \
  --method simple \
  --output-dir results_compare
```

### 3.3 Single-Cell Pathway Scoring

Calculate pathway activity scores for single cells (e.g., from scRNA-seq):

```bash
pathwaylens analyze single-cell \
  --input matrix.csv \
  --database kegg \
  --species human \
  --method mean_zscore \
  --output-dir results_sc
```

**Note:** Input can be a CSV (genes x cells) or `.h5ad` file.

### 3.4 Normalization

Convert gene IDs (e.g., Symbol to Entrez):

```bash
pathwaylens normalize gene-ids \
  --input genes.txt \
  --input-format symbol \
  --output-format entrez \
  --service mygene \
  --output normalized.json
```

## 4. Configuration File

For reproducible runs, define your parameters in a `config.yaml` file:

```yaml
# config.yaml
input: genes.csv
omic_type: transcriptomics
data_type: bulk
databases: [kegg, reactome]
species: human
lfc_threshold: 1.5
output_dir: results_config
```

Run with:
```bash
pathwaylens analyze ora --config config.yaml
```

## 5. Python API

You can also use PathwayLens within Python scripts or Jupyter Notebooks:

```python
import asyncio
from pathwaylens_core.api import PathwayLens

async def main():
    pl = PathwayLens()
    
    # Run Analysis
    result = await pl.analyze(
        gene_list=["TP53", "BRCA1", "EGFR"],
        omic_type="transcriptomics",
        data_type="bulk",
        databases=["kegg"],
        species="human"
    )
    
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## 6. Interpreting Results

After running `analyze`, check the output directory:
- **`report.html`**: Interactive dashboard with summary stats and plots.
- **`results.csv`**: Detailed table of enriched pathways.
- **`network.html`**: Enrichment Map visualization (if applicable).
- **`analysis_metadata.json`**: Record of all parameters used.

Happy analyzing!
