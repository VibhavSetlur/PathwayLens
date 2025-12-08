
# PathwayLens Tutorial: End-to-End Analysis Pipeline

This tutorial guides you through a complete pathway analysis workflow, from raw gene identifiers to publication-ready visualization.

## Prerequisites

Ensure PathwayLens is installed:
```bash
pip install pathwaylens
```

## 1. Data Preparation (Raw Input)

Start with a list of gene identifiers. For this tutorial, we assume you have a file `genes.txt` containing Ensembl IDs (e.g., from a differential expression analysis).

**Example `genes.txt`:**
```text
ENSG00000141510
ENSG00000130234
ENSG00000111640
...
```

## 2. Normalization (ID Conversion)

Pathway databases (KEGG, Reactome) often require Entrez IDs or specific formats. Use the `normalize` command to convert your identifiers and ensure they are valid.

```bash
pathwaylens normalize gene_ids \
    --input genes.txt \
    --input-format ensembl \
    --output-format entrez \
    --species human \
    --output normalized_genes.json
```

**Output:**
- `normalized_genes.json`: Contains the mapped IDs.
- `normalization.log`: Detailed log of successful and failed mappings. **Check this log to ensure high coverage.**

## 3. Pathway Analysis (Filtering & Statistics)

Run Over-Representation Analysis (ORA) on the normalized gene list. 

**Critical Step:** Apply strict filtering to ensure statistical rigor.
- `--fdr`: False Discovery Rate threshold (default 0.05).
- `--min-size`: Minimum pathway size (e.g., 10) to avoid small, spurious pathways.
- `--max-size`: Maximum pathway size (e.g., 500) to avoid overly broad terms.

```bash
pathwaylens analyze ora \
    --input normalized_genes.json \
    --database kegg \
    --species human \
    --fdr 0.05 \
    --min-size 10 \
    --max-size 500 \
    --output results/
```

## 4. Interpretation & Visualization

Navigate to the `results/` directory to explore the findings.

### Output Files
- `analysis_result.json`: Complete results with all statistical metrics.
- `pathway_summary.csv`: Tabular summary suitable for Excel/Pandas.

### Visualizations
- `dotplot.png`: Shows top enriched pathways with size and significance.
- `enrichment_map.html`: Interactive network of related pathways.

### Interactive Report
Open `report.html` in your browser to:
- Filter pathways by p-value.
- Hover over nodes to see gene overlap.
- Click pathways to view external database entries.

## 5. Advanced: Single-Cell Analysis

For scRNA-seq data (`.h5ad`), use the `single-cell` mode.

```bash
pathwaylens analyze single-cell \
    --input data.h5ad \
    --database reactome \
    --method mean_zscore \
    --output sc_results/
```

**Note:** Ensure your `.h5ad` file contains normalized counts (e.g., after SCTransform or similar). PathwayLens assumes the input matrix is already normalized for technical noise.
