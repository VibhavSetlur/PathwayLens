# PathwayLens CLI Commands: Complete User Guide

## Table of Contents
- [normalize](#normalize-command)
- [analyze](#analyze-command)
- [compare](#compare-command)
- [visualize](#visualize-command)
- [workflow](#workflow-command)

---

## normalize Command

Convert gene identifiers between different formats and species.

### Basic Usage

```bash
pathwaylens normalize INPUT [OPTIONS]
```

### Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--species` | TEXT | Target species (human, mouse, rat) | Required |
| `--from-type` | TEXT | Input ID type (auto-detected if not specified) | auto |
| `--to-type` | TEXT | Output ID type (symbol, ensembl, entrez) | symbol |
| `--output` | PATH | Output file path | stdout |
| `--unmapped` | PATH | Save unmapped genes to file | None |
| `--validate` | FLAG | Validate gene symbols against databases | False |

### Examples

**Example 1: Convert Ensembl IDs to Gene Symbols**
```bash
pathwaylens normalize ensembl_genes.txt \
  --species human \
  --to-type symbol \
  --output gene_symbols.txt
```

**Example 2: Auto-Detect ID Type**
```bash
pathwaylens normalize mixed_ids.csv \
  --species mouse \
  --to-type ensembl \
  --validate
```

**Example 3: Save Unmapped Genes**
```bash
pathwaylens normalize genes.txt \
  --species human \
  --to-type symbol \
  --output normalized.txt \
  --unmapped failed_mappings.txt
```

### Input Formats

**Supported**:
- Plain text (one gene per line)
- CSV/TSV with gene column
- Excel files (.xlsx, .xls)

**Auto-Detection**:
- ENSEMBL: `ENSG00000139618`, `ENSMUSG00000055866`
- Entrez: `672`, `7157`
- Gene symbols: `BRCA1`, `TP53`
- RefSeq: `NM_007294`, `NP_000537`

### Backend Details

**Implementation**: `pathwaylens_core/normalization/normalizer.py`

**Process**:
1. **Format Detection**: Analyze first 100 entries to detect ID type
2. **Database Query**: Query MyGene.info or local databases `
3. **Validation**: Check against species-specific databases
4. **Conflict Resolution**: Handle one-to-many mappings
5. **Quality Report**: Generate mapping statistics

**Databases Used**:
- MyGene.info (primary)
- Ensembl BioMart (for complex mappings)
- NCBI Gene (for Entrez conversions)
- HGNC/MGI (for official symbols)

---

## analyze Command

Perform pathway enrichment analysis using multiple methods.

### Subcommands

- `analyze ora` - Over-Representation Analysis
- `analyze gsea` - Gene Set Enrichment Analysis
- `analyze batch` - Batch process multiple files

### analyze ora

Over-representation analysis with hypergeometric test.

```bash
pathwaylens analyze ora --input FILE --databases DB [OPTIONS]
```

#### Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--input` | PATH | Input gene list or DE results | Required |
| `--databases` | TEXT | Databases (kegg, reactome, go, wikipathways) | Required |
| `--species` | TEXT | Species | human |
| `--fdr-threshold` | FLOAT | Adjusted p-value cutoff | 0.05 |
| `--min-genes` | INT | Minimum pathway size | 10 |
| `--max-genes` | INT | Maximum pathway size | 500 |
| `--background` | PATH | Custom background gene list | genome-wide |
| `--correction` | TEXT | Multiple testing correction (fdr_bh, bonferroni) | fdr_bh |
| `--output` | PATH | Output directory | results/ |

#### Input Handling

**Gene Lists (TXT/CSV)**:
```bash
# Simple gene list
pathwaylens analyze ora --input genes.txt --databases kegg --species human
```

**DESeq2 Results**:
```bash
# Automatically extracts significant genes
pathwaylens analyze ora --input deseq2_results.csv \
  --databases kegg reactome go \
  --fdr-threshold 0.01 \
  --lfc-threshold 1.5
```

**Seurat Markers**:
```bash
# Auto-detects Seurat format
pathwaylens analyze ora --input seurat_markers.csv \
  --cluster-column cluster \
  --databases go \
  --min-pct 0.25
```

#### Output Structure

```
results/
├── KEGG_enrichment.tsv              # Significant pathways
├── KEGG_enrichment_full.tsv         # All tested pathways
├── gene_pathway_mapping.tsv         # Gene-to-pathway map
├── summary_statistics.json          # Summary stats
└── enrichment_barplot.svg           # Visualization
```

#### Statistical Metrics

Every pathway includes:
- **P-value**: Hypergeometric test p-value
- **Adjusted P-value**: FDR-corrected p-value
- **Odds Ratio**: Effect size with 95% CI
- **Fold Enrichment**: Observed/expected ratio
- **Cohen's h**: Standardized effect size
- **Statistical Power**: Post-hoc power estimate
- **Expected Genes**: Chance overlap

#### Backend Details

**Implementation**: `pathwaylens_core/analysis/ora_engine.py`

**Algorithm**:
1. **Input Parsing**: Auto-detect format and extract genes
2. **Background**: Use genome-wide or custom background
3. **Pathway Loading**: Load from local or remote databases
4. **Enrichment Test**: Hypergeometric distribution
5. **Effect Size**: Calculate OR, CI, Cohen's h
6. **Multiple Testing**: FDR correction (Benjamini-Hochberg)
7. **Power Analysis**: Post-hoc power calculation

### analyze gsea

Gene Set Enrichment Analysis for ranked gene lists.

```bash
pathwaylens analyze gsea --input FILE --databases DB [OPTIONS]
```

#### Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--input` | PATH | Ranked gene list | Required |
| `--databases` | TEXT | Databases to test | Required |
| `--species` | TEXT | Species | human |
| `--ranking-metric` | TEXT | Metric (log2fc, stat, custom) | stat |
| `--permutations` | INT | Number of permutations | 1000 |
| `--min-size` | INT | Minimum pathway size | 15 |
| `--max-size` | INT | Maximum pathway size | 500 |
| `--output` | PATH | Output directory | results/ |

#### Input Formats

**Two-Column Format**:
```
gene,score
TP53,5.23
BRCA1,4.87
EGFR,-3.21
```

**DESeq2 Results** (auto-extracts stat column):
```bash
pathwaylens analyze gsea --input deseq2_results.csv \
  --databases kegg \
  --ranking-metric stat
```

#### Backend Details

**Implementation**: `pathwaylens_core/analysis/gsea_engine.py`

**Algorithm**:
1. **Ranking**: Sort genes by metric
2. **Enrichment Score**: Calculate ES using weighted KS statistic
3. **Normalization**: Normalize by pathway size
4. **Significance**: Permutation testing (1000+ permutations)
5. **FDR**: Estimate false discovery rate

---

## compare Command

Compare multiple datasets with robust handling of different formats.

### Basic Usage

```bash
pathwaylens compare FILE1 FILE2 [FILE3 ...] [OPTIONS]
```

### Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--type` | TEXT | Comparison type (bulk, singlecell, timeseries) | bulk |
| `--species` | TEXT | Species | human |
| `--metadata` | PATH | JSON metadata file | None |
| `--normalization` | TEXT | Normalization method | quantile |
| `--method` | TEXT | Analysis method (ora, gsea) | ora |
| `--aggregate-by` | TEXT | Aggregate scRNA by (cluster, celltype) | None |
| `--adjust-batch` | FLAG | Perform batch correction | False |
| `--output` | PATH | Output directory | comparison/ |

### Comparison Types

#### Bulk RNA-seq Comparison

```bash
pathwaylens compare \
  treated_deseq2.csv \
  untreated_deseq2.csv \
  knockdown_edger.csv \
  --type bulk \
  --species human \
  --method ora
```

**Handles**:
- Different tools (DESeq2, edgeR, limma)
- Different gene ID types
- Different significance thresholds

#### Single-Cell Comparison

```bash
pathwaylens compare \
  cluster1_markers.csv \
  cluster2_markers.csv \
  cluster3_markers.csv \
  --type singlecell \
  --aggregate-by cluster \
  --normalization sct \
  --min-cells 50
```

**SCTransform Handling**:
- Preserves variance structure
- Accounts for sequencing depth
- Handles dropout rates

#### Time-Series Comparison

```bash
pathwaylens compare \
  day0.csv day1.csv day3.csv day7.csv \
  --type timeseries \
  --timepoints 0,1,3,7 \
  --trend-analysis
```

### Metadata Integration

**Metadata JSON Format**:
```json
{
  "datasets": {
    "treated_deseq2.csv": {
      "condition": "treated",
      "batch": "batch1",
      "sequencing_depth": 50000000,
      "notes": "Drug treatment, 24h"
    },
    "untreated_deseq2.csv": {
      "condition": "untreated",
      "batch": "batch1",
      "sequencing_depth": 48000000,
      "notes": "Control"
    }
  }
}
```

Usage:
```bash
pathwaylens compare *.csv \
  --metadata experiment_info.json \
  --adjust-batch
```

### Normalization Methods

| Method | Use Case | Description |
|--------|----------|-------------|
| `none` | Same platform/tool | No normalization |
| `quantile` | Cross-platform | Quantile normalization |
| `zscore` | Different scales | Z-score standardization |
| `rank` | Robust comparison | Rank-based |
| `sct` | scRNA-seq | SCTransform-aware |
| `pseudobulk` | scRNA to bulk | Aggregate to pseudo-bulk |

### Output

```
comparison/
├── overlap_matrix.tsv               # Pathway overlap stats
├── concordance_heatmap.svg          # Correlation heatmap
├── venn_diagrams.svg                # Shared pathways
├── pathway_consistency.tsv          # Per-pathway stats
├── comparison_summary.json          # Complete results
└── metadata_report.txt              # Metadata summary
```

### Backend Details

**Implementation**: `pathwaylens_core/comparison/engine.py`

**Process**:
1. **Input Parsing**: Parse all datasets with tool detection
2. **ID Normalization**: Convert to common gene IDs
3. **Background Harmonization**: Create union background
4. **Enrichment**: Run chosen method on each dataset
5. **Overlap Analysis**: Calculate Jaccard, overlap coefficients
6. **Concordance**: Compare effect directions and magnitudes
7. **Meta-Analysis**: Combine p-values using Fisher's method

---

## visualize Command

Generate publication-quality figures.

```bash
pathwaylens visualize RESULTS_DIR [OPTIONS]
```

### Options

| Option | Type | Description |
|--------|------|-------------|
| `--type` | TEXT | Plot types (barplot, volcano, network, heatmap) |
| `--format` | TEXT | Output format (svg, pdf, png) |
| `--theme` | TEXT | Color theme (colorblind, nature, science) |
| `--top-pathways` | INT | Number of top pathways to show |

### Available Plot Types

- `barplot`: Enrichment barplot with error bars
- `volcano`: Volcano plot (enrichment vs significance)
- `network`: Pathway interaction network
- `heatmap`: Pathway-gene heatmap
- `dotplot`: Dot plot (size=genes, color=p-value)
- `diagnostic`: QC diagnostic panel

---

## workflow Command

Run complete analysis workflows.

```bash
pathwaylens workflow RUN WORKFLOW_FILE [OPTIONS]
```

**Example Workflow** (YAML):
```yaml
name: bulk_rnaseq_analysis
steps:
  - normalize:
      input: raw_genes.txt
      species: human
      output: normalized_genes.txt
  
  - analyze:
      method: ora
      input: normalized_genes.txt
      databases: [kegg, reactome]
      output: enrichment_results/
  
  - visualize:
      input: enrichment_results/
      types: [barplot, network]
```

---

For more details, see technical documentation in `docs/technical/`.
