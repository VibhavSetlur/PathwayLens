# PathwayLens Benchmarks

Performance metrics for standard analysis tasks.

## Hardware Specifications
- **CPU**: (User to populate, e.g., Intel Core i9-9900K @ 3.60GHz)
- **RAM**: (User to populate, e.g., 32GB DDR4)
- **Storage**: (User to populate, e.g., NVMe SSD)
- **OS**: Linux (Ubuntu 22.04 LTS)

## Software Versions
- **Python**: 3.10+
- **Pandas**: 2.0+
- **NumPy**: 1.24+
- **PathwayLens**: 0.1.0

## Input Data
- **ORA**: Randomly generated gene lists (100 - 10,000 genes).
- **GSEA**: Ranked gene lists (10,000 - 20,000 genes).
- **Single-Cell**: Synthetic sparse matrices (5k - 100k cells, 20k genes).

## 1. ORA Analysis

| Dataset Size | Databases | Runtime | Memory |
|--------------|-----------|---------|--------|
| 100 genes    | KEGG      | ~2s     | <500MB |
| 1,000 genes  | KEGG, Reactome, GO | ~15s | ~1GB |
| 10,000 genes | All (8 DBs) | ~45s | ~2GB |

## 2. GSEA Analysis

| Dataset Size | Permutations | Runtime | Memory |
|--------------|--------------|---------|--------|
| 10,000 genes | 1000         | ~2m     | ~2GB   |
| 20,000 genes | 1000         | ~4m     | ~3GB   |

## 3. Single-Cell Scoring (scRNA-seq)

Using `mean_zscore` method on sparse inputs (`.h5ad`).

| Cells | Genes | Pathways | Runtime | Memory (Peak) |
|-------|-------|----------|---------|---------------|
| 5k    | 20k   | KEGG (300) | ~30s  | ~2GB          |
| 50k   | 20k   | KEGG (300) | ~3m   | ~8GB          |
| 100k  | 20k   | Reactome (2k) | ~15m | ~16GB         |

**Note:** Using sparse matrices (`.h5ad`) significantly reduces memory usage compared to dense CSVs.

## Reproducibility
To reproduce these benchmarks on your system, run:
```bash
python3 scripts/benchmarking/run_all.py
```
