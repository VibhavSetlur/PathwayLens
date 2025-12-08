#!/bin/bash
# Comprehensive PathwayLens Test Suite
# Tests all omic types with their specific data types

set -e

echo "🧬 PathwayLens Comprehensive Test Suite"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Clean outputs
rm -rf tests/outputs/*
mkdir -p tests/outputs

test_num=1

# ============================================
# TRANSCRIPTOMICS TESTS
# ============================================
echo -e "${YELLOW}=== TRANSCRIPTOMICS TESTS ===${NC}"
echo ""

echo -e "${BLUE}Test $test_num: Transcriptomics - Bulk RNA-seq (DESeq2)${NC}"
pathwaylens analyze ora \
    --input tests/data/transcriptomics/bulk/deseq2_results.csv \
    --omic-type transcriptomics \
    --data-type bulk \
    --tool deseq2 \
    --databases kegg \
    --species human \
    --output-dir tests/outputs/$(printf "%02d" $test_num)_transcriptomics_bulk_deseq2 \
    --run-name bulk_rnaseq_deseq2 \
    --min-genes 3
echo -e "${GREEN}✓ Test $test_num Complete${NC}\n"
((test_num++))

echo -e "${BLUE}Test $test_num: Transcriptomics - Single-cell (Seurat)${NC}"
pathwaylens analyze ora \
    --input tests/data/transcriptomics/singlecell/seurat_markers.csv \
    --omic-type transcriptomics \
    --data-type singlecell \
    --tool auto \
    --databases kegg \
    --species human \
    --output-dir tests/outputs/$(printf "%02d" $test_num)_transcriptomics_singlecell \
    --run-name scrna_seurat_markers \
    --min-genes 3
echo -e "${GREEN}✓ Test $test_num Complete${NC}\n"
((test_num++))

echo -e "${BLUE}Test $test_num: Transcriptomics - Spatial${NC}"
pathwaylens analyze ora \
    --input tests/data/transcriptomics/spatial/spatial_genes.csv \
    --omic-type transcriptomics \
    --data-type spatial \
    --tool auto \
    --databases kegg \
    --species human \
    --output-dir tests/outputs/$(printf "%02d" $test_num)_transcriptomics_spatial \
    --min-genes 3
echo -e "${GREEN}✓ Test $test_num Complete${NC}\n"
((test_num++))

# ============================================
# PROTEOMICS TESTS
# ============================================
echo -e "${YELLOW}=== PROTEOMICS TESTS ===${NC}"
echo ""

echo -e "${BLUE}Test $test_num: Proteomics - Shotgun (MaxQuant)${NC}"
pathwaylens analyze ora \
    --input tests/data/proteomics/shotgun/maxquant_proteinGroups.txt \
    --omic-type proteomics \
    --data-type shotgun \
    --tool maxquant \
    --databases kegg \
    --species human \
    --output-dir tests/outputs/$(printf "%02d" $test_num)_proteomics_shotgun_maxquant \
    --run-name shotgun_proteomics \
    --min-genes 5
echo -e "${GREEN}✓ Test $test_num Complete${NC}\n"
((test_num++))

echo -e "${BLUE}Test $test_num: Proteomics - Targeted (SRM/MRM)${NC}"
pathwaylens analyze ora \
    --input tests/data/proteomics/targeted/srm_results.csv \
    --omic-type proteomics \
    --data-type targeted \
    --tool auto \
    --databases kegg \
    --species human \
    --output-dir tests/outputs/$(printf "%02d" $test_num)_proteomics_targeted_srm \
    --min-genes 3
echo -e "${GREEN}✓ Test $test_num Complete${NC}\n"
((test_num++))

echo -e "${BLUE}Test $test_num: Proteomics - DIA${NC}"
pathwaylens analyze ora \
    --input tests/data/proteomics/dia/dia_proteins.csv \
    --omic-type proteomics \
    --data-type dia \
    --tool auto \
    --databases kegg \
    --species human \
    --output-dir tests/outputs/$(printf "%02d" $test_num)_proteomics_dia \
    --min-genes 3
echo -e "${GREEN}✓ Test $test_num Complete${NC}\n"
((test_num++))

# ============================================
# EPIGENOMICS TESTS
# ============================================
echo -e "${YELLOW}=== EPIGENOMICS TESTS ===${NC}"
echo ""

echo -e "${BLUE}Test $test_num: Epigenomics - ATAC-seq${NC}"
pathwaylens analyze ora \
    --input tests/data/epigenomics/atacseq/atac_peaks.bed \
    --omic-type epigenomics \
    --data-type atacseq \
    --tool auto \
    --databases kegg \
    --species human \
    --output-dir tests/outputs/$(printf "%02d" $test_num)_epigenomics_atacseq \
    --min-genes 3
echo -e "${GREEN}✓ Test $test_num Complete${NC}\n"
((test_num++))

echo -e "${BLUE}Test $test_num: Epigenomics - ChIP-seq${NC}"
pathwaylens analyze ora \
    --input tests/data/epigenomics/chipseq/chip_peaks.bed \
    --omic-type epigenomics \
    --data-type chipseq \
    --tool auto \
    --databases kegg \
    --species human \
    --output-dir tests/outputs/$(printf "%02d" $test_num)_epigenomics_chipseq \
    --min-genes 3
echo -e "${GREEN}✓ Test $test_num Complete${NC}\n"
((test_num++))

echo -e "${BLUE}Test $test_num: Epigenomics - Methylation${NC}"
pathwaylens analyze ora \
    --input tests/data/epigenomics/methyl/methylation_dmr.csv \
    --omic-type epigenomics \
    --data-type methyl \
    --tool auto \
    --databases kegg \
    --species human \
    --output-dir tests/outputs/$(printf "%02d" $test_num)_epigenomics_methylation \
    --min-genes 3
echo -e "${GREEN}✓ Test $test_num Complete${NC}\n"
((test_num++))

# ============================================
# METABOLOMICS TESTS
# ============================================
echo -e "${YELLOW}=== METABOLOMICS TESTS ===${NC}"
echo ""

echo -e "${BLUE}Test $test_num: Metabolomics - Targeted${NC}"
pathwaylens analyze ora \
    --input tests/data/metabolomics/targeted/targeted_metabolites.csv \
    --omic-type metabolomics \
    --data-type targeted \
    --tool auto \
    --databases kegg \
    --species human \
    --output-dir tests/outputs/$(printf "%02d" $test_num)_metabolomics_targeted \
    --min-genes 3
echo -e "${GREEN}✓ Test $test_num Complete${NC}\n"
((test_num++))

echo -e "${BLUE}Test $test_num: Metabolomics - Untargeted${NC}"
pathwaylens analyze ora \
    --input tests/data/metabolomics/untargeted/untargeted_features.csv \
    --omic-type metabolomics \
    --data-type untargeted \
    --tool auto \
    --databases kegg \
    --species human \
    --output-dir tests/outputs/$(printf "%02d" $test_num)_metabolomics_untargeted \
    --min-genes 3
echo -e "${GREEN}✓ Test $test_num Complete${NC}\n"
((test_num++))

echo -e "${BLUE}Test $test_num: Metabolomics - Lipidomics${NC}"
pathwaylens analyze ora \
    --input tests/data/metabolomics/lipidomics/lipid_species.csv \
    --omic-type metabolomics \
    --data-type lipidomics \
    --tool auto \
    --databases kegg \
    --species human \
    --output-dir tests/outputs/$(printf "%02d" $test_num)_metabolomics_lipidomics \
    --min-genes 3
echo -e "${GREEN}✓ Test $test_num Complete${NC}\n"
((test_num++))

# ============================================
# COMPARISON TESTS
# ============================================
echo -e "${YELLOW}=== COMPARISON TESTS ===${NC}"
echo ""

echo -e "${BLUE}Test $test_num: Compare - Gene Lists (Transcriptomics)${NC}"
pathwaylens compare \
    --inputs tests/data/transcriptomics/bulk/deseq2_results.csv \
    --inputs tests/data/transcriptomics/singlecell/seurat_markers.csv \
    --labels "Bulk_RNA" \
    --labels "scRNA" \
    --mode genes \
    --omic-type transcriptomics \
    --data-type bulk \
    --output-dir tests/outputs/$(printf "%02d" $test_num)_compare_genes_transcriptomics
echo -e "${GREEN}✓ Test $test_num Complete${NC}\n"
((test_num++))

echo -e "${BLUE}Test $test_num: Compare - Pathways (Multi-condition)${NC}"
pathwaylens compare \
    --inputs tests/data/transcriptomics/bulk/deseq2_results.csv \
    --inputs tests/data/transcriptomics/singlecell/seurat_markers.csv \
    --inputs tests/data/transcriptomics/spatial/spatial_genes.csv \
    --labels "Bulk" "SingleCell" "Spatial" \
    --mode pathways \
    --omic-type transcriptomics \
    --data-type bulk \
    --databases kegg \
    --species human \
    --output-dir tests/outputs/$(printf "%02d" $test_num)_compare_pathways_multi
echo -e "${GREEN}✓ Test $test_num Complete${NC}\n"
((test_num++))

echo -e "${BLUE}Test $test_num: Compare - Cross-Omics (Transcriptomics vs Proteomics)${NC}"
pathwaylens compare \
    --inputs tests/data/transcriptomics/bulk/deseq2_results.csv \
    --inputs tests/data/proteomics/shotgun/maxquant_proteinGroups.txt \
    --labels "RNA" "Protein" \
    --mode genes \
    --omic-type transcriptomics \
    --data-type bulk \
    --output-dir tests/outputs/$(printf "%02d" $test_num)_compare_cross_omics
echo -e "${GREEN}✓ Test $test_num Complete${NC}\n"
((test_num++))

# ============================================
# SUMMARY
# ============================================
echo ""
echo "========================================"
echo "🎉 All Tests Complete!"
echo "========================================"
echo ""
echo "Test Summary:"
echo "-------------"
echo "Total tests run: $((test_num - 1))"
echo ""
echo "Outputs by category:"
echo ""
echo "Transcriptomics:"
ls -1 tests/outputs/ | grep transcriptomics | nl
echo ""
echo "Proteomics:"
ls -1 tests/outputs/ | grep proteomics | nl
echo ""
echo "Epigenomics:"
ls -1 tests/outputs/ | grep epigenomics | nl
echo ""
echo "Metabolomics:"
ls -1 tests/outputs/ | grep metabolomics | nl
echo ""
echo "Comparisons:"
ls -1 tests/outputs/ | grep compare | nl
echo ""
echo "Generated files:"
find tests/outputs -type f \( -name "*.html" -o -name "*.json" -o -name "*.csv" \) | wc -l
echo ""
echo "Total output size:"
du -sh tests/outputs
echo ""
echo "To view results:"
echo "  HTML visualizations: tests/outputs/*/database_comparison.html"
echo "  JSON summaries: tests/outputs/*/run.json"
echo "  CSV results: tests/outputs/*/results.csv"
