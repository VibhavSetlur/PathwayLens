#!/bin/bash
# Comprehensive test data generation for PathwayLens
# Covers all omic types with their specific data types

set -e

echo "🧬 Creating comprehensive PathwayLens test data..."
echo ""

# Clean and create directory structure
rm -rf tests/data
mkdir -p tests/data/{transcriptomics,proteomics,epigenomics,metabolomics}

# ============================================
# TRANSCRIPTOMICS
# ============================================
echo "Creating transcriptomics test data..."

# Bulk RNA-seq (DESeq2)
mkdir -p tests/data/transcriptomics/bulk
cat > tests/data/transcriptomics/bulk/deseq2_results.csv << 'EOF'
gene,baseMean,log2FoldChange,lfcSE,stat,pvalue,padj
TP53,1523.45,2.34,0.12,19.5,1.2e-85,3.4e-83
BRCA1,892.12,-1.87,0.15,-12.47,2.3e-35,4.5e-33
EGFR,2341.67,1.92,0.11,17.45,5.6e-68,1.2e-65
MYC,1876.23,3.12,0.14,22.29,1.1e-109,5.2e-107
PTEN,654.89,-2.45,0.18,-13.61,3.4e-42,8.9e-40
AKT1,1234.56,1.67,0.13,12.85,7.8e-38,1.6e-35
PIK3CA,987.34,2.01,0.16,12.56,4.5e-36,9.2e-34
KRAS,1543.21,1.45,0.14,10.36,3.2e-25,5.4e-23
NRAS,876.54,1.23,0.17,7.24,4.5e-13,6.7e-11
BRAF,1098.76,2.67,0.15,17.8,1.2e-70,2.8e-68
RB1,765.43,-1.98,0.19,-10.42,2.1e-25,3.6e-23
CDKN2A,543.21,-2.34,0.21,-11.14,8.9e-29,1.7e-26
ATM,1432.10,1.56,0.12,13.0,2.3e-38,4.8e-36
CHEK2,698.45,1.34,0.16,8.38,5.4e-17,9.8e-15
VEGFA,2134.56,2.89,0.13,22.23,1.5e-109,7.1e-107
TNF,1654.32,3.45,0.17,20.29,2.3e-91,6.7e-89
IL6,1432.10,2.98,0.15,19.87,8.9e-88,2.3e-85
STAT3,1234.56,2.12,0.13,16.31,1.2e-59,2.8e-57
JUN,876.54,1.67,0.15,11.13,8.7e-29,1.6e-26
FOS,765.43,1.98,0.16,12.38,3.4e-35,6.9e-33
EOF

# Single-cell RNA-seq (Seurat markers)
mkdir -p tests/data/transcriptomics/singlecell
cat > tests/data/transcriptomics/singlecell/seurat_markers.csv << 'EOF'
gene,p_val,avg_log2FC,pct.1,pct.2,p_val_adj,cluster
CD3D,0,2.45,0.95,0.12,0,T_cells
CD4,0,1.87,0.89,0.08,0,T_cells
CD8A,1.2e-250,2.12,0.78,0.05,3.4e-247,T_cells
IL7R,2.3e-180,1.67,0.72,0.11,5.6e-177,T_cells
CD79A,0,3.45,0.98,0.03,0,B_cells
MS4A1,0,2.98,0.94,0.04,0,B_cells
CD14,0,4.12,0.99,0.02,0,Monocytes
LYZ,0,3.67,0.97,0.06,0,Monocytes
FCGR3A,1.5e-300,2.34,0.85,0.03,4.2e-297,NK_cells
GNLY,2.1e-280,3.21,0.91,0.04,6.7e-277,NK_cells
EOF

# Spatial transcriptomics
mkdir -p tests/data/transcriptomics/spatial
cat > tests/data/transcriptomics/spatial/spatial_genes.csv << 'EOF'
gene,spatial_score,pvalue,padj,region
EPCAM,4.56,1.2e-45,3.4e-43,epithelial
KRT8,3.89,2.3e-38,5.6e-36,epithelial
KRT18,4.12,5.6e-42,1.2e-39,epithelial
CDH1,3.67,7.8e-35,1.6e-32,epithelial
VIM,4.12,5.6e-42,1.2e-39,stromal
COL1A1,3.67,7.8e-35,1.6e-32,stromal
COL1A2,3.98,6.7e-40,1.4e-37,stromal
FN1,4.23,3.4e-44,8.9e-42,stromal
CD45,4.23,3.4e-44,8.9e-42,immune
PTPRC,3.98,6.7e-40,1.4e-37,immune
CD3D,4.56,1.2e-45,3.4e-43,immune
CD8A,3.89,2.3e-38,5.6e-36,immune
EOF

# Time-series
mkdir -p tests/data/transcriptomics/timeseries
cat > tests/data/transcriptomics/timeseries/timecourse_genes.csv << 'EOF'
gene,timepoint,log2FC,pvalue,padj
MYC,0h,0,1,1
MYC,2h,1.23,0.001,0.01
MYC,6h,2.45,1e-8,1e-6
MYC,12h,1.87,1e-5,1e-3
FOS,0h,0,1,1
FOS,2h,2.12,1e-10,1e-8
FOS,6h,1.67,1e-6,1e-4
FOS,12h,0.89,0.01,0.05
EOF

# ============================================
# PROTEOMICS
# ============================================
echo "Creating proteomics test data..."

# Shotgun proteomics (MaxQuant)
mkdir -p tests/data/proteomics/shotgun
cat > tests/data/proteomics/shotgun/maxquant_proteinGroups.txt << 'EOF'
Protein IDs	Gene names	Peptides	Razor + unique peptides	Unique peptides	Sequence coverage [%]	Mol. weight [kDa]	Intensity	LFQ intensity	MS/MS count
P04637	TP53	12	12	10	45.3	43.7	2.34e8	1.87e8	45
P38398	BRCA1	18	18	15	38.2	207.7	1.23e8	9.87e7	67
P00533	EGFR	25	25	22	52.1	134.3	3.45e8	2.98e8	89
P01106	MYC	8	8	7	28.9	48.8	1.98e8	1.65e8	34
P60484	PTEN	14	14	12	41.7	47.2	8.76e7	7.23e7	42
P31749	AKT1	16	16	14	48.3	55.7	2.12e8	1.78e8	56
P42336	PIK3CA	22	22	19	44.6	124.3	1.67e8	1.43e8	71
P01116	KRAS	9	9	8	35.2	21.7	1.45e8	1.21e8	38
P01111	NRAS	7	7	6	32.1	21.3	9.87e7	8.34e7	29
P15056	BRAF	19	19	17	39.8	84.4	2.34e8	1.98e8	63
EOF

# Targeted proteomics (SRM/MRM)
mkdir -p tests/data/proteomics/targeted
cat > tests/data/proteomics/targeted/srm_results.csv << 'EOF'
protein,peptide,transition,area,concentration,cv
TP53,DLGEYFTLQIR,y7,1.23e6,45.6,5.2
TP53,DLGEYFTLQIR,y8,2.34e6,45.6,5.2
BRCA1,VFESIQK,y5,8.76e5,32.1,7.8
BRCA1,VFESIQK,y6,1.45e6,32.1,7.8
EGFR,GSTAENAEYLR,y8,3.21e6,67.8,4.5
EGFR,GSTAENAEYLR,y9,4.56e6,67.8,4.5
EOF

# DIA proteomics
mkdir -p tests/data/proteomics/dia
cat > tests/data/proteomics/dia/dia_proteins.csv << 'EOF'
protein,gene,intensity,qvalue,pg_qvalue
P04637,TP53,2.34e8,0.001,0.005
P38398,BRCA1,1.23e8,0.002,0.008
P00533,EGFR,3.45e8,0.0005,0.002
P01106,MYC,1.98e8,0.003,0.01
P60484,PTEN,8.76e7,0.004,0.015
EOF

# ============================================
# EPIGENOMICS
# ============================================
echo "Creating epigenomics test data..."

# ATAC-seq
mkdir -p tests/data/epigenomics/atacseq
cat > tests/data/epigenomics/atacseq/atac_peaks.bed << 'EOF'
chr1	1000000	1000500	peak_1	100	.	TP53,BRCA1
chr1	2000000	2000800	peak_2	150	.	EGFR,MYC
chr2	3000000	3000600	peak_3	120	.	PTEN,AKT1
chr3	4000000	4000700	peak_4	180	.	KRAS,BRAF
chr4	5000000	5000550	peak_5	95	.	PIK3CA,NRAS
chr5	6000000	6000650	peak_6	110	.	RB1,CDKN2A
chr6	7000000	7000700	peak_7	130	.	ATM,CHEK2
chr7	8000000	8000800	peak_8	140	.	VEGFA,TNF
chr8	9000000	9000600	peak_9	125	.	IL6,STAT3
chr9	10000000	10000750	peak_10	160	.	JUN,FOS
EOF

# ChIP-seq
mkdir -p tests/data/epigenomics/chipseq
cat > tests/data/epigenomics/chipseq/chip_peaks.bed << 'EOF'
chr1	1500000	1500800	H3K27ac_peak1	200	.	TP53,MYC
chr2	2500000	2500900	H3K27ac_peak2	180	.	EGFR,VEGFA
chr3	3500000	3500700	H3K27ac_peak3	150	.	STAT3,JUN
chr4	4500000	4500650	H3K27ac_peak4	220	.	FOS,TNF
chr5	5500000	5500750	H3K27ac_peak5	190	.	IL6,KRAS
chr6	6500000	6500800	H3K27ac_peak6	170	.	BRAF,PTEN
chr7	7500000	7500700	H3K27ac_peak7	210	.	AKT1,PIK3CA
EOF

# Methylation
mkdir -p tests/data/epigenomics/methyl
cat > tests/data/epigenomics/methyl/methylation_dmr.csv << 'EOF'
gene,chr,start,end,meth_diff,pvalue,padj
TP53,chr17,7571720,7590868,0.45,1.2e-15,3.4e-13
BRCA1,chr17,43044295,43125483,0.38,2.3e-12,5.6e-10
CDKN2A,chr9,21967752,21995301,-0.52,5.6e-18,1.2e-15
MLH1,chr3,36993333,37050918,-0.48,7.8e-16,1.6e-13
VHL,chr3,10183318,10195354,0.42,3.4e-14,8.9e-12
EOF

# ============================================
# METABOLOMICS
# ============================================
echo "Creating metabolomics test data..."

# Targeted metabolomics
mkdir -p tests/data/metabolomics/targeted
cat > tests/data/metabolomics/targeted/targeted_metabolites.csv << 'EOF'
metabolite,hmdb_id,concentration,fold_change,pvalue,padj
Glucose,HMDB0000122,5.2,1.45,0.001,0.01
Pyruvate,HMDB0000243,2.3,2.12,1e-5,1e-3
Lactate,HMDB0000190,8.7,1.87,1e-4,0.005
Citrate,HMDB0000094,3.4,0.67,0.002,0.02
Succinate,HMDB0000254,1.9,1.34,0.005,0.03
Fumarate,HMDB0000134,1.2,1.23,0.01,0.05
Malate,HMDB0000156,2.1,0.78,0.003,0.025
ATP,HMDB0000538,4.5,1.56,1e-6,1e-4
ADP,HMDB0001341,3.2,0.89,0.008,0.04
Glutamine,HMDB0000641,6.8,1.98,1e-7,1e-5
EOF

# Untargeted metabolomics
mkdir -p tests/data/metabolomics/untargeted
cat > tests/data/metabolomics/untargeted/untargeted_features.csv << 'EOF'
feature_id,mz,rt,intensity,fold_change,pvalue,padj,putative_id
M180T45,180.0634,45.2,1.23e6,2.34,1e-8,1e-6,Glucose
M89T120,89.0244,120.5,8.76e5,1.87,1e-5,1e-3,Lactate
M192T78,192.0270,78.3,2.34e6,3.12,1e-10,1e-8,Citrate
M147T95,147.0532,95.1,1.45e6,1.67,1e-6,1e-4,Glutamine
M507T210,507.0000,210.8,5.67e5,2.45,1e-7,1e-5,ATP
EOF

# Lipidomics
mkdir -p tests/data/metabolomics/lipidomics
cat > tests/data/metabolomics/lipidomics/lipid_species.csv << 'EOF'
lipid_name,class,mz,intensity,fold_change,pvalue,padj
PC(16:0/18:1),PC,760.5851,2.34e7,1.56,0.001,0.01
PE(18:0/20:4),PE,766.5387,1.23e7,2.12,1e-5,1e-3
SM(d18:1/16:0),SM,703.5754,8.76e6,0.78,0.005,0.03
Cer(d18:1/24:1),Cer,648.6281,5.43e6,1.87,1e-4,0.005
TAG(16:0/18:1/18:1),TAG,876.8012,3.21e7,1.45,0.002,0.02
EOF

# Flux analysis
mkdir -p tests/data/metabolomics/flux
cat > tests/data/metabolomics/flux/flux_rates.csv << 'EOF'
reaction,flux_rate,std_error,pvalue,pathway
Glycolysis_v1,12.5,0.8,0.001,Glycolysis
Glycolysis_v2,11.3,0.9,0.002,Glycolysis
TCA_v1,8.7,0.6,0.005,TCA_cycle
TCA_v2,7.9,0.7,0.008,TCA_cycle
PPP_v1,3.4,0.4,0.01,Pentose_phosphate
Oxidative_phos_v1,15.6,1.2,0.0005,Oxidative_phosphorylation
EOF

echo ""
echo "✅ Comprehensive test data created!"
echo ""
echo "Summary:"
echo "--------"
echo "Transcriptomics:"
echo "  - bulk: DESeq2 results"
echo "  - singlecell: Seurat markers"
echo "  - spatial: Spatial gene scores"
echo "  - timeseries: Time-course data"
echo ""
echo "Proteomics:"
echo "  - shotgun: MaxQuant protein groups"
echo "  - targeted: SRM/MRM results"
echo "  - dia: DIA protein quantification"
echo ""
echo "Epigenomics:"
echo "  - atacseq: ATAC-seq peaks"
echo "  - chipseq: ChIP-seq peaks"
echo "  - methyl: Methylation DMRs"
echo ""
echo "Metabolomics:"
echo "  - targeted: Targeted metabolite quantification"
echo "  - untargeted: Untargeted metabolomics features"
echo "  - lipidomics: Lipid species"
echo "  - flux: Metabolic flux rates"
echo ""
echo "File sizes:"
du -sh tests/data/*/*
