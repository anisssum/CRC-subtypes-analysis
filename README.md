# CRC Subtypes Analysis

A bioinformatics pipeline for preprocessing TCGA colorectal cancer transcriptomic data, predicting Consensus Molecular Subtypes (CMS), and exploring their association with clinical characteristics and patient survival.

## Dataset

The analysis uses publicly available TCGA COAD/READ data downloaded from the UCSC Xena platform.

Input data include:

- RNA-seq gene expression (RSEM normalized, log2-transformed)
- clinical annotations
- survival data

## Workflow

```
Raw expression data
        │
        ▼
Quality control
        │
        ▼
Outlier removal
        │
        ▼
Selection of highly variable genes
        │
        ▼
Feature scaling
        │
        ▼
PCA / t-SNE / UMAP
        │
        ▼
Clinical data cleaning
        │
        ▼
CMS classification (CMScaller)
        │
        ▼
Clinical association analysis
        │
        ▼
Survival analysis
```

## Repository Structure

```
.
├── data/                  # Input datasets
├── figures/               # Generated figures
├── python/                # Python preprocessing scripts
├── R/                     # CMS prediction and survival analysis
├── expression_clean.csv
├── clinical_clean.csv
└── README.md
```

## Requirements

### Python

- pandas: 2.3.3
- numpy: 2.4.6
- scipy: 1.18.0
- sklearn: 1.9.0
- matplotlib: 3.11.0
- umap: 0.5.12 

### R

- CMScaller: 0.99.2
- tidyverse: 2.0.0

## Running the Analysis

### 1. Clone the repository

```bash
git clone https://github.com/anisssum/CRC-subtypes-analysis.git
cd CRC-subtypes-analysis
```

### 2. Download the TCGA datasets

Place the expression matrix and clinical tables into the appropriate data directory.

### 3. Run the preprocessing pipeline

```bash
python ./scripts/CRC_analysis.py
```

This step performs:

- quality control;
- filtering;
- dimensionality reduction;
- generation of cleaned expression and clinical datasets.

### 4. Predict CMS subtypes

Run the R script:

```r
Rscript ./scripts/CMS_classification.R
```

### 5. Perform downstream analyses

```bash
python ./scripts/CMS_analysis.py
```

- clinical association plots;
- statistical tests;

```bash
python ./scripts/KaplanMeier.py 
```

- Kaplan–Meier survival curves.

## Author

Anna Chesnokova
