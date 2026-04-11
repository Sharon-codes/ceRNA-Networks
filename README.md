# ceRNA Early-Stage Cancer Detection Pipeline

This repository contains a computational framework for the early-stage detection of cancer using competitive endogenous RNA (ceRNA) network topology and expression analysis.

## Overview

The project aims to identify robust biomarkers for cancer by analyzing the relationships between circRNAs and miRNAs. By constructing regulatory networks and extracting topological features using Topological Data Analysis (TDA) and graph metrics, the pipeline detects systemic changes associated with malignancy across various cancer types (Colorectal, Lung, Breast, etc.).

## Project Structure

- `config/`: Configuration settings and metric logging utilities.
- `data/`: Modules for downloading and processing GEO datasets (GSE115513, GSE73002, etc.).
- `network/`: Logic for building ceRNA interaction networks based on circBase and miRBase.
- `features/`: Topological feature extraction (persistent homology, centrality, motifs).
- `models/`: Machine learning models (XGBoost) for binary and multi-class classification.
- `run_pipeline.py`: Main entry point to execute the full end-to-end pipeline.

## Implementation Progress

- [x] **Data Ingestion**: Automated retrieval of GEO soft files and supplementary matrices.
- [x] **Normalization**: CPM/RPM normalization of expression counts.
- [x] **Network Construction**: Competition-based linkage between circRNAs and miRNAs.
- [x] **TDA Features**: Extraction of persistent homology components (Betti numbers) and network centrality.
- [x] **Classification Pipeline**: Nested cross-validation for cancer vs. healthy and cancer type prediction.

## Known Issues

### 1. Zero AUROC for Stage-I Detection
A critical issue currently observed is that the **Stage-I AUROC reports as 0.0** in the evaluation logs.

**Analysis:**
- The issue stems from the metadata parsing logic in `data/load_geo.py`.
- Different GEO datasets use non-standardized keys (e.g., `characteristics_ch1`) for clinical staging.
- Samples are being labeled as "unknown" or "Healthy" incorrectly, or the Stage-I keyword is not being captured (e.g., "ajcc-stage: I" vs "tumor stage: 1").
- During evaluation in `models/classify.py`, the lack of correctly labeled Stage-I samples results in empty evaluation sets, producing zero metrics.

### 2. Missing Database Files
The pipeline requires external databases (miRBase, miRTarBase) which are currently filtered or partially downloaded. Full execution depends on the presence of `mature_all.fa` and interaction CSVs in the `data/raw/databases` directory.

## Current Setup

To run the pipeline, ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

Initialize the data download:
```bash
python data/download_dbs.py
python data/load_geo.py
```

Run the pipeline:
```bash
python run_pipeline.py
```

## Future Work
- Enhance `normalize_cancer_label` and `normalize_stage` with more robust regex patterns.
- Implement data augmentation for minority cancer stages to balance cross-validation folds.
- Optimize TDA computation for larger ceRNA networks.
