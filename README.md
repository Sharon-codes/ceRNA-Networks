# Auditing Topological Batch Leakage in Multi-Cohort Graph Machine Learning: A Methodological Framework for ceRNA Networks

This repository contains the official implementation and reproduction suite for the research paper focusing on robust ceRNA network-based cancer detection.

## 🚀 Current Project Status & Work Completed So Far

Extensive work has been completed to establish an end-to-end machine learning and data processing pipeline. Here are the core accomplishments currently reflected in the codebase:

- **Automated Data Retrieval & Preprocessing**: 
  - Scripts implemented to download raw GEO datasets (GSE73002, GSE115513, etc.) and interaction databases (miRTarBase, CircInteractome).
- **ceRNA Graph Construction**: 
  - Implementation of tripartite graph building (mRNA-miRNA-circRNA/lncRNA) for over 2,000 patient samples, serialized into a massive `graphs.pkl` object.
- **Topological Data Analysis (TDA) & Feature Extraction**: 
  - Graph-theoretic features (Betti numbers, persistence entropy, sub-graph hubs) extracted successfully and exported as `feature_matrix.csv`.
- **Batch Effect Mitigation**: 
  - Applied ComBat harmonization to rectify platform-specific technical batch effects, yielding `feature_matrix_combat.csv`.
- **Comprehensive Modeling & Validation Protocols**:
  - Implemented Protocol 1 (Dataset-Naive Nested CV) and Protocol 2 (Dataset-Stratified CV) using XGBoost.
  - Implemented Elastic Net classifiers to benchmark full expression profiles vs. topology-based features.
  - Leave-One-Dataset-Out (LODO) validation pipelines finalized.
- **Interpretability & Sensitivity Analysis**: 
  - SHAP value calculations complete.
  - Topology sensitivity diagnostic implemented to flag features confounded by technical artifacts.
- **Manuscript Figures Reproduction**: 
  - Automated python scripts configured to accurately reproduce Figures 1 through 8 directly from the processed data.

---

## 💻 Detailed Codebase Breakdown

Below is a granular view of the code written and what each component does.

### Data Management (`data/`)
- `download_dbs.py`: Downloads external interaction databases (miRTarBase, CircInteractome) necessary for inferring the ceRNA network edges.
- `load_geo.py`: Connects to GEO, downloads raw expression matrices, and standardizes cohort metadata.

### Network Construction (`network/`)
- `build_cerna.py`: The core script that utilizes the interaction databases and patient expression data to build personalized, weighted ceRNA tripartite graphs for each patient. Output is cached in `graphs.pkl`.

### Feature Engineering & TDA (`features/`)
- `extract_topology.py`: Runs Ripser and persim on the constructed networks to generate persistence diagrams, extracting Betti numbers, persistent entropy, and classic node/edge density statistics.
- `visualize_network.py`: Helper script for visualizing localized patient ceRNA subgraphs.

### Machine Learning & Benchmarking (`models/`)
- `classify_nested_cv.py` & `classify.py`: Implements XGBoost models with nested cross-validation protocols.
- `train_full_expression_elasticnet.py` & `lodo_full_expression_elasticnet.py`: Baseline benchmarking scripts applying Elastic Net linear classifiers over raw expression profiles to contrast with our topology-based approaches.
- `sensitivity_analysis.py`: Statistical diagnostics for batch-effect confounding.
- `plot_pr_curves.py`: Evaluates precision-recall metrics.

### Orchestration & Validation (Root Directory)
- `run_pipeline.py`: The master orchestration script handling stage-by-stage execution (e.g., `python run_pipeline.py --stage features`).
- `robust_validation.py`: Executes the Stratified Nested CV, separating folds by GEO dataset to prevent platform leakage.
- `lodo_validation.py`: Conducts formal Leave-One-Dataset-Out external validation.
- `shap_ablation.py`: Performs model ablation studies using SHAP feature importance calculations.
- `sensitivity_analysis_topology.py`: Analyzes the distributional shift of topological invariants across GEO cohorts.

### Figure Generation (`scripts/reproduction/`)
- `generate_final_figure1.py` to `generate_final_figure8.py`: Self-contained scripts that read from `/models/` and `/features/` to output the exact high-resolution vector figures used in the manuscript (workflow diagram, hub dissolution, distributional shifts, performance gaps, SHAP starring, robustness baselines).

---

## 📑 Overview
Liquid biopsy-based cancer detection using competitive endogenous RNA (ceRNA) networks has shown high promise. However, our research demonstrates that traditional dataset-naive evaluation protocols significantly inflate performance estimates by failing to account for platform-specific batch effects in network topology. 

This framework introduces:
1.  **A Dataset-Stratified Evaluation Pipeline**: To measure true cross-platform generalizability.
2.  **A Topology Sensitivity Diagnostic**: A statistical framework to identify graph features confounded by technical batch effects.
3.  **A Hybrid TDA-Expression Model**: Integrating Topological Data Analysis (Betti numbers, persistence entropy) with classical expression metrics.

---

## 🔬 Methodology & Evaluation Protocols

### The Two Evaluation Protocols
*   **Protocol 1 (Dataset-Naive CV)**: Samples from all cohorts are pooled and randomly shuffled. This is the standard practice in existing literature but is prone to "platform leakage" where the model learns to identify the sequencing center rather than the biology.
*   **Protocol 2 (Dataset-Stratified CV)**: Folds are constructed such that no samples from the same GEO dataset appear in both training and validation sets. This provides a realistic estimate of clinical utility in a multi-center setting.

---

## 🛠️ Step-by-Step Reproduction Guide

### 1. Prerequisites
Ensure you have Python 3.9+ and the following computational libraries installed:
```bash
pip install -r requirements.txt
```
*Note: Topological Data Analysis (TDA) requires `ripser` and `persim`.*

### 2. Data Preparation
Run the automated pipeline to download databases and preprocess the meta-cohort:
```bash
python data/download_dbs.py
python data/load_geo.py
```

### 3. Feature Extraction & Harmonisation
Construct the ceRNA networks and compute topological features.
```bash
python run_pipeline.py --stage features
```
*The resulting harmonised feature matrix is saved at `features/feature_matrix_combat.csv`.*

### 4. Model Training & Validation
Execute the contrasted evaluation protocols and LODO validation:
```bash
python robust_validation.py
python lodo_validation.py
```

### 5. Generating Manuscript Figures
Run the dedicated scripts in `scripts/reproduction/` to generate the visual artifacts for the manuscript.
```bash
python scripts/reproduction/generate_final_figure4.py
```

---

## 📜 Key Findings Summary

| Experiment | Naive CV (P1) | Stratified CV (P2) | Insight |
| :--- | :---: | :---: | :--- |
| **AUROC (Hybrid)** | 0.927 | 0.45 -- 0.76 | Performance drops significantly when platform signals are hidden. |
| **Batch Sensitivity** | 0% | 57.1% | 12/21 features are technically confounded, not just biological. |
| **Robustness Check** | 0.999 | 0.978 (Filtered) | Removing batch features drops performance by ~2% even in Naive settings. |

---

## 📖 Citation
If you use this pipeline or the topology sensitivity diagnostic in your research, please cite:
```bibtex
@article{sharon2026cerna,
  title={Dataset-Stratified Evaluation Reveals the Impact of Platform Batch Effects on ceRNA Network-Based Cancer Classifiers},
  author={Sharon, et al.},
  journal={Journal of Biomedical Informatics},
  year={2026}
}
```

---
**License**: MIT | **Author**: [Sharon-codes](https://github.com/Sharon-codes)
