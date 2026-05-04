# Integrating ceRNA Network Topology with Expression Features for Pan-Cancer Detection

This repository contains the official implementation and reproduction suite for the research paper focusing on robust ceRNA network-based cancer detection.

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

### Workflow
The framework follows a 6-stage process as visualized in `figures/figure1_workflow.png`:
1.  **Data Acquisition**: Automated retrieval of 4 GEO datasets (Breast, CRC, Prostate, Healthy).
2.  **Network Construction**: Tripartite graph building using miRTarBase and CircInteractome.
3.  **Feature Extraction**: 21 hybrid features (4 expression-based, 17 topological/TDA).
4.  **Batch Correction**: ComBat harmonisation for raw expression levels.
5.  **Robust Classification**: Nested CV using XGBoost with Optuna optimization.
6.  **Interpretability**: SHAP value analysis with batch-sensitivity diagnostic starring ($\star$).

---

## 🛠️ Step-by-Step Reproduction Guide

### 1. Prerequisites
Ensure you have Python 3.9+ and the following computational libraries installed:
```bash
pip install -r requirements.txt
```
*Note: Topological Data Analysis (TDA) requires `ripser` and `persim`. On Windows, you may need the [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/).*

### 2. Data Preparation
Run the automated pipeline to download databases and preprocess the meta-cohort:
```bash
# Downloads miRTarBase and CircInteractome mappings
python data/download_dbs.py

# Retrieves and filters raw GEO datasets (GSE73002, GSE115513, etc.)
python data/load_geo.py
```

### 3. Feature Extraction & Harmonisation
Construct the ceRNA networks and compute topological features.
```bash
# This stage performs graph construction and TDA for 2,000+ patients
python run_pipeline.py --stage features
```
*The resulting harmonised feature matrix is saved at `features/feature_matrix_combat.csv`.*

### 4. Model Training & Validation
Execute the contrasted evaluation protocols and LODO (Leave-One-Dataset-Out) validation:
```bash
# Run Protocol 1 and Protocol 2 Nested CV Comparison
python robust_validation.py

# Run LODO External Validation for each dataset
python lodo_validation.py
```

### 5. Generating Manuscript Figures
To reproduce the visualizations exactly as they appear in the paper, run the following:

| Figure | Reproduction Script | Key Insight |
| :--- | :--- | :--- |
| **Fig 1** | `scripts/reproduction/generate_final_figure1.py` | Methodological Workflow Diagram |
| **Fig 2** | `scripts/reproduction/generate_final_figure2.py` | Subgraph Hub Dissolution (Healthy vs Stage I) |
| **Fig 3** | `scripts/reproduction/generate_final_figure3.py` | Distributional Shift Diagnostic (Batch Effects) |
| **Fig 4** | `scripts/reproduction/generate_final_figure4.py` | Performance Gap (Naive vs Stratified) |
| **Fig 5** | `scripts/reproduction/generate_final_figure5.py` | Per-Cancer ROC Curves (Protocol 1) |
| **Fig 6** | `scripts/reproduction/generate_final_figure6.py` | SHAP Importance & Sensitivity Starring |
| **Fig 7** | `scripts/reproduction/generate_final_figure7.py` | Robustness Drop & Linear Baselines |

---

## 📜 Key Findings Summary

| Experiment | Naive CV (P1) | Stratified CV (P2) | Insight |
| :--- | :---: | :---: | :--- |
| **AUROC (Hybrid)** | 0.927 | 0.45 -- 0.76 | Performance drops significantly when platform signals are hidden. |
| **Batch Sensitivity** | 0% | 57.1% | 12/21 features are technically confounded, not just biological. |
| **Robustness Check** | 0.999 | 0.978 (Filtered) | Removing batch features drops performance by ~2% even in Naive settings. |

---

## 📂 Repository Structure
- `config/`: Centralized settings (`config.py`) and seeds.
- `data/`: GEO retrieval and preprocessing scripts.
- `network/`: Logic for building tripartite ceRNA graphs.
- `features/`: TDA (Ripser) and graph-theoretic extraction tools.
- `models/`: Implementations for XGBoost, Logistic Regression, and MLP.
- `scripts/`: Reproduction and sensitivity analysis tools.
- `figures/`: High-resolution PNGs for the manuscript.

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
