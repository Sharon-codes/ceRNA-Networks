# Auditing Topological Batch Leakage in Multi-Cohort Graph Machine Learning: A Methodological Framework for ceRNA Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Sharon-codes/ceRNA-Networks/graphs/commit-activity)

This repository contains the official implementation, auditing framework, and reproduction suite for the research paper: **"Auditing Topological Batch Leakage in Multi-Cohort Graph Machine Learning: A Methodological Framework for ceRNA Networks."**

## 📑 Overview

Graph-based machine learning for multi-omics frequently reports exceptionally high predictive performance. However, our research demonstrates that aggregating independent cohorts introduces profound methodological vulnerabilities by conflating sequencing platform artifacts with biological signals. 

While vector-based batch effects are well-characterized, their manifestation as structural confounders in patient-specific graph topology requires systematic auditing. Because network generation requires hard mathematical thresholding, minor baseline differences in sequencing depth are discretized, permanently altering macro-scale graph components (such as community modularity and homology groups).

**This framework introduces:**
* **A Dataset-Stratified Evaluation Pipeline:** To mathematically quantify the topological batch leakage across multiple architectures (XGBoost, GCNs).
* **A Continuous TDA Batch Diagnostic:** Utilizing Vietoris-Rips filtrations to identify graph invariants confounded by technical batch effects prior to model training.
* **Latent Alignment Benchmarking:** Proof that standard vector-space correction algorithms (ComBat, Harmony, DANN) fundamentally fail to rescue threshold-dependent graph topology.
* **The "Interpretability Trap" Auditing:** SHAP-based workflows demonstrating how models inadvertently prioritize imputation structures over underlying physiological signatures.

---

## 🚀 Key Findings Summary

| Metric / Experiment | Protocol 1 (Dataset-Naive) | Protocol 2 (Dataset-Stratified) | Methodological Insight |
| :--- | :--- | :--- | :--- |
| **XGBoost (Hybrid) AUROC** | **0.926** | **0.760** | Evaluative performance gap ($\Delta \approx 0.16$) exposes massive topological batch leakage. |
| **GCN (Latent) AUROC** | **0.915** | **0.742** | Graph Neural Networks inherently propagate batch effects through message-passing neighborhoods. |
| **Topological Batch Sensitivity** | N/A | **64.7%** | Over half of extracted structural/TDA features are significantly confounded by sequencing origin ($p < 10^{-5}$). |
| **Batch Mitigation (Harmony/DANN)** | N/A | **Failed (<0.60)** | Latent alignment tools optimized for vectors fail to rescue discretized graph topology. |

---

## 💻 Detailed Codebase Breakdown

### Data Management (`data/`)
* `download_dbs.py`: Downloads external interaction databases (miRTarBase, CircInteractome) necessary for inferring ceRNA network edges.
* `load_geo.py`: Connects to GEO, downloads raw expression matrices for the training meta-cohort (GSE73002, GSE115513, GSE126094, GSE101684) and the independent external clinical validation cohort (GSE172232).

### Network Construction (`network/`)
* `build_cerna.py`: Constructs personalized, weighted ceRNA tripartite graphs for each patient utilizing a strict 90th-percentile transcriptomic threshold. Output is cached in `graphs.pkl`.

### Feature Engineering & TDA (`features/`)
* `extract_topology.py`: Runs `ripser` and `persim` on the constructed networks to generate persistence diagrams, extracting Betti numbers ($\beta_0, \beta_1$), persistent entropy ($H_p$), and discrete descriptors (modularity, community count, diameter).
* `visualize_network.py`: Helper script for visualizing localized patient ceRNA subgraphs.

### Machine Learning & Benchmarking (`models/`)
* `classify_nested_cv.py` & `classify.py`: Implements XGBoost models with nested cross-validation protocols and inverse-frequency sample weighting.
* `train_gcn.py`: Implements the GraphSAGE Graph Convolutional Network operating directly on raw adjacency matrices.
* `latent_alignment_benchmarks.py`: Benchmarking scripts applying ComBat-Seq, Harmony, and DANN to evaluate topological rescue.
* `sensitivity_analysis_topology.py`: Non-parametric Kruskal-Wallis diagnostic for batch-effect confounding across domains.

### Orchestration & Validation (Root Directory)
* `run_pipeline.py`: The master orchestration script handling stage-by-stage execution.
* `robust_validation.py`: Executes the Stratified Nested CV, separating folds by GEO dataset to prevent platform leakage.
* `lodo_validation.py`: Conducts formal Leave-One-Dataset-Out external validation against the $N=452$ hold-out.

### Figure Generation (`scripts/reproduction/`)
* `generate_figures.py`: Self-contained script utilizing `seaborn`, `sklearn`, `umap-learn`, and `shap` to output the exact high-resolution vector figures used in the manuscript (Boxplots, ROC curves, Harmony UMAP projections, and SHAP Waterfall plots).

---

## 🛠️ Step-by-Step Reproduction Guide

### 1. Prerequisites
Ensure you have Python 3.9+ and the following computational libraries installed:
```bash
pip install -r requirements.txt
