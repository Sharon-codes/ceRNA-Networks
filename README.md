# ceRNA Networks: Empirical Audit of Network Graph Binarization

This repository contains the complete empirical evaluation pipeline and software suite demonstrating that threshold-based graph binarization in transcriptomic networks converts continuous, non-structural technical batch noise into discrete, catastrophic topological confounding.

## 🧬 Repository Structure

- `main_pipeline.py`: Master CLI runner executing all 6 empirical tests.
- `data_loader.py`: Log2-CPM normalization, GEO data ingestion, and batch shift simulation.
- `profiler.py`: Computational profiling, memory scaling, and runtime ETA estimation.
- `graph_utils.py`: Pearson correlation affinity matrices, binarization, Betti_0, Louvain community count & modularity $Q$, exact Graph Edit Distance (GED), and Spectral Distance on Normalized Laplacians.
- `tda_utils.py`: Persistent homology (Vietoris-Rips) and 1D Wasserstein distance stability evaluation.
- `gnn_utils.py`: PyTorch Graph Convolutional Network (GCN) classifier and AUROC collapse evaluator.
- `export_to_arghhhh.py`: Standalone visualizer exporting 300 DPI PNG images and PDF figures to `./arghhhh/`.
- `arghhhh/`: Directory containing all high-resolution PNG images and vector PDF plots for all 6 tests.
- `results/`: Directory containing raw CSV metrics (`./results/data/`) and figures (`./results/figures/`).

## 📊 Summary of Empirical Tests

1. **Test 1: Incremental Gaussian Perturbation on Empirical Affinity**
2. **Test 2: Direct Topological Distance Tracking (GED & Spectral Distance)**
3. **Test 3: The Boundary Density Audit**
4. **Test 4: Structural Sensitivity vs. Network Sparsity Sweep**
5. **Test 5: Continuous vs. Discrete TDA Filtration Stability**
6. **Test 6: GNN Generalization Collapse Analysis**

## 🚀 Quickstart

```bash
# Run pipeline on default benchmark transcriptomic data
python main_pipeline.py

# Run pipeline on custom GEO expression and metadata files
python main_pipeline.py --expr /path/to/expression.csv --meta /path/to/metadata.csv

# Export high-res images to arghhhh folder
python export_to_arghhhh.py
```
