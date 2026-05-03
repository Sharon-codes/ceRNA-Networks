# ceRNA Pan-Cancer Detection Pipeline

A computational framework for early-stage cancer detection using **Competitive Endogenous RNA (ceRNA)** network topology and high-dimensional expression analysis.

## 🚀 Overview

This repository implements a multi-stage pipeline to identify robust biomarkers for cancer by analyzing the regulatory relationships between **circRNAs** and **miRNAs**. It leverages **Topological Data Analysis (TDA)** and graph metrics to detect systemic changes in regulatory networks across various cancer types (Lung, Gastric, HCC, CRC, etc.).

### Key Research Findings

| Comparison | Valid? | Insight |
| :--- | :---: | :--- |
| **Hybrid (0.927) vs. Elastic Net (0.979)** | ❌ | Different datasets; Hybrid used smaller multi-omics overlap. |
| **Hybrid LODO (0.749) vs. EN LODO (0.971)** | ❌ | Different held-out sets; EN focused on high-sample binary tasks. |
| **Null Network Validation** | ✅ | Confirms topology signal stands independently of expression levels. |
| **Stage Stratification (GSE115513)** | ✅ | Validates early-stage (Stage-I) detection within a single cohort. |

---

## 📂 Project Structure

- `config/`: Centralized configuration (`config.py`) and metric logging.
- `data/`: Automated retrieval and processing of GEO datasets.
- `network/`: Logic for building ceRNA interaction networks (circBase, miRBase).
- `features/`: Topological feature extraction (Persistent Homology, Centrality, Graph Entropy).
- `models/`: 
  - `robust_validation.py`: Dataset-blocked nested CV for topology models.
  - `train_full_expression_elasticnet.py`: High-dimensional Elastic Net on raw circRNA profiles.
  - `lodo_full_expression_elasticnet.py`: External validation (LODO) for expression models.
- `figures/`: Scripts to generate publication-ready plots and SHAP visualizations.
- `run_pipeline.py`: Main entry point for end-to-end execution.

---

## 🛠️ Installation & Setup

1. **Clone and Install Dependencies**:
   ```bash
   git clone https://github.com/Sharon-codes/CeRNA-Early-Stage-Cancer-Detection.git
   cd CeRNA-Early-Stage-Cancer-Detection
   pip install -r requirements.txt
   ```

2. **Initialize Databases & GEO Data**:
   ```bash
   python data/download_dbs.py
   python data/load_geo.py
   ```

---

## 📈 Usage

### 1. Run Full Pipeline
To execute data processing, network construction, and initial classification:
```bash
python run_pipeline.py
```

### 2. Robust Topology Validation
To run the dataset-blocked nested CV and LODO for the Hybrid Topology model:
```bash
python robust_validation.py
```

### 3. Full Expression Baseline
To train the Elastic Net model on the full log2(CPM+1) circRNA matrix:
```bash
python models/train_full_expression_elasticnet.py
python models/lodo_full_expression_elasticnet.py
```

### 4. Generate Figures
Scripts like `generate_final_figure1.py` reproduce the visualizations for the manuscript.

---

## 🔬 Methodology

- **Network Construction**: Competition-based linkage defined by shared miRNA binding sites.
- **Topological Features**: Betti numbers (B0, B1), persistence entropy, and hub-based centrality metrics.
- **Machine Learning**: Nested 5×3-fold Cross-Validation with Optuna hyperparameter optimization.
- **Validation**: Leave-One-Dataset-Out (LODO) to ensure platform and batch robustness.

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
