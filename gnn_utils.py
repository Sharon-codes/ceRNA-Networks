"""
GNN Generalization Collapse Module
Implements a Graph Convolutional Network (GCN) in PyTorch to classify clinical target labels
and measures AUROC performance drop as a function of threshold-induced Graph Edit Distance (GED).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from typing import Tuple, List, Dict, Any

from graph_utils import compute_pearson_affinity, binarize_affinity, compute_graph_edit_distance

# PyTorch GCN Layer (PyTorch Native Implementation for Universal Compatibility)
class PyTorchGCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Z = D_hat^{-1/2} A_hat D_hat^{-1/2} X W
        """
        # Add self loops
        n = adj.size(0)
        adj_hat = adj + torch.eye(n, device=adj.device)
        deg = torch.sum(adj_hat, dim=1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-6), -0.5)
        d_mat = torch.diag(deg_inv_sqrt)
        
        norm_adj = d_mat @ adj_hat @ d_mat
        support = torch.mm(x, self.weight)
        output = torch.mm(norm_adj, support) + self.bias
        return output


class TranscriptomicGCN(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 64, out_classes: int = 2):
        super().__init__()
        self.gcn1 = PyTorchGCNLayer(in_features, hidden_dim)
        self.gcn2 = PyTorchGCNLayer(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gcn1(x, adj))
        h = self.dropout(h)
        h = F.relu(self.gcn2(h, adj))
        h = self.dropout(h)
        logits = self.classifier(h).squeeze(-1)
        return logits


def train_gcn_model(
    expr_train: pd.DataFrame,
    y_train: pd.Series,
    threshold: float = 0.80,
    epochs: int = 80,
    lr: float = 0.01
) -> Tuple[TranscriptomicGCN, float]:
    """
    Trains GCN classifier on primary cohort sample-sample graph.

    Args:
        expr_train (pd.DataFrame): Training expression matrix (samples x genes).
        y_train (pd.Series): Training clinical target labels.
        threshold (float): Binarization threshold theta for sample correlation.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.

    Returns:
        Tuple[TranscriptomicGCN, float]: Trained GCN model and training AUROC.
    """
    device = torch.device("cpu")
    # Sample-sample correlation network
    R_sample = np.corrcoef(expr_train.values)
    R_sample = np.nan_to_num(R_sample, nan=0.0)
    np.fill_diagonal(R_sample, 1.0)
    
    adj_bin = binarize_affinity(R_sample, threshold)

    x_tensor = torch.tensor(expr_train.values, dtype=torch.float32, device=device)
    adj_tensor = torch.tensor(adj_bin, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32, device=device)

    in_features = expr_train.shape[1]
    model = TranscriptomicGCN(in_features=in_features, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_tensor, adj_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(x_tensor, adj_tensor)
        probs = torch.sigmoid(test_logits).numpy()
        try:
            train_auroc = float(roc_auc_score(y_train.values, probs))
        except ValueError:
            train_auroc = 0.50

    return model, train_auroc


def evaluate_gnn_collapse(
    model: TranscriptomicGCN,
    expr_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.80,
    shift_magnitudes: List[float] = None
) -> List[Dict[str, float]]:
    """
    Executes Test 6: Evaluates GCN AUROC drop on test cohort subject to escalating
    continuous batch shifts, measuring performance as a strict function of threshold-induced GED.

    Args:
        model (TranscriptomicGCN): Trained GCN classifier.
        expr_test (pd.DataFrame): Test expression matrix.
        y_test (pd.Series): Test clinical labels.
        threshold (float): Binarization cutoff theta.
        shift_magnitudes (List[float]): Array of batch shift scalars.

    Returns:
        List[Dict[str, float]]: Metrics containing shift, GED, AUROC, Accuracy.
    """
    if shift_magnitudes is None:
        shift_magnitudes = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]

    device = torch.device("cpu")
    model.eval()

    # Baseline test sample-sample adjacency
    R_clean = np.corrcoef(expr_test.values)
    R_clean = np.nan_to_num(R_clean, nan=0.0)
    np.fill_diagonal(R_clean, 1.0)
    adj_clean = binarize_affinity(R_clean, threshold)

    results = []

    np.random.seed(42)
    for shift in shift_magnitudes:
        # Inject continuous batch shift
        gene_shift = np.random.uniform(shift * 0.5, shift * 1.5, size=expr_test.shape[1])
        expr_shifted_vals = expr_test.values + gene_shift
        
        # Re-compute correlation & re-threshold
        R_shifted = np.corrcoef(expr_shifted_vals)
        R_shifted = np.nan_to_num(R_shifted, nan=0.0)
        np.fill_diagonal(R_shifted, 1.0)
        adj_shifted = binarize_affinity(R_shifted, threshold)

        # Graph Edit Distance caused by thresholding shift
        ged = compute_graph_edit_distance(adj_clean, adj_shifted)

        # Evaluate GCN on shifted graph
        x_tensor = torch.tensor(expr_shifted_vals, dtype=torch.float32, device=device)
        adj_tensor = torch.tensor(adj_shifted, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = model(x_tensor, adj_tensor)
            probs = torch.sigmoid(logits).numpy()
            preds = (probs >= 0.5).astype(int)

        try:
            auroc = float(roc_auc_score(y_test.values, probs))
        except ValueError:
            auroc = 0.50

        acc = float(np.mean(preds == y_test.values))

        results.append({
            "batch_shift": float(shift),
            "graph_edit_distance": ged,
            "auroc": auroc,
            "accuracy": acc
        })

    return results
