"""
Graph Utilities & Network Topology Metrics Module
Calculates Pearson affinity matrices, threshold-based binarization,
Betti_0 connected components, Louvain communities & modularity,
Graph Edit Distance (GED), and Spectral Distance on Normalized Laplacians.
"""

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import scipy.linalg as la
from typing import Tuple, Dict, Any, List

try:
    import community as community_louvain
    HAS_PYTHON_LOUVAIN = True
except ImportError:
    HAS_PYTHON_LOUVAIN = False


def compute_pearson_affinity(expr_df: pd.DataFrame) -> np.ndarray:
    """
    Computes pairwise Pearson correlation affinity matrix among genes.

    Args:
        expr_df (pd.DataFrame): Expression matrix (samples x genes).

    Returns:
        np.ndarray: Continuous affinity matrix R of shape (genes x genes).
    """
    # np.corrcoef expects variables as rows, so transpose expression (genes x samples)
    corr_matrix = np.corrcoef(expr_df.values.T)
    # Handle NaN values due to zero-variance genes
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)
    return corr_matrix


def binarize_affinity(affinity_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Applies hard thresholding to convert continuous affinity matrix into a binary adjacency matrix.

    Args:
        affinity_matrix (np.ndarray): Continuous correlation matrix R.
        threshold (float): Binarization cutoff theta.

    Returns:
        np.ndarray: Binary adjacency matrix A of shape (genes x genes).
    """
    adj = (affinity_matrix >= threshold).astype(np.int8)
    np.fill_diagonal(adj, 0)
    return adj


def compute_network_metrics(adj_matrix: np.ndarray) -> Dict[str, float]:
    """
    Calculates 0D Betti number (connected components), Louvain community count,
    and Network Modularity (Q).

    Args:
        adj_matrix (np.ndarray): Binary adjacency matrix.

    Returns:
        Dict[str, float]: Dictionary containing betti_0, community_count, modularity.
    """
    G = nx.from_numpy_array(adj_matrix)
    num_edges = G.number_of_edges()

    # 0-dimensional Betti number (number of connected components)
    betti_0 = float(nx.number_connected_components(G))

    if num_edges == 0:
        return {
            "betti_0": betti_0,
            "community_count": float(G.number_of_nodes()),
            "modularity": 0.0,
            "edge_count": 0.0,
            "density": 0.0
        }

    # Louvain Community Detection & Modularity
    if HAS_PYTHON_LOUVAIN:
        partition = community_louvain.best_partition(G, random_state=42)
        community_count = float(len(set(partition.values())))
        modularity = float(community_louvain.modularity(partition, G))
    else:
        # Fallback to NetworkX Louvain implementation
        communities = list(nx.community.louvain_communities(G, seed=42))
        community_count = float(len(communities))
        modularity = float(nx.community.modularity(G, communities))

    density = float(nx.density(G))

    return {
        "betti_0": betti_0,
        "community_count": community_count,
        "modularity": modularity,
        "edge_count": float(num_edges),
        "density": density
    }


def compute_graph_edit_distance(adj1: np.ndarray, adj2: np.ndarray) -> float:
    """
    Computes exact Graph Edit Distance (GED) between two unweighted graphs
    sharing aligned node sets. On aligned nodes, GED is equal to the edge symmetric difference.

    Formula: GED(G1, G2) = 0.5 * sum(|A1_ij - A2_ij|) = |E1 \\ E2| + |E2 \\ E1|

    Args:
        adj1 (np.ndarray): Binary adjacency matrix 1.
        adj2 (np.ndarray): Binary adjacency matrix 2.

    Returns:
        float: Graph Edit Distance.
    """
    diff = np.abs(adj1 - adj2)
    # Divide by 2 for undirected graph symmetry
    return float(np.sum(diff) / 2.0)


def compute_spectral_distance(adj1: np.ndarray, adj2: np.ndarray) -> float:
    """
    Computes Spectral Distance as the L2 norm difference between the sorted eigenvalues
    of the Normalized Laplacian matrices of two graphs.

    Formula: Spectral_Dist = sqrt( sum( (lambda_i(L1) - lambda_i(L2))^2 ) )

    Args:
        adj1 (np.ndarray): Binary adjacency matrix 1.
        adj2 (np.ndarray): Binary adjacency matrix 2.

    Returns:
        float: Spectral Distance.
    """
    def normalized_laplacian_eigs(adj: np.ndarray) -> np.ndarray:
        n = adj.shape[0]
        deg = np.sum(adj, axis=1).astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        
        # Normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
        L = np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt
        
        # Eigenvalues sorted in ascending order
        eigs = la.eigvalsh(L)
        return eigs

    eigs1 = normalized_laplacian_eigs(adj1)
    eigs2 = normalized_laplacian_eigs(adj2)

    spectral_dist = np.sqrt(np.sum((eigs1 - eigs2) ** 2))
    return float(spectral_dist)
