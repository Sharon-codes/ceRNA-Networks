"""
Timing & Profiling Module
Estimates computational runtime for TDA, Graph Edit Distance (GED), and Spectral computations
on a small subset of samples/genes, extrapolating full pipeline execution time and ETA.
"""

import time
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any


def estimate_runtime(
    pipeline_fn: Callable[[pd.DataFrame, Dict[str, Any]], None],
    expr_df: pd.DataFrame,
    sample_size: int = 5,
    num_genes_subset: int = 150,
    config: Optional[Dict[str, Any]] = None
) -> float:
    """
    Executes a dry-run benchmark on a tiny fraction of the data to estimate
    and extrapolate the total execution runtime for the full pipeline.

    Args:
        pipeline_fn (Callable): Function or routine running representative steps.
        expr_df (pd.DataFrame): Full expression matrix (samples x genes).
        sample_size (int): Number of benchmark samples to test.
        num_genes_subset (int): Number of benchmark genes to test.
        config (Dict[str, Any], optional): Configuration parameters.

    Returns:
        float: Estimated total runtime in seconds.
    """
    if config is None:
        config = {}

    n_samples, n_genes = expr_df.shape
    bench_samples = min(sample_size, n_samples)
    bench_genes = min(num_genes_subset, n_genes)

    print("\n" + "=" * 75)
    print(" PIPELINE RUNTIME ESTIMATION & COMPUTATIONAL PROFILING ")
    print("=" * 75)
    print(f"[*] Benchmarking with subset: {bench_samples} samples x {bench_genes} genes...")

    subset_df = expr_df.iloc[:bench_samples, :bench_genes]

    t0 = time.time()
    try:
        pipeline_fn(subset_df, config)
    except Exception as e:
        print(f"[Profiling Warning] Dry run completed with partial execution notice: {e}")
    t1 = time.time()

    elapsed_bench = t1 - t0

    # Quadratic scaling scaling factor for correlation matrix & graph calculations O(G^2) or O(G^3)
    gene_scaling = (n_genes / bench_genes) ** 2.2
    sample_scaling = (n_samples / bench_samples) ** 1.1
    
    # Scale factor considering overall pipeline iterations
    total_scale_factor = gene_scaling * sample_scaling * 0.15
    estimated_total_seconds = max(elapsed_bench * total_scale_factor, elapsed_bench * 10.0)

    # Format ETA string
    hours = int(estimated_total_seconds // 3600)
    minutes = int((estimated_total_seconds % 3600) // 60)
    seconds = int(estimated_total_seconds % 60)
    eta_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    print(f"[+] Benchmark execution time: {elapsed_bench:.3f} seconds")
    print(f"[+] Extrapolating to full dataset ({n_samples} samples x {n_genes} genes)...")
    print(f"[+] ESTIMATED PIPELINE ETA: {eta_str} (HH:MM:SS)")
    print(f"[+] Estimated Memory Peak: ~{int(n_genes * n_genes * 8 / (1024 * 1024))} MB for Affinity Matrices")
    print("=" * 75 + "\n")

    return estimated_total_seconds
