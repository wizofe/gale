"""
neurogale Framework Experimental Runner.

This module provides the main entry point for running brain connectivity
analysis experiments with CPU/GPU comparison and benchmarking.
"""

import argparse
import time
import warnings
import numpy as np
from typing import Tuple, Optional

from src.preprocessing.standardise import standardise_cpu, standardise_gpu
from src.gpu_engine.correlation import compute_correlation_cpu, compute_correlation_gpu
from src.connectivity.degree_centrality import (
    threshold_matrix_gpu, degree_centrality_gpu,
    threshold_matrix_cpu, degree_centrality_cpu
)
from src.utils.gpu_utils import check_gpu_availability, get_gpu_info, warn_gpu_unavailable

# Check GPU availability at module load time
try:
    import cupy as cp
    GPU_AVAILABLE, GPU_ERROR = check_gpu_availability()
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    GPU_ERROR = "CuPy not installed"


def compute_cpu(data: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """
    Compute degree centrality using CPU (NumPy) implementation.

    Complete pipeline: standardization → correlation → thresholding → degree centrality.

    Args:
        data: Input fMRI data of shape (n_voxels, n_timepoints).
        threshold: Correlation threshold for network construction. Defaults to 0.2.

    Returns:
        Degree centrality values of shape (n_voxels,).

    Raises:
        ValueError: If input data is invalid.
    """
    # Standardise data on CPU
    data_std = standardise_cpu(data)
    n_timepoints = data_std.shape[1]
    # Compute correlation using NumPy
    corr_matrix = compute_correlation_cpu(data_std)
    # Apply thresholding
    corr_thresh = threshold_matrix_cpu(corr_matrix, threshold)
    # Compute degree centrality
    degree = degree_centrality_cpu(corr_thresh)
    return degree


def compute_gpu(data: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """
    Compute degree centrality using GPU (CuPy) implementation.

    Complete pipeline: standardization → correlation → thresholding → degree centrality.

    Args:
        data: Input fMRI data of shape (n_voxels, n_timepoints).
        threshold: Correlation threshold for network construction. Defaults to 0.2.

    Returns:
        Degree centrality values of shape (n_voxels,).

    Raises:
        ImportError: If CuPy is not available or GPU is not functional.
        ValueError: If input data is invalid.
    """
    if not GPU_AVAILABLE:
        raise ImportError(f"GPU not available: {GPU_ERROR}")

    # Transfer data to GPU and standardise
    data_gpu = cp.asarray(data)
    data_std = standardise_gpu(data_gpu)
    n_timepoints = data_std.shape[1]
    # Compute correlation using GPU
    corr_matrix = compute_correlation_gpu(data_std)
    # Apply thresholding
    corr_thresh = threshold_matrix_gpu(corr_matrix, threshold)
    # Compute degree centrality
    degree = degree_centrality_gpu(corr_thresh)
    # Transfer result back to CPU
    return cp.asnumpy(degree)


def run_comparison(data: np.ndarray,
                   threshold: float = 0.2,
                   run_gpu: bool = True) -> dict:
    """
    Run CPU/GPU comparison experiment for degree centrality computation.

    Args:
        data: Input fMRI data of shape (n_voxels, n_timepoints).
        threshold: Correlation threshold. Defaults to 0.2.
        run_gpu: Whether to run GPU implementation. Defaults to True.

    Returns:
        Dictionary containing:
            - 'cpu_time': CPU execution time in seconds
            - 'cpu_result': CPU degree centrality values
            - 'gpu_time': GPU execution time (None if not run)
            - 'gpu_result': GPU degree centrality values (None if not run)
            - 'correlation': Correlation between CPU/GPU results (None if not run)
            - 'speedup': GPU speedup factor (None if not run)
    """
    results = {}

    # Run CPU implementation
    print("\nRunning CPU implementation...")
    start_cpu = time.time()
    degree_cpu = compute_cpu(data, threshold)
    time_cpu = time.time() - start_cpu
    results['cpu_time'] = time_cpu
    results['cpu_result'] = degree_cpu
    print(f"✓ CPU degree centrality computed in {time_cpu:.4f} seconds.")

    # Run GPU implementation if available and requested
    if run_gpu and GPU_AVAILABLE:
        print("\nRunning GPU implementation...")
        # Synchronize before timing
        cp.cuda.Stream.null.synchronize()
        start_gpu = time.time()
        degree_gpu = compute_gpu(data, threshold)
        # Synchronize after computation for accurate timing
        cp.cuda.Stream.null.synchronize()
        time_gpu = time.time() - start_gpu
        results['gpu_time'] = time_gpu
        results['gpu_result'] = degree_gpu
        print(f"✓ GPU degree centrality computed in {time_gpu:.4f} seconds.")

        # Validate results
        correlation = np.corrcoef(degree_cpu, degree_gpu)[0, 1]
        results['correlation'] = correlation
        results['speedup'] = time_cpu / time_gpu if time_gpu > 0 else float('inf')

        print(f"\nValidation:")
        print(f"  Correlation between CPU and GPU results: {correlation:.6f}")
        print(f"  Speedup (CPU/GPU): {results['speedup']:.2f}x")

        if correlation < 0.999:
            warnings.warn(
                f"CPU-GPU correlation is {correlation:.6f}, which is below expected threshold (0.999). "
                "Results may not be numerically equivalent."
            )
    elif run_gpu and not GPU_AVAILABLE:
        warn_gpu_unavailable(GPU_ERROR)
        results['gpu_time'] = None
        results['gpu_result'] = None
        results['correlation'] = None
        results['speedup'] = None
    else:
        results['gpu_time'] = None
        results['gpu_result'] = None
        results['correlation'] = None
        results['speedup'] = None

    return results


def main():
    """Main entry point for gale experiments."""
    parser = argparse.ArgumentParser(
        description="gale: GPU-accelerated brain connectivity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_path', type=str, default=None,
                        help="Path to NIfTI file. If not provided, synthetic data is used.")
    parser.add_argument('--n_voxels', type=int, default=5000,
                        help="Number of voxels for synthetic data")
    parser.add_argument('--n_timepoints', type=int, default=200,
                        help="Number of timepoints for synthetic data")
    parser.add_argument('--threshold', type=float, default=0.2,
                        help="Correlation threshold (typical: 0.2-0.3)")
    parser.add_argument('--cpu-only', action='store_true',
                        help="Run CPU implementation only (skip GPU)")
    parser.add_argument('--show-gpu-info', action='store_true',
                        help="Display GPU information and exit")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # Show GPU info if requested
    if args.show_gpu_info:
        gpu_info = get_gpu_info()
        print("\n=== GPU Information ===")
        for key, value in gpu_info.items():
            print(f"{key}: {value}")
        return

    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        if GPU_AVAILABLE:
            cp.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Display configuration
    print("=" * 60)
    print("gale - GPU-Accelerated Brain Connectivity Analysis")
    print("=" * 60)

    # Show GPU status
    if GPU_AVAILABLE and not args.cpu_only:
        gpu_info = get_gpu_info()
        print(f"\nGPU: {gpu_info.get('name', 'Unknown')}")
        print(f"GPU Memory: {gpu_info.get('free_memory_gb', 0):.2f} GB free / "
              f"{gpu_info.get('total_memory_gb', 0):.2f} GB total")
    elif not GPU_AVAILABLE:
        print(f"\nGPU: Not available ({GPU_ERROR})")
        print("Running in CPU-only mode")
    else:
        print("\nGPU: Available but disabled (--cpu-only flag)")

    # Load or generate data
    if args.data_path:
        # For now, NIfTI loading is not fully implemented
        print(f"\nNote: NIfTI data loading is not yet fully implemented.")
        print(f"Requested path: {args.data_path}")
        print("Falling back to synthetic data.")
        data = np.random.randn(args.n_voxels, args.n_timepoints)
    else:
        print(f"\nGenerating synthetic data...")
        data = np.random.randn(args.n_voxels, args.n_timepoints)

    print(f"Data shape: {data.shape} (voxels × timepoints)")
    print(f"Correlation threshold: {args.threshold}")
    print(f"Memory footprint: ~{data.nbytes / 1024**2:.2f} MB")

    # Run experiment
    try:
        results = run_comparison(
            data,
            threshold=args.threshold,
            run_gpu=not args.cpu_only
        )

        # Display summary
        print("\n" + "=" * 60)
        print("Experiment Summary")
        print("=" * 60)
        print(f"CPU Time:     {results['cpu_time']:.4f} seconds")
        if results['gpu_time'] is not None:
            print(f"GPU Time:     {results['gpu_time']:.4f} seconds")
            print(f"Speedup:      {results['speedup']:.2f}x")
            print(f"CPU-GPU Corr: {results['correlation']:.6f}")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during experiment: {e}")
        raise


if __name__ == '__main__':
    main()
