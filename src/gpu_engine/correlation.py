"""
Correlation matrix computation module for GPU-accelerated analysis.

This module provides functions to compute Pearson correlation matrices
for brain connectivity analysis using GPU (CuPy) or CPU (NumPy) computation.
"""

import numpy as np
from typing import Union

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def compute_correlation_cpu(data: np.ndarray) -> np.ndarray:
    """
    Compute the Pearson correlation matrix for a CPU array using NumPy.

    Assumes the data is already standardized (zero mean, unit variance).
    The correlation is computed as: corr = (data @ data.T) / (n_timepoints - 1)

    Args:
        data: Standardized input array of shape (n_voxels, n_timepoints).
              Each row represents a voxel's z-scored timeseries.

    Returns:
        Correlation matrix of shape (n_voxels, n_voxels).
        Element [i,j] represents Pearson correlation between voxels i and j.

    Raises:
        ValueError: If input is not a 2D array or has fewer than 2 timepoints.

    Examples:
        >>> import numpy as np
        >>> data = np.random.randn(100, 200)
        >>> # Standardize first
        >>> data_std = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
        >>> corr = compute_correlation_cpu(data_std)
        >>> corr.shape
        (100, 100)
        >>> np.allclose(np.diag(corr), 1.0)  # Diagonal should be 1.0
        True
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    n_timepoints = data.shape[1]
    if n_timepoints < 2:
        raise ValueError(f"Need at least 2 timepoints, got {n_timepoints}")

    corr_matrix = np.dot(data, data.T) / (n_timepoints - 1)
    return corr_matrix


def compute_correlation_gpu(data: 'cp.ndarray') -> 'cp.ndarray':
    """
    Compute the Pearson correlation matrix for a GPU array using CuPy.

    GPU-accelerated version of compute_correlation_cpu using CuPy.
    Assumes the data is already standardized (zero mean, unit variance).

    Args:
        data: Standardized input CuPy array of shape (n_voxels, n_timepoints).
              Each row represents a voxel's z-scored timeseries.

    Returns:
        Correlation matrix (CuPy array) of shape (n_voxels, n_voxels).
        Element [i,j] represents Pearson correlation between voxels i and j.

    Raises:
        ImportError: If CuPy is not available.
        ValueError: If input is not a 2D array or has fewer than 2 timepoints.

    Examples:
        >>> import cupy as cp
        >>> data = cp.random.randn(100, 200)
        >>> # Standardize first
        >>> data_std = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
        >>> corr = compute_correlation_gpu(data_std)
        >>> corr.shape
        (100, 100)
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is not available. Cannot use GPU correlation computation.")

    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    n_timepoints = data.shape[1]
    if n_timepoints < 2:
        raise ValueError(f"Need at least 2 timepoints, got {n_timepoints}")

    corr_matrix = cp.dot(data, data.T) / (n_timepoints - 1)
    return corr_matrix


def compute_correlation(data: Union[np.ndarray, 'cp.ndarray'],
                        use_gpu: bool = False) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Compute Pearson correlation matrix, automatically selecting CPU or GPU.

    Convenience wrapper that automatically routes to the appropriate
    correlation function based on input type and user preference.

    Args:
        data: Standardized input array of shape (n_voxels, n_timepoints).
        use_gpu: If True, use GPU computation (requires CuPy). If False, use CPU.
                 Defaults to False for compatibility.

    Returns:
        Correlation matrix of same type as input, shape (n_voxels, n_voxels).

    Raises:
        ValueError: If input is invalid.
        ImportError: If use_gpu=True but CuPy is not available.

    Examples:
        >>> import numpy as np
        >>> data = np.random.randn(100, 200)
        >>> data_std = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
        >>> corr = compute_correlation(data_std, use_gpu=False)
        >>> corr.shape
        (100, 100)
    """
    if use_gpu:
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not available. Cannot use GPU correlation computation.")
        if not isinstance(data, cp.ndarray):
            data = cp.asarray(data)
        return compute_correlation_gpu(data)
    else:
        if CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            data = cp.asnumpy(data)
        return compute_correlation_cpu(data)
