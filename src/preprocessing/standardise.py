"""
Data standardization module for CPU and GPU arrays.

This module provides functions to standardize (z-score normalize) neuroimaging data
to zero mean and unit variance, supporting both CPU (NumPy) and GPU (CuPy) computation.
"""

import numpy as np
from typing import Union

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def standardise_cpu(data: np.ndarray) -> np.ndarray:
    """
    Standardise a NumPy array to zero mean and unit variance.

    Computes z-score normalization per voxel (across timepoints):
        z = (x - mean) / std

    Args:
        data: Input array of shape (n_voxels, n_timepoints).
              Each row represents a voxel's timeseries.

    Returns:
        Standardized array of same shape, where each voxel timeseries
        has mean=0 and std=1.

    Raises:
        ValueError: If input is not a 2D array.
        ZeroDivisionError: If any voxel has zero standard deviation.

    Examples:
        >>> import numpy as np
        >>> data = np.random.randn(100, 200)
        >>> data_std = standardise_cpu(data)
        >>> np.allclose(data_std.mean(axis=1), 0, atol=1e-10)
        True
        >>> np.allclose(data_std.std(axis=1), 1, atol=1e-10)
        True
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)

    # Check for zero standard deviation
    if np.any(std == 0):
        raise ZeroDivisionError(
            "One or more voxels have zero standard deviation. "
            "Cannot standardize constant timeseries."
        )

    return (data - mean) / std


def standardise_gpu(data: 'cp.ndarray') -> 'cp.ndarray':
    """
    Standardise a CuPy array to zero mean and unit variance.

    GPU-accelerated version of standardise_cpu using CuPy.

    Args:
        data: Input CuPy array of shape (n_voxels, n_timepoints).
              Each row represents a voxel's timeseries.

    Returns:
        Standardized CuPy array of same shape, where each voxel timeseries
        has mean=0 and std=1.

    Raises:
        ImportError: If CuPy is not installed.
        ValueError: If input is not a 2D array.
        ZeroDivisionError: If any voxel has zero standard deviation.

    Examples:
        >>> import cupy as cp
        >>> data = cp.random.randn(100, 200)
        >>> data_std = standardise_gpu(data)
        >>> cp.allclose(data_std.mean(axis=1), 0, atol=1e-10)
        True
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is not available. Cannot use GPU standardization.")

    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    mean = cp.mean(data, axis=1, keepdims=True)
    std = cp.std(data, axis=1, keepdims=True)

    # Check for zero standard deviation
    if cp.any(std == 0):
        raise ZeroDivisionError(
            "One or more voxels have zero standard deviation. "
            "Cannot standardize constant timeseries."
        )

    return (data - mean) / std


def standardise(data: Union[np.ndarray, 'cp.ndarray'],
                use_gpu: bool = False) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Standardise data to zero mean and unit variance, automatically selecting CPU or GPU.

    This is a convenience wrapper that automatically routes to the appropriate
    standardization function based on input type and user preference.

    Args:
        data: Input array of shape (n_voxels, n_timepoints).
        use_gpu: If True, use GPU computation (requires CuPy). If False, use CPU.
                 Defaults to False for compatibility.

    Returns:
        Standardized array of same type as input.

    Raises:
        ValueError: If input is not a 2D array.
        ImportError: If use_gpu=True but CuPy is not available.

    Examples:
        >>> import numpy as np
        >>> data = np.random.randn(100, 200)
        >>> data_std = standardise(data, use_gpu=False)  # CPU
        >>> data_std.shape
        (100, 200)
    """
    if use_gpu:
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not available. Cannot use GPU standardization.")
        if not isinstance(data, cp.ndarray):
            data = cp.asarray(data)
        return standardise_gpu(data)
    else:
        if CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            data = cp.asnumpy(data)
        return standardise_cpu(data)
