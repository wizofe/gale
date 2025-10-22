"""
Graph theory metrics for brain connectivity analysis.

This module provides functions to compute degree centrality and related
graph metrics from brain correlation matrices. Supports both CPU (NumPy)
and GPU (CuPy) computation.
"""

import numpy as np
from typing import Union

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def threshold_matrix_cpu(corr_matrix: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """
    Apply a correlation threshold to a CPU correlation matrix.

    Sets all correlations below the threshold to zero, creating a sparse
    connectivity matrix. This is a common preprocessing step in brain
    network analysis to remove weak or spurious connections.

    Args:
        corr_matrix: Correlation matrix of shape (n_voxels, n_voxels).
                     Values should be in range [-1, 1].
        threshold: Minimum correlation value to retain. Typical values are
                   0.2-0.3 for brain connectivity. Defaults to 0.2.

    Returns:
        Thresholded correlation matrix where values below threshold are set to 0.

    Raises:
        ValueError: If input is not a 2D square matrix or threshold is invalid.

    Examples:
        >>> import numpy as np
        >>> corr = np.array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.3], [0.1, 0.3, 1.0]])
        >>> thresh_corr = threshold_matrix_cpu(corr, threshold=0.2)
        >>> thresh_corr
        array([[1. , 0.5, 0. ],
               [0.5, 1. , 0.3],
               [0. , 0.3, 1. ]])
    """
    if corr_matrix.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {corr_matrix.shape}")

    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError(
            f"Expected square matrix, got shape {corr_matrix.shape}"
        )

    if not (-1 <= threshold <= 1):
        raise ValueError(
            f"Threshold must be in range [-1, 1], got {threshold}"
        )

    return np.where(corr_matrix >= threshold, corr_matrix, 0)


def threshold_matrix_gpu(corr_matrix: 'cp.ndarray', threshold: float = 0.2) -> 'cp.ndarray':
    """
    Apply a correlation threshold to a GPU correlation matrix.

    GPU-accelerated version of threshold_matrix_cpu using CuPy.

    Args:
        corr_matrix: CuPy correlation matrix of shape (n_voxels, n_voxels).
        threshold: Minimum correlation value to retain. Defaults to 0.2.

    Returns:
        Thresholded CuPy correlation matrix where values below threshold are set to 0.

    Raises:
        ImportError: If CuPy is not available.
        ValueError: If input is not a 2D square matrix or threshold is invalid.

    Examples:
        >>> import cupy as cp
        >>> corr = cp.array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.3], [0.1, 0.3, 1.0]])
        >>> thresh_corr = threshold_matrix_gpu(corr, threshold=0.2)
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is not available. Cannot use GPU thresholding.")

    if corr_matrix.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {corr_matrix.shape}")

    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError(
            f"Expected square matrix, got shape {corr_matrix.shape}"
        )

    if not (-1 <= threshold <= 1):
        raise ValueError(
            f"Threshold must be in range [-1, 1], got {threshold}"
        )

    return cp.where(corr_matrix >= threshold, corr_matrix, 0)


def degree_centrality_cpu(thresholded_matrix: np.ndarray) -> np.ndarray:
    """
    Compute weighted degree centrality from a thresholded CPU correlation matrix.

    Degree centrality measures the connectivity strength of each voxel by
    summing all its connections. In a weighted network, this is the sum of
    all edge weights (correlation values) for each node (voxel).

    Formula: degree_i = sum_j(w_ij) where w_ij is the edge weight between nodes i and j.

    Args:
        thresholded_matrix: Thresholded correlation matrix of shape (n_voxels, n_voxels).
                           Should be output from threshold_matrix_cpu/gpu.

    Returns:
        Degree centrality values of shape (n_voxels,).
        Higher values indicate more strongly connected voxels.

    Raises:
        ValueError: If input is not a 2D square matrix.

    Examples:
        >>> import numpy as np
        >>> corr = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 1.0]])
        >>> degree = degree_centrality_cpu(corr)
        >>> degree
        array([1.5, 1.8, 1.3])

    References:
        Rubinov M, Sporns O (2010) NeuroImage. Complex network measures of brain
        connectivity: Uses and interpretations. doi:10.1016/j.neuroimage.2009.10.003
    """
    if thresholded_matrix.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {thresholded_matrix.shape}")

    if thresholded_matrix.shape[0] != thresholded_matrix.shape[1]:
        raise ValueError(
            f"Expected square matrix, got shape {thresholded_matrix.shape}"
        )

    degree = np.sum(thresholded_matrix, axis=1)
    return degree


def degree_centrality_gpu(thresholded_matrix: 'cp.ndarray') -> 'cp.ndarray':
    """
    Compute weighted degree centrality from a thresholded GPU correlation matrix.

    GPU-accelerated version of degree_centrality_cpu using CuPy.

    Args:
        thresholded_matrix: Thresholded CuPy correlation matrix of shape (n_voxels, n_voxels).

    Returns:
        Degree centrality values (CuPy array) of shape (n_voxels,).

    Raises:
        ImportError: If CuPy is not available.
        ValueError: If input is not a 2D square matrix.

    References:
        Rubinov M, Sporns O (2010) NeuroImage. Complex network measures of brain
        connectivity: Uses and interpretations. doi:10.1016/j.neuroimage.2009.10.003
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is not available. Cannot use GPU degree centrality.")

    if thresholded_matrix.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {thresholded_matrix.shape}")

    if thresholded_matrix.shape[0] != thresholded_matrix.shape[1]:
        raise ValueError(
            f"Expected square matrix, got shape {thresholded_matrix.shape}"
        )

    degree = cp.sum(thresholded_matrix, axis=1)
    return degree


def threshold_matrix(corr_matrix: Union[np.ndarray, 'cp.ndarray'],
                     threshold: float = 0.2,
                     use_gpu: bool = False) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Apply correlation threshold, automatically selecting CPU or GPU.

    Convenience wrapper that automatically routes to the appropriate
    thresholding function based on input type and user preference.

    Args:
        corr_matrix: Correlation matrix of shape (n_voxels, n_voxels).
        threshold: Minimum correlation value to retain. Defaults to 0.2.
        use_gpu: If True, use GPU computation. If False, use CPU.

    Returns:
        Thresholded correlation matrix of same type as input.

    Raises:
        ValueError: If input is invalid.
        ImportError: If use_gpu=True but CuPy is not available.
    """
    if use_gpu:
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not available. Cannot use GPU thresholding.")
        if not isinstance(corr_matrix, cp.ndarray):
            corr_matrix = cp.asarray(corr_matrix)
        return threshold_matrix_gpu(corr_matrix, threshold)
    else:
        if CUPY_AVAILABLE and isinstance(corr_matrix, cp.ndarray):
            corr_matrix = cp.asnumpy(corr_matrix)
        return threshold_matrix_cpu(corr_matrix, threshold)


def degree_centrality(thresholded_matrix: Union[np.ndarray, 'cp.ndarray'],
                      use_gpu: bool = False) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Compute degree centrality, automatically selecting CPU or GPU.

    Convenience wrapper that automatically routes to the appropriate
    degree centrality function based on input type and user preference.

    Args:
        thresholded_matrix: Thresholded correlation matrix of shape (n_voxels, n_voxels).
        use_gpu: If True, use GPU computation. If False, use CPU.

    Returns:
        Degree centrality values of same type as input, shape (n_voxels,).

    Raises:
        ValueError: If input is invalid.
        ImportError: If use_gpu=True but CuPy is not available.
    """
    if use_gpu:
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not available. Cannot use GPU degree centrality.")
        if not isinstance(thresholded_matrix, cp.ndarray):
            thresholded_matrix = cp.asarray(thresholded_matrix)
        return degree_centrality_gpu(thresholded_matrix)
    else:
        if CUPY_AVAILABLE and isinstance(thresholded_matrix, cp.ndarray):
            thresholded_matrix = cp.asnumpy(thresholded_matrix)
        return degree_centrality_cpu(thresholded_matrix)
