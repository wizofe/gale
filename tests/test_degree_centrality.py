"""
Original consistency test for degree centrality (legacy test - kept for compatibility).

This test checks CPU/GPU consistency but requires GPU to run.
For comprehensive tests, see test_degree_centrality_comprehensive.py
"""

import numpy as np
import pytest

from src.preprocessing.standardise import standardise_cpu, CUPY_AVAILABLE
from src.gpu_engine.correlation import compute_correlation_cpu
from src.connectivity.degree_centrality import threshold_matrix_cpu, degree_centrality_cpu

# Conditional GPU imports
if CUPY_AVAILABLE:
    import cupy as cp
    from src.preprocessing.standardise import standardise_gpu
    from src.gpu_engine.correlation import compute_correlation_gpu
    from src.connectivity.degree_centrality import threshold_matrix_gpu, degree_centrality_gpu
    from src.utils.gpu_utils import check_gpu_availability
    GPU_AVAILABLE, _ = check_gpu_availability()
else:
    GPU_AVAILABLE = False


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU/CUDA not available")
def test_degree_centrality_consistency():
    """Test CPU-GPU consistency for degree centrality pipeline (requires GPU)."""
    # Create synthetic data
    n_voxels = 100
    n_timepoints = 50
    data = np.random.randn(n_voxels, n_timepoints)

    # CPU pipeline
    data_std_cpu = standardise_cpu(data)
    corr_cpu = compute_correlation_cpu(data_std_cpu)
    thresh_cpu = threshold_matrix_cpu(corr_cpu, threshold=0.2)
    degree_cpu = degree_centrality_cpu(thresh_cpu)

    # GPU pipeline
    data_gpu = cp.asarray(data)
    data_std_gpu = standardise_gpu(data_gpu)
    corr_gpu = compute_correlation_gpu(data_std_gpu)
    thresh_gpu = threshold_matrix_gpu(corr_gpu, threshold=0.2)
    degree_gpu = cp.asnumpy(degree_centrality_gpu(thresh_gpu))

    # Check that the results are nearly equal
    np.testing.assert_allclose(degree_cpu, degree_gpu, rtol=1e-5)


def test_cpu_only_pipeline():
    """Test CPU-only degree centrality pipeline (no GPU required)."""
    # Create synthetic data
    n_voxels = 100
    n_timepoints = 50
    np.random.seed(42)
    data = np.random.randn(n_voxels, n_timepoints)

    # CPU pipeline
    data_std_cpu = standardise_cpu(data)
    corr_cpu = compute_correlation_cpu(data_std_cpu)
    thresh_cpu = threshold_matrix_cpu(corr_cpu, threshold=0.2)
    degree_cpu = degree_centrality_cpu(thresh_cpu)

    # Check output properties
    assert degree_cpu.shape == (n_voxels,)
    assert np.all(degree_cpu >= 0)
    assert degree_cpu.max() > 0  # At least some connections above threshold


if __name__ == '__main__':
    pytest.main([__file__])
