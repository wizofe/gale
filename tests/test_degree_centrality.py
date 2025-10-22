import numpy as np
import cupy as cp
import pytest
from src.preprocessing.standardise import standardise_cpu, standardise_gpu
from src.gpu_engine.correlation import compute_correlation_gpu
from src.connectivity.degree_centrality import threshold_matrix_cpu, degree_centrality_cpu, threshold_matrix_gpu, degree_centrality_gpu

def test_degree_centrality_consistency():
    # Create synthetic data
    n_voxels = 100
    n_timepoints = 50
    data = np.random.randn(n_voxels, n_timepoints)

    # CPU pipeline
    data_std_cpu = standardise_cpu(data)
    corr_cpu = np.dot(data_std_cpu, data_std_cpu.T) / (n_timepoints - 1)
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

if __name__ == '__main__':
    pytest.main([__file__])
