"""
Comprehensive test suite for degree centrality and thresholding functions.

Tests graph theory metrics computation with validation of mathematical
properties, edge cases, and numerical accuracy.
"""

import numpy as np
import pytest

from src.connectivity.degree_centrality import (
    threshold_matrix_cpu, threshold_matrix_gpu,
    degree_centrality_cpu, degree_centrality_gpu,
    threshold_matrix, degree_centrality, CUPY_AVAILABLE
)

# Skip GPU tests if CuPy is not available
if CUPY_AVAILABLE:
    import cupy as cp
    from src.utils.gpu_utils import check_gpu_availability
    GPU_AVAILABLE, _ = check_gpu_availability()
else:
    GPU_AVAILABLE = False

requires_gpu = pytest.mark.skipif(
    not GPU_AVAILABLE,
    reason="GPU/CUDA not available"
)


class TestThresholdMatrixCPU:
    """Test suite for CPU thresholding function."""

    def test_basic_thresholding(self):
        """Test basic thresholding operation."""
        corr = np.array([[1.0, 0.5, 0.1],
                         [0.5, 1.0, 0.3],
                         [0.1, 0.3, 1.0]])
        result = threshold_matrix_cpu(corr, threshold=0.2)

        expected = np.array([[1.0, 0.5, 0.0],
                             [0.5, 1.0, 0.3],
                             [0.0, 0.3, 1.0]])

        np.testing.assert_array_equal(result, expected)

    def test_threshold_zeros_negative_correlations(self):
        """Test that negative correlations are zeroed."""
        corr = np.array([[1.0, -0.5], [-0.5, 1.0]])
        result = threshold_matrix_cpu(corr, threshold=0.0)

        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_equal(result, expected)

    def test_threshold_preserves_high_correlations(self):
        """Test that high correlations are preserved."""
        corr = np.array([[1.0, 0.9], [0.9, 1.0]])
        result = threshold_matrix_cpu(corr, threshold=0.5)

        np.testing.assert_array_equal(result, corr)

    def test_different_thresholds(self):
        """Test various threshold values."""
        corr = np.eye(5) + 0.3  # Correlations of 0.3 off-diagonal

        result_01 = threshold_matrix_cpu(corr, threshold=0.1)
        result_05 = threshold_matrix_cpu(corr, threshold=0.5)

        # Lower threshold keeps more values
        assert np.count_nonzero(result_01) > np.count_nonzero(result_05)

    def test_invalid_non_square_matrix(self):
        """Test error handling for non-square matrix."""
        corr = np.random.randn(5, 10)

        with pytest.raises(ValueError, match="Expected square matrix"):
            threshold_matrix_cpu(corr)

    def test_invalid_1d_input(self):
        """Test error handling for 1D input."""
        corr = np.random.randn(10)

        with pytest.raises(ValueError, match="Expected 2D array"):
            threshold_matrix_cpu(corr)

    def test_invalid_threshold_too_high(self):
        """Test error handling for threshold > 1."""
        corr = np.eye(5)

        with pytest.raises(ValueError, match="Threshold must be in range"):
            threshold_matrix_cpu(corr, threshold=1.5)

    def test_invalid_threshold_too_low(self):
        """Test error handling for threshold < -1."""
        corr = np.eye(5)

        with pytest.raises(ValueError, match="Threshold must be in range"):
            threshold_matrix_cpu(corr, threshold=-2.0)

    def test_threshold_boundary_values(self):
        """Test threshold at boundary values."""
        corr = np.array([[1.0, 0.2], [0.2, 1.0]])

        # Threshold at exactly 0.2 - should include the edge
        result = threshold_matrix_cpu(corr, threshold=0.2)
        assert result[0, 1] == 0.2

        # Threshold just above 0.2
        result = threshold_matrix_cpu(corr, threshold=0.21)
        assert result[0, 1] == 0.0

    def test_preserves_symmetry(self):
        """Test that thresholding preserves symmetry."""
        corr = np.random.randn(10, 10)
        corr = (corr + corr.T) / 2  # Make symmetric

        result = threshold_matrix_cpu(corr, threshold=0.3)

        np.testing.assert_allclose(result, result.T)

    def test_deterministic(self):
        """Test deterministic output."""
        corr = np.random.randn(10, 10)

        result1 = threshold_matrix_cpu(corr, threshold=0.2)
        result2 = threshold_matrix_cpu(corr, threshold=0.2)

        np.testing.assert_array_equal(result1, result2)


@requires_gpu
class TestThresholdMatrixGPU:
    """Test suite for GPU thresholding function."""

    def test_basic_thresholding(self):
        """Test basic thresholding on GPU."""
        corr = cp.array([[1.0, 0.5, 0.1],
                         [0.5, 1.0, 0.3],
                         [0.1, 0.3, 1.0]])
        result = threshold_matrix_gpu(corr, threshold=0.2)

        expected = cp.array([[1.0, 0.5, 0.0],
                             [0.5, 1.0, 0.3],
                             [0.0, 0.3, 1.0]])

        cp.testing.assert_array_equal(result, expected)

    def test_cpu_gpu_consistency(self):
        """Test that CPU and GPU produce identical results."""
        corr_cpu = np.random.randn(50, 50)
        corr_gpu = cp.asarray(corr_cpu)

        result_cpu = threshold_matrix_cpu(corr_cpu, threshold=0.3)
        result_gpu = cp.asnumpy(threshold_matrix_gpu(corr_gpu, threshold=0.3))

        np.testing.assert_array_equal(result_cpu, result_gpu)

    def test_invalid_input(self):
        """Test error handling."""
        corr = cp.random.randn(10)  # 1D

        with pytest.raises(ValueError):
            threshold_matrix_gpu(corr)


class TestDegreeCentralityCPU:
    """Test suite for CPU degree centrality computation."""

    def test_basic_degree_centrality(self):
        """Test basic degree centrality computation."""
        corr = np.array([[1.0, 0.5, 0.0],
                         [0.5, 1.0, 0.3],
                         [0.0, 0.3, 1.0]])
        degree = degree_centrality_cpu(corr)

        expected = np.array([1.5, 1.8, 1.3])
        np.testing.assert_allclose(degree, expected)

    def test_fully_connected_network(self):
        """Test degree centrality on fully connected network."""
        n = 10
        corr = np.ones((n, n))  # All connections = 1
        degree = degree_centrality_cpu(corr)

        # Each node has n connections
        expected = np.full(n, n)
        np.testing.assert_array_equal(degree, expected)

    def test_no_connections(self):
        """Test degree centrality with no connections (diagonal only)."""
        n = 10
        corr = np.eye(n)  # Only self-connections
        degree = degree_centrality_cpu(corr)

        # Each node has degree = 1 (self-connection)
        expected = np.ones(n)
        np.testing.assert_array_equal(degree, expected)

    def test_weighted_vs_binary(self):
        """Test that degree centrality accounts for weights."""
        # Weighted network
        corr_weighted = np.array([[1.0, 0.5], [0.5, 1.0]])
        degree_weighted = degree_centrality_cpu(corr_weighted)

        # Binary network (threshold at 0)
        corr_binary = (corr_weighted > 0).astype(float)
        degree_binary = degree_centrality_cpu(corr_binary)

        # Weighted should be less than binary (0.5 < 1.0)
        assert degree_weighted[0] < degree_binary[0]

    def test_star_network(self):
        """Test degree centrality on star network."""
        # Hub-and-spoke network: one central node connected to all others
        n = 10
        corr = np.eye(n)
        corr[0, 1:] = 0.5  # Hub connects to all
        corr[1:, 0] = 0.5  # Symmetric

        degree = degree_centrality_cpu(corr)

        # Hub should have highest degree
        assert degree[0] == max(degree)
        assert degree[0] > degree[1]

    def test_invalid_non_square(self):
        """Test error handling for non-square matrix."""
        corr = np.random.randn(5, 10)

        with pytest.raises(ValueError, match="Expected square matrix"):
            degree_centrality_cpu(corr)

    def test_invalid_1d_input(self):
        """Test error handling for 1D input."""
        corr = np.random.randn(10)

        with pytest.raises(ValueError, match="Expected 2D array"):
            degree_centrality_cpu(corr)

    def test_deterministic(self):
        """Test deterministic output."""
        corr = np.random.randn(10, 10)

        result1 = degree_centrality_cpu(corr)
        result2 = degree_centrality_cpu(corr)

        np.testing.assert_array_equal(result1, result2)


@requires_gpu
class TestDegreeCentralityGPU:
    """Test suite for GPU degree centrality computation."""

    def test_basic_degree_centrality(self):
        """Test basic degree centrality on GPU."""
        corr = cp.array([[1.0, 0.5, 0.0],
                         [0.5, 1.0, 0.3],
                         [0.0, 0.3, 1.0]])
        degree = degree_centrality_gpu(corr)

        expected = cp.array([1.5, 1.8, 1.3])
        cp.testing.assert_allclose(degree, expected)

    def test_cpu_gpu_consistency(self):
        """Test that CPU and GPU produce identical results."""
        corr_cpu = np.random.randn(100, 100)
        corr_cpu = (corr_cpu + corr_cpu.T) / 2  # Symmetric

        degree_cpu = degree_centrality_cpu(corr_cpu)

        corr_gpu = cp.asarray(corr_cpu)
        degree_gpu = cp.asnumpy(degree_centrality_gpu(corr_gpu))

        np.testing.assert_allclose(degree_cpu, degree_gpu, rtol=1e-5)

    def test_invalid_input(self):
        """Test error handling."""
        corr = cp.random.randn(10)  # 1D

        with pytest.raises(ValueError):
            degree_centrality_gpu(corr)


class TestPipelineIntegration:
    """Test complete threshold + degree centrality pipeline."""

    def test_complete_pipeline_cpu(self):
        """Test complete pipeline on CPU."""
        # Create correlation matrix
        n = 50
        np.random.seed(42)
        corr = np.random.randn(n, n)
        corr = (corr + corr.T) / 2  # Symmetric
        np.fill_diagonal(corr, 1.0)  # Perfect self-correlation

        # Apply threshold
        thresh_corr = threshold_matrix_cpu(corr, threshold=0.2)

        # Compute degree
        degree = degree_centrality_cpu(thresh_corr)

        # Check properties
        assert degree.shape == (n,)
        assert np.all(degree >= 0)
        assert degree.max() <= n  # Max degree is n (all connections)

    @requires_gpu
    def test_complete_pipeline_gpu(self):
        """Test complete pipeline on GPU."""
        n = 50
        cp.random.seed(42)
        corr = cp.random.randn(n, n)
        corr = (corr + corr.T) / 2
        cp.fill_diagonal(corr, 1.0)

        thresh_corr = threshold_matrix_gpu(corr, threshold=0.2)
        degree = degree_centrality_gpu(thresh_corr)

        assert degree.shape == (n,)
        assert cp.all(degree >= 0)

    @requires_gpu
    def test_pipeline_cpu_gpu_consistency(self):
        """Test that complete pipeline gives same results on CPU and GPU."""
        n = 100
        np.random.seed(42)
        corr = np.random.randn(n, n)
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)

        # CPU pipeline
        thresh_cpu = threshold_matrix_cpu(corr, threshold=0.3)
        degree_cpu = degree_centrality_cpu(thresh_cpu)

        # GPU pipeline
        corr_gpu = cp.asarray(corr)
        thresh_gpu = threshold_matrix_gpu(corr_gpu, threshold=0.3)
        degree_gpu = cp.asnumpy(degree_centrality_gpu(thresh_gpu))

        np.testing.assert_allclose(degree_cpu, degree_gpu, rtol=1e-5)


class TestWrapperFunctions:
    """Test wrapper functions for threshold and degree centrality."""

    def test_threshold_wrapper_cpu(self):
        """Test threshold wrapper with CPU."""
        corr = np.random.randn(10, 10)
        result = threshold_matrix(corr, threshold=0.2, use_gpu=False)

        assert isinstance(result, np.ndarray)

    def test_degree_wrapper_cpu(self):
        """Test degree centrality wrapper with CPU."""
        corr = np.random.randn(10, 10)
        result = degree_centrality(corr, use_gpu=False)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10,)

    @requires_gpu
    def test_threshold_wrapper_gpu(self):
        """Test threshold wrapper with GPU."""
        corr = np.random.randn(10, 10)
        result = threshold_matrix(corr, threshold=0.2, use_gpu=True)

        assert isinstance(result, cp.ndarray)

    @requires_gpu
    def test_degree_wrapper_gpu(self):
        """Test degree centrality wrapper with GPU."""
        corr = np.random.randn(10, 10)
        result = degree_centrality(corr, use_gpu=True)

        assert isinstance(result, cp.ndarray)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_voxel(self):
        """Test with single voxel (1x1 matrix)."""
        corr = np.array([[1.0]])
        thresh = threshold_matrix_cpu(corr, threshold=0.5)
        degree = degree_centrality_cpu(thresh)

        assert thresh.shape == (1, 1)
        assert degree.shape == (1,)
        assert degree[0] == 1.0

    def test_all_zeros_after_threshold(self):
        """Test when threshold removes all connections."""
        corr = np.eye(5) + 0.1  # Very weak off-diagonal correlations
        thresh = threshold_matrix_cpu(corr, threshold=0.5)

        # Only diagonal remains
        assert np.allclose(thresh, np.eye(5))

        degree = degree_centrality_cpu(thresh)
        np.testing.assert_array_equal(degree, np.ones(5))

    def test_negative_threshold(self):
        """Test with negative threshold to include negative correlations."""
        corr = np.array([[1.0, -0.5], [-0.5, 1.0]])
        thresh = threshold_matrix_cpu(corr, threshold=-0.3)

        # Should keep the -0.5 correlation
        assert thresh[0, 1] == -0.5

        degree = degree_centrality_cpu(thresh)
        # Degree includes negative weights
        expected = np.array([0.5, 0.5])  # 1.0 + (-0.5) = 0.5
        np.testing.assert_allclose(degree, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
