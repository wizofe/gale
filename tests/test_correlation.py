"""
Comprehensive test suite for correlation computation module.

Tests CPU and GPU implementations of Pearson correlation with validation
of mathematical properties, edge cases, and numerical accuracy.
"""

import numpy as np
import pytest

from src.preprocessing.standardise import standardise_cpu
from src.gpu_engine.correlation import (
    compute_correlation_cpu, compute_correlation_gpu,
    compute_correlation, CUPY_AVAILABLE
)

# Skip GPU tests if CuPy is not available
if CUPY_AVAILABLE:
    import cupy as cp
    from src.utils.gpu_utils import check_gpu_availability
    from src.preprocessing.standardise import standardise_gpu
    GPU_AVAILABLE, _ = check_gpu_availability()
else:
    GPU_AVAILABLE = False

requires_gpu = pytest.mark.skipif(
    not GPU_AVAILABLE,
    reason="GPU/CUDA not available"
)


class TestCorrelationCPU:
    """Test suite for CPU correlation computation."""

    def test_basic_correlation(self):
        """Test basic correlation computation."""
        np.random.seed(42)
        data = np.random.randn(100, 200)
        data_std = standardise_cpu(data)
        corr = compute_correlation_cpu(data_std)

        # Check shape
        assert corr.shape == (100, 100)

        # Check symmetry
        np.testing.assert_allclose(corr, corr.T, rtol=1e-10)

        # Check diagonal is approximately 1
        np.testing.assert_allclose(np.diag(corr), 1.0, rtol=1e-10)

    def test_correlation_range(self):
        """Test that correlations are in valid range [-1, 1]."""
        data = np.random.randn(50, 100)
        data_std = standardise_cpu(data)
        corr = compute_correlation_cpu(data_std)

        assert np.all(corr >= -1.0)
        assert np.all(corr <= 1.0)

    def test_perfect_correlation(self):
        """Test perfect positive correlation."""
        # Create identical timeseries
        base = np.random.randn(1, 100)
        data = np.repeat(base, 5, axis=0)  # 5 identical voxels
        data_std = standardise_cpu(data)
        corr = compute_correlation_cpu(data_std)

        # All correlations should be 1.0
        np.testing.assert_allclose(corr, 1.0, atol=1e-5)

    def test_perfect_anticorrelation(self):
        """Test perfect negative correlation."""
        base = np.random.randn(1, 100)
        data = np.vstack([base, -base])  # Perfectly anticorrelated
        data_std = standardise_cpu(data)
        corr = compute_correlation_cpu(data_std)

        # Off-diagonal should be -1.0
        expected = np.array([[1.0, -1.0], [-1.0, 1.0]])
        np.testing.assert_allclose(corr, expected, atol=1e-5)

    def test_uncorrelated_data(self):
        """Test that uncorrelated data produces near-zero correlations."""
        np.random.seed(42)
        # Create independent random data
        data = np.random.randn(100, 5000)  # Many timepoints for statistical power
        data_std = standardise_cpu(data)
        corr = compute_correlation_cpu(data_std)

        # Off-diagonal elements should be close to 0 (within statistical noise)
        off_diag = corr[np.triu_indices(100, k=1)]
        assert np.mean(np.abs(off_diag)) < 0.1  # Average should be small

    def test_invalid_1d_input(self):
        """Test error handling for 1D input."""
        data = np.random.randn(100)

        with pytest.raises(ValueError, match="Expected 2D array"):
            compute_correlation_cpu(data)

    def test_invalid_single_timepoint(self):
        """Test error handling for single timepoint."""
        data = np.random.randn(100, 1)

        with pytest.raises(ValueError, match="at least 2 timepoints"):
            compute_correlation_cpu(data)

    def test_small_data(self):
        """Test on minimal valid data."""
        data = np.random.randn(10, 10)
        data_std = standardise_cpu(data)
        corr = compute_correlation_cpu(data_std)

        assert corr.shape == (10, 10)
        np.testing.assert_allclose(np.diag(corr), 1.0, rtol=1e-5)

    def test_deterministic(self):
        """Test deterministic output for same input."""
        data = np.random.randn(50, 100)
        data_std = standardise_cpu(data)

        result1 = compute_correlation_cpu(data_std)
        result2 = compute_correlation_cpu(data_std)

        np.testing.assert_array_equal(result1, result2)


@requires_gpu
class TestCorrelationGPU:
    """Test suite for GPU correlation computation."""

    def test_basic_correlation(self):
        """Test basic correlation computation on GPU."""
        data = cp.random.randn(100, 200)
        data_std = standardise_gpu(data)
        corr = compute_correlation_gpu(data_std)

        # Check shape
        assert corr.shape == (100, 100)

        # Check symmetry
        assert cp.allclose(corr, corr.T, rtol=1e-10)

        # Check diagonal
        assert cp.allclose(cp.diag(corr), 1.0, rtol=1e-10)

    def test_cpu_gpu_consistency(self):
        """Test that CPU and GPU produce identical results."""
        np.random.seed(42)
        data_cpu = np.random.randn(100, 200)
        data_std_cpu = standardise_cpu(data_cpu)
        corr_cpu = compute_correlation_cpu(data_std_cpu)

        data_gpu = cp.asarray(data_cpu)
        data_std_gpu = standardise_gpu(data_gpu)
        corr_gpu = cp.asnumpy(compute_correlation_gpu(data_std_gpu))

        np.testing.assert_allclose(corr_cpu, corr_gpu, rtol=1e-5)

    def test_large_matrix(self):
        """Test correlation on larger data."""
        data = cp.random.randn(500, 300)
        data_std = standardise_gpu(data)
        corr = compute_correlation_gpu(data_std)

        assert corr.shape == (500, 500)
        assert cp.allclose(cp.diag(corr), 1.0, rtol=1e-5)

    def test_invalid_input(self):
        """Test error handling for invalid input."""
        data = cp.random.randn(100)  # 1D

        with pytest.raises(ValueError, match="Expected 2D array"):
            compute_correlation_gpu(data)


class TestCorrelationWrapper:
    """Test suite for correlation wrapper function."""

    def test_cpu_mode(self):
        """Test wrapper with CPU mode."""
        data = np.random.randn(50, 100)
        data_std = standardise_cpu(data)
        result = compute_correlation(data_std, use_gpu=False)

        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)

    @requires_gpu
    def test_gpu_mode(self):
        """Test wrapper with GPU mode."""
        data = np.random.randn(50, 100)
        data_std = standardise_cpu(data)
        result = compute_correlation(data_std, use_gpu=True)

        assert isinstance(result, cp.ndarray)
        assert result.shape == (50, 50)

    def test_gpu_mode_raises_without_cupy(self):
        """Test that GPU mode raises error when CuPy unavailable."""
        if not CUPY_AVAILABLE:
            data = np.random.randn(50, 100)

            with pytest.raises(ImportError):
                compute_correlation(data, use_gpu=True)


class TestCorrelationProperties:
    """Test mathematical properties of correlation matrices."""

    def test_positive_semidefinite(self):
        """Test that correlation matrix is positive semidefinite."""
        data = np.random.randn(50, 100)
        data_std = standardise_cpu(data)
        corr = compute_correlation_cpu(data_std)

        # Check eigenvalues are non-negative
        eigenvalues = np.linalg.eigvalsh(corr)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors

    def test_correlation_with_numpy_corrcoef(self):
        """Test that our implementation matches numpy.corrcoef."""
        data = np.random.randn(50, 100)
        data_std = standardise_cpu(data)
        our_corr = compute_correlation_cpu(data_std)

        # NumPy's corrcoef
        numpy_corr = np.corrcoef(data_std)

        np.testing.assert_allclose(our_corr, numpy_corr, rtol=1e-10)

    def test_sum_of_squares(self):
        """Test that sum of squared correlations relates to variance."""
        data = np.random.randn(100, 500)
        data_std = standardise_cpu(data)
        corr = compute_correlation_cpu(data_std)

        # Each row should have reasonable sum (trace is n_voxels)
        assert np.trace(corr) == pytest.approx(100.0, rel=1e-5)


class TestCorrelationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_two_voxels_only(self):
        """Test minimal case with only 2 voxels."""
        data = np.random.randn(2, 100)
        data_std = standardise_cpu(data)
        corr = compute_correlation_cpu(data_std)

        assert corr.shape == (2, 2)
        assert corr[0, 0] == pytest.approx(1.0)
        assert corr[1, 1] == pytest.approx(1.0)
        assert corr[0, 1] == corr[1, 0]  # Symmetry

    def test_many_voxels_few_timepoints(self):
        """Test case with more voxels than timepoints."""
        # This is common in brain data
        data = np.random.randn(1000, 100)
        data_std = standardise_cpu(data)
        corr = compute_correlation_cpu(data_std)

        assert corr.shape == (1000, 1000)
        np.testing.assert_allclose(np.diag(corr), 1.0, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
