"""
Comprehensive test suite for data standardization module.

Tests CPU and GPU implementations of z-score normalization with edge cases,
error handling, and numerical accuracy validation.
"""

import numpy as np
import pytest

from src.preprocessing.standardise import (
    standardise_cpu, standardise_gpu, standardise, CUPY_AVAILABLE
)

# Skip GPU tests if CuPy is not available
if CUPY_AVAILABLE:
    import cupy as cp
    from src.utils.gpu_utils import check_gpu_availability
    GPU_AVAILABLE, _ = check_gpu_availability()
else:
    GPU_AVAILABLE = False

# Pytest markers
requires_gpu = pytest.mark.skipif(
    not GPU_AVAILABLE,
    reason="GPU/CUDA not available"
)


class TestStandardiseCPU:
    """Test suite for CPU standardization function."""

    def test_basic_standardization(self):
        """Test basic standardization on random data."""
        data = np.random.randn(100, 200)
        result = standardise_cpu(data)

        # Check shape preservation
        assert result.shape == data.shape

        # Check zero mean (per voxel)
        assert np.allclose(result.mean(axis=1), 0, atol=1e-10)

        # Check unit variance (per voxel)
        assert np.allclose(result.std(axis=1), 1, atol=1e-10)

    def test_small_data(self):
        """Test on minimal valid data."""
        data = np.random.randn(10, 10)
        result = standardise_cpu(data)

        assert result.shape == (10, 10)
        assert np.allclose(result.mean(axis=1), 0, atol=1e-10)

    def test_large_timepoints(self):
        """Test with large number of timepoints."""
        data = np.random.randn(50, 1000)
        result = standardise_cpu(data)

        assert result.shape == (50, 1000)
        assert np.allclose(result.mean(axis=1), 0, atol=1e-10)
        assert np.allclose(result.std(axis=1), 1, atol=1e-10)

    def test_invalid_1d_input(self):
        """Test error handling for 1D input."""
        data = np.random.randn(100)

        with pytest.raises(ValueError, match="Expected 2D array"):
            standardise_cpu(data)

    def test_invalid_3d_input(self):
        """Test error handling for 3D input."""
        data = np.random.randn(10, 10, 10)

        with pytest.raises(ValueError, match="Expected 2D array"):
            standardise_cpu(data)

    def test_constant_timeseries_raises_error(self):
        """Test that constant timeseries raises ZeroDivisionError."""
        data = np.ones((5, 10))  # All constant

        with pytest.raises(ZeroDivisionError, match="zero standard deviation"):
            standardise_cpu(data)

    def test_mixed_constant_and_variable(self):
        """Test that mixed data (some constant voxels) raises error."""
        data = np.random.randn(5, 10)
        data[2, :] = 1.0  # One constant voxel

        with pytest.raises(ZeroDivisionError):
            standardise_cpu(data)

    def test_positive_data(self):
        """Test standardization on positive-only data."""
        data = np.abs(np.random.randn(50, 100))
        result = standardise_cpu(data)

        assert np.allclose(result.mean(axis=1), 0, atol=1e-10)
        assert np.allclose(result.std(axis=1), 1, atol=1e-10)

    def test_negative_data(self):
        """Test standardization on negative-only data."""
        data = -np.abs(np.random.randn(50, 100))
        result = standardise_cpu(data)

        assert np.allclose(result.mean(axis=1), 0, atol=1e-10)
        assert np.allclose(result.std(axis=1), 1, atol=1e-10)

    def test_deterministic(self):
        """Test that same input produces same output."""
        np.random.seed(42)
        data = np.random.randn(30, 50)

        result1 = standardise_cpu(data)
        result2 = standardise_cpu(data)

        np.testing.assert_array_equal(result1, result2)


@requires_gpu
class TestStandardiseGPU:
    """Test suite for GPU standardization function."""

    def test_basic_standardization(self):
        """Test basic standardization on random data."""
        data = cp.random.randn(100, 200)
        result = standardise_gpu(data)

        # Check shape preservation
        assert result.shape == data.shape

        # Check zero mean (per voxel)
        assert cp.allclose(result.mean(axis=1), 0, atol=1e-10)

        # Check unit variance (per voxel)
        assert cp.allclose(result.std(axis=1), 1, atol=1e-10)

    def test_cpu_gpu_consistency(self):
        """Test that CPU and GPU produce identical results."""
        np.random.seed(42)
        data_cpu = np.random.randn(100, 200)
        data_gpu = cp.asarray(data_cpu)

        result_cpu = standardise_cpu(data_cpu)
        result_gpu = cp.asnumpy(standardise_gpu(data_gpu))

        np.testing.assert_allclose(result_cpu, result_gpu, rtol=1e-5)

    def test_invalid_input(self):
        """Test error handling for invalid input."""
        data = cp.random.randn(100)  # 1D

        with pytest.raises(ValueError, match="Expected 2D array"):
            standardise_gpu(data)

    def test_constant_timeseries_raises_error(self):
        """Test that constant timeseries raises ZeroDivisionError."""
        data = cp.ones((5, 10))

        with pytest.raises(ZeroDivisionError):
            standardise_gpu(data)


class TestStandardiseWrapperFunction:
    """Test suite for the wrapper function that selects CPU/GPU."""

    def test_cpu_mode(self):
        """Test wrapper with CPU mode."""
        data = np.random.randn(50, 100)
        result = standardise(data, use_gpu=False)

        assert isinstance(result, np.ndarray)
        assert np.allclose(result.mean(axis=1), 0, atol=1e-10)

    @requires_gpu
    def test_gpu_mode(self):
        """Test wrapper with GPU mode."""
        data = np.random.randn(50, 100)
        result = standardise(data, use_gpu=True)

        # Result should be CuPy array
        assert isinstance(result, cp.ndarray)

        result_cpu = cp.asnumpy(result)
        assert np.allclose(result_cpu.mean(axis=1), 0, atol=1e-10)

    def test_gpu_mode_raises_without_cupy(self):
        """Test that GPU mode raises error when CuPy unavailable."""
        if not CUPY_AVAILABLE:
            data = np.random.randn(50, 100)

            with pytest.raises(ImportError, match="CuPy is not available"):
                standardise(data, use_gpu=True)

    def test_invalid_input(self):
        """Test wrapper error handling."""
        data = np.random.randn(100)  # 1D

        with pytest.raises(ValueError):
            standardise(data, use_gpu=False)


class TestStandardisationProperties:
    """Test mathematical properties of standardization."""

    def test_idempotence(self):
        """Test that standardizing twice gives same result."""
        data = np.random.randn(50, 100)
        result1 = standardise_cpu(data)

        # Standardizing already-standardized data should give same result
        # (within numerical precision)
        result2 = standardise_cpu(result1)

        np.testing.assert_allclose(result1, result2, atol=1e-5)

    def test_shift_invariance(self):
        """Test that adding constant to all timepoints doesn't affect standardization pattern."""
        data = np.random.randn(50, 100)
        result1 = standardise_cpu(data)

        # Add different constants to each voxel
        shift = np.random.randn(50, 1)
        data_shifted = data + shift
        result2 = standardise_cpu(data_shifted)

        # Results should be identical (standardization removes mean)
        np.testing.assert_allclose(result1, result2, rtol=1e-10)

    def test_scale_invariance_pattern(self):
        """Test that scaling affects magnitude but not correlation pattern."""
        data = np.random.randn(50, 100)
        result1 = standardise_cpu(data)

        # Scale each voxel by different factor
        scale = np.abs(np.random.randn(50, 1)) + 0.1  # Avoid zero
        data_scaled = data * scale
        result2 = standardise_cpu(data_scaled)

        # Standardized results should be identical
        np.testing.assert_allclose(result1, result2, rtol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
