"""
Test suite for GPU utility functions.

Tests GPU availability detection, error handling, and graceful fallback.
"""

import pytest
from src.utils.gpu_utils import (
    check_gpu_availability, get_array_module, warn_gpu_unavailable,
    get_gpu_info, CUPY_AVAILABLE
)

import numpy as np


class TestGPUAvailability:
    """Test GPU availability checking."""

    def test_check_gpu_availability_returns_tuple(self):
        """Test that check_gpu_availability returns a tuple."""
        result = check_gpu_availability()

        assert isinstance(result, tuple)
        assert len(result) == 2

        is_available, error_msg = result
        assert isinstance(is_available, bool)

        if not is_available:
            assert error_msg is not None
            assert isinstance(error_msg, str)
        else:
            assert error_msg is None

    def test_check_gpu_availability_consistency(self):
        """Test that repeated calls give consistent results."""
        result1 = check_gpu_availability()
        result2 = check_gpu_availability()

        assert result1[0] == result2[0]  # Same availability status

    def test_cupy_not_available_handling(self):
        """Test handling when CuPy is not installed."""
        if not CUPY_AVAILABLE:
            is_available, error_msg = check_gpu_availability()

            assert is_available is False
            assert "CuPy not available" in error_msg


class TestGetArrayModule:
    """Test array module selection (NumPy/CuPy)."""

    def test_get_array_module_returns_numpy_when_gpu_false(self):
        """Test that NumPy is returned when use_gpu=False."""
        xp = get_array_module(use_gpu=False)

        assert xp is np
        assert xp.__name__ == 'numpy'

    def test_get_array_module_with_gpu_preference(self):
        """Test module selection with GPU preference."""
        xp = get_array_module(use_gpu=True)

        # Should return either cupy or numpy with warning
        assert xp.__name__ in ['cupy', 'numpy']

        if not CUPY_AVAILABLE:
            # Should fallback to NumPy
            assert xp is np

    def test_array_module_creates_arrays(self):
        """Test that returned module can create arrays."""
        xp = get_array_module(use_gpu=False)
        arr = xp.zeros((10, 10))

        assert arr.shape == (10, 10)
        assert xp.all(arr == 0)

    def test_numpy_fallback_with_warning(self):
        """Test that NumPy fallback produces warning when GPU unavailable."""
        if not CUPY_AVAILABLE:
            # When CuPy not installed, we don't get a warning - just silent fallback
            xp = get_array_module(use_gpu=True)
            assert xp is np
        else:
            # When CuPy installed but GPU unavailable, we should get warning
            from src.utils.gpu_utils import check_gpu_availability
            is_available, _ = check_gpu_availability()
            if not is_available:
                with pytest.warns(UserWarning, match="GPU requested but not available"):
                    xp = get_array_module(use_gpu=True)
                    assert xp is np


class TestWarnGPUUnavailable:
    """Test GPU unavailability warning function."""

    def test_warn_gpu_unavailable_produces_warning(self):
        """Test that warning is issued."""
        with pytest.warns(UserWarning, match="GPU/CUDA not available"):
            warn_gpu_unavailable()

    def test_warn_with_reason(self):
        """Test warning with specific reason."""
        with pytest.warns(UserWarning, match="Test reason"):
            warn_gpu_unavailable(reason="Test reason")


class TestGetGPUInfo:
    """Test GPU information retrieval."""

    def test_get_gpu_info_returns_dict(self):
        """Test that get_gpu_info returns a dictionary."""
        info = get_gpu_info()

        assert isinstance(info, dict)
        assert 'available' in info
        assert isinstance(info['available'], bool)

    def test_get_gpu_info_unavailable_structure(self):
        """Test structure when GPU is unavailable."""
        if not CUPY_AVAILABLE:
            info = get_gpu_info()

            assert info['available'] is False
            assert 'error' in info
            assert info['device_count'] == 0

    def test_get_gpu_info_available_structure(self):
        """Test structure when GPU is available."""
        is_available, _ = check_gpu_availability()

        if is_available:
            info = get_gpu_info()

            assert info['available'] is True
            assert 'device_id' in info
            assert 'name' in info
            assert 'compute_capability' in info
            assert 'total_memory_gb' in info
            assert 'free_memory_gb' in info
            assert 'device_count' in info

            # Validate types
            assert isinstance(info['device_id'], int)
            assert isinstance(info['name'], str)
            assert isinstance(info['total_memory_gb'], (int, float))
            assert isinstance(info['free_memory_gb'], (int, float))

    def test_get_gpu_info_memory_values(self):
        """Test that memory values are reasonable."""
        is_available, _ = check_gpu_availability()

        if is_available:
            info = get_gpu_info()

            # Memory should be positive
            assert info['total_memory_gb'] > 0
            assert info['free_memory_gb'] >= 0

            # Free memory should not exceed total
            assert info['free_memory_gb'] <= info['total_memory_gb']


class TestGPUUtilsIntegration:
    """Integration tests for GPU utilities."""

    def test_module_selection_and_array_creation(self):
        """Test end-to-end module selection and array creation."""
        xp = get_array_module(use_gpu=False)
        data = xp.random.randn(100, 200)

        assert data.shape == (100, 200)
        assert xp is np

    def test_consistent_behavior_across_calls(self):
        """Test that behavior is consistent across multiple calls."""
        xp1 = get_array_module(use_gpu=False)
        xp2 = get_array_module(use_gpu=False)

        assert xp1 is xp2

    def test_gpu_info_and_availability_consistency(self):
        """Test that GPU info and availability are consistent."""
        is_available, error_msg = check_gpu_availability()
        info = get_gpu_info()

        assert is_available == info['available']

        if not is_available:
            assert 'error' in info
            assert info['device_count'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
