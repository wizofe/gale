"""
Pytest configuration and shared fixtures for neurogale framework tests.

This module provides test fixtures, configuration, and utilities used
across the test suite.
"""

import pytest
import numpy as np
import warnings

from src.utils.gpu_utils import check_gpu_availability, CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU/CUDA"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow-running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "cpu_only: mark test as CPU-only"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark GPU tests and add skip markers."""
    gpu_available, gpu_error = check_gpu_availability()

    for item in items:
        # Auto-mark GPU tests if they use GPU fixtures
        if "gpu" in item.fixturenames:
            item.add_marker(pytest.mark.gpu)

        # Skip GPU tests if GPU not available
        if "gpu" in item.keywords and not gpu_available:
            skip_marker = pytest.mark.skip(
                reason=f"GPU/CUDA not available: {gpu_error}"
            )
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def gpu_available():
    """Session-scoped fixture that checks GPU availability once."""
    is_available, error = check_gpu_availability()
    if not is_available:
        pytest.skip(f"GPU not available: {error}")
    return True


@pytest.fixture
def random_data_small():
    """Generate small random test data (CPU)."""
    np.random.seed(42)
    return np.random.randn(10, 20)


@pytest.fixture
def random_data_medium():
    """Generate medium random test data (CPU)."""
    np.random.seed(42)
    return np.random.randn(100, 200)


@pytest.fixture
def random_data_large():
    """Generate large random test data (CPU)."""
    np.random.seed(42)
    return np.random.randn(1000, 500)


@pytest.fixture
def correlation_matrix_small():
    """Generate small correlation matrix."""
    np.random.seed(42)
    n = 10
    corr = np.random.randn(n, n)
    corr = (corr + corr.T) / 2  # Symmetric
    np.fill_diagonal(corr, 1.0)
    # Ensure valid correlation range
    corr = np.clip(corr, -1, 1)
    return corr


@pytest.fixture
def correlation_matrix_medium():
    """Generate medium correlation matrix."""
    np.random.seed(42)
    n = 100
    corr = np.random.randn(n, n) * 0.3  # Scale to reasonable correlation range
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1, 1)
    return corr


@pytest.fixture
def gpu_random_data_medium(gpu_available):
    """Generate medium random test data (GPU)."""
    if not CUPY_AVAILABLE:
        pytest.skip("CuPy not available")
    cp.random.seed(42)
    return cp.random.randn(100, 200)


@pytest.fixture
def suppress_warnings():
    """Context manager to suppress warnings in tests."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    if CUPY_AVAILABLE:
        cp.random.seed(42)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


# Helper functions for tests

def assert_standardized(data, axis=1, atol=1e-10):
    """
    Assert that data is standardized (zero mean, unit variance).

    Args:
        data: Array to check (NumPy or CuPy)
        axis: Axis along which to check (default: 1 for voxels)
        atol: Absolute tolerance for comparison
    """
    if CUPY_AVAILABLE and isinstance(data, cp.ndarray):
        assert cp.allclose(data.mean(axis=axis), 0, atol=atol), \
            f"Mean not zero: {cp.asnumpy(data.mean(axis=axis))}"
        assert cp.allclose(data.std(axis=axis), 1, atol=atol), \
            f"Std not one: {cp.asnumpy(data.std(axis=axis))}"
    else:
        assert np.allclose(data.mean(axis=axis), 0, atol=atol), \
            f"Mean not zero: {data.mean(axis=axis)}"
        assert np.allclose(data.std(axis=axis), 1, atol=atol), \
            f"Std not one: {data.std(axis=axis)}"


def assert_correlation_matrix_properties(corr, rtol=1e-5):
    """
    Assert that matrix has valid correlation properties.

    Args:
        corr: Correlation matrix (NumPy or CuPy)
        rtol: Relative tolerance
    """
    if CUPY_AVAILABLE and isinstance(corr, cp.ndarray):
        # Symmetric
        assert cp.allclose(corr, corr.T, rtol=rtol), "Matrix not symmetric"
        # Diagonal is 1.0
        assert cp.allclose(cp.diag(corr), 1.0, rtol=rtol), "Diagonal not 1.0"
        # Range [-1, 1]
        assert cp.all(corr >= -1.0) and cp.all(corr <= 1.0), "Values outside [-1, 1]"
    else:
        # Symmetric
        assert np.allclose(corr, corr.T, rtol=rtol), "Matrix not symmetric"
        # Diagonal is 1.0
        assert np.allclose(np.diag(corr), 1.0, rtol=rtol), "Diagonal not 1.0"
        # Range [-1, 1]
        assert np.all(corr >= -1.0) and np.all(corr <= 1.0), "Values outside [-1, 1]"


# Export helper functions
__all__ = [
    'assert_standardized',
    'assert_correlation_matrix_properties',
]
