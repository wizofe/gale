"""
GPU utility functions for detecting and managing GPU availability.

This module provides utilities for gracefully handling GPU/CUDA availability,
particularly important for cross-platform compatibility (e.g., macOS without NVIDIA GPUs).
"""

import warnings
from typing import Tuple, Optional

# Try to import CuPy, but don't fail if it's not available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except (ImportError, Exception) as e:
    CUPY_AVAILABLE = False
    _CUPY_IMPORT_ERROR = str(e)
    cp = None


def check_gpu_availability() -> Tuple[bool, Optional[str]]:
    """
    Check if GPU/CUDA is available for computation.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing:
            - bool: True if GPU is available and functional, False otherwise
            - Optional[str]: Error message if GPU is not available, None otherwise

    Examples:
        >>> is_available, error_msg = check_gpu_availability()
        >>> if is_available:
        ...     print("GPU is ready!")
        ... else:
        ...     print(f"GPU not available: {error_msg}")
    """
    if not CUPY_AVAILABLE:
        return False, f"CuPy not available: {_CUPY_IMPORT_ERROR}"

    try:
        # Try to access GPU device
        device = cp.cuda.Device()
        # Try to allocate a small test array
        test_array = cp.array([1.0])
        # Clean up
        del test_array
        return True, None
    except Exception as e:
        return False, f"GPU/CUDA runtime error: {str(e)}"


def get_array_module(use_gpu: bool = True):
    """
    Get the appropriate array module (NumPy or CuPy) based on GPU availability.

    Args:
        use_gpu: Whether to prefer GPU if available. Defaults to True.

    Returns:
        module: Either cupy or numpy module, depending on availability and preference.

    Examples:
        >>> xp = get_array_module(use_gpu=True)
        >>> data = xp.random.randn(100, 100)  # Will use GPU if available
    """
    import numpy as np

    if use_gpu and CUPY_AVAILABLE:
        is_available, error_msg = check_gpu_availability()
        if is_available:
            return cp
        else:
            warnings.warn(f"GPU requested but not available: {error_msg}. Falling back to CPU.")
            return np
    else:
        return np


def warn_gpu_unavailable(reason: str = None):
    """
    Issue a warning that GPU is not available.

    Args:
        reason: Optional specific reason why GPU is unavailable.
    """
    msg = "GPU/CUDA not available. Running on CPU only."
    if reason:
        msg += f" Reason: {reason}"
    msg += " This may result in slower performance for large datasets."
    warnings.warn(msg, UserWarning)


def get_gpu_info() -> dict:
    """
    Get information about available GPU(s).

    Returns:
        dict: Dictionary containing GPU information, or error information if unavailable.

    Examples:
        >>> info = get_gpu_info()
        >>> print(f"GPU: {info.get('name', 'N/A')}")
    """
    is_available, error_msg = check_gpu_availability()

    if not is_available:
        return {
            'available': False,
            'error': error_msg,
            'device_count': 0
        }

    try:
        device = cp.cuda.Device()
        return {
            'available': True,
            'device_count': cp.cuda.runtime.getDeviceCount(),
            'device_id': device.id,
            'name': device.name.decode() if isinstance(device.name, bytes) else device.name,
            'compute_capability': device.compute_capability,
            'total_memory_gb': device.mem_info[1] / (1024**3),
            'free_memory_gb': device.mem_info[0] / (1024**3)
        }
    except Exception as e:
        return {
            'available': False,
            'error': f"Error retrieving GPU info: {str(e)}",
            'device_count': 0
        }
