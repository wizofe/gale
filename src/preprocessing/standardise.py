import numpy as np
import cupy as cp

def standardise_cpu(data):
    """
    Standardise a NumPy array to zero mean and unit variance.
    """
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    return (data - mean) / std

def standardise_gpu(data):
    """
    Standardise a CuPy array to zero mean and unit variance.
    """
    mean = cp.mean(data, axis=1, keepdims=True)
    std = cp.std(data, axis=1, keepdims=True)
    return (data - mean) / std
