import cupy as cp

def compute_correlation_gpu(data):
    """
    Compute the correlation matrix for a GPU array using CuPy.
    Assumes the data is standardised (zero mean, unit variance).
    """
    n_timepoints = data.shape[1]
    corr_matrix = cp.dot(data, data.T) / (n_timepoints - 1)
    return corr_matrix
