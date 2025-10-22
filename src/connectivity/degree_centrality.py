import cupy as cp
import numpy as np

def threshold_matrix_gpu(corr_matrix, threshold=0.2):
    """
    Apply a threshold to a GPU correlation matrix.
    """
    return cp.where(corr_matrix >= threshold, corr_matrix, 0)

def degree_centrality_gpu(thresholded_matrix):
    """
    Compute the weighted degree centrality from a thresholded GPU matrix.
    """
    degree = cp.sum(thresholded_matrix, axis=1)
    return degree

def threshold_matrix_cpu(corr_matrix, threshold=0.2):
    """
    Apply a threshold to a CPU correlation matrix.
    """
    return np.where(corr_matrix >= threshold, corr_matrix, 0)

def degree_centrality_cpu(thresholded_matrix):
    """
    Compute the weighted degree centrality from a thresholded CPU matrix.
    """
    degree = np.sum(thresholded_matrix, axis=1)
    return degree
