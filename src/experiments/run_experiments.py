import argparse
import time
import numpy as np
import cupy as cp
from src.preprocessing.standardise import standardise_cpu, standardise_gpu
from src.gpu_engine.correlation import compute_correlation_gpu
from src.connectivity.degree_centrality import threshold_matrix_gpu, degree_centrality_gpu, threshold_matrix_cpu, degree_centrality_cpu

def compute_cpu(data, threshold=0.2):
    # Standardise data on CPU
    data_std = standardise_cpu(data)
    n_timepoints = data_std.shape[1]
    # Compute correlation using NumPy
    corr_matrix = np.dot(data_std, data_std.T) / (n_timepoints - 1)
    # Apply thresholding
    corr_thresh = threshold_matrix_cpu(corr_matrix, threshold)
    # Compute degree centrality
    degree = degree_centrality_cpu(corr_thresh)
    return degree

def compute_gpu(data, threshold=0.2):
    # Transfer data to GPU and standardise
    data_gpu = cp.asarray(data)
    data_std = standardise_gpu(data_gpu)
    n_timepoints = data_std.shape[1]
    # Compute correlation using GPU
    corr_matrix = compute_correlation_gpu(data_std)
    # Apply thresholding
    corr_thresh = threshold_matrix_gpu(corr_matrix, threshold)
    # Compute degree centrality
    degree = degree_centrality_gpu(corr_thresh)
    # Transfer result back to CPU
    return cp.asnumpy(degree)

def main():
    parser = argparse.ArgumentParser(description="Run HERMES experiments.")
    parser.add_argument('--data_path', type=str, default=None,
                        help="Path to NIfTI file. If not provided, synthetic data is used.")
    parser.add_argument('--n_voxels', type=int, default=5000,
                        help="Number of voxels for synthetic data")
    parser.add_argument('--n_timepoints', type=int, default=200,
                        help="Number of timepoints for synthetic data")
    parser.add_argument('--threshold', type=float, default=0.2,
                        help="Correlation threshold")
    args = parser.parse_args()

    if args.data_path:
        # For now, NIfTI loading is not implemented; use synthetic data.
        print("NIfTI data loading is not yet implemented. Using synthetic data.")
        data = np.random.randn(args.n_voxels, args.n_timepoints)
    else:
        print("Using synthetic data.")
        data = np.random.randn(args.n_voxels, args.n_timepoints)

    print("Running CPU implementation...")
    start_cpu = time.time()
    degree_cpu = compute_cpu(data, args.threshold)
    time_cpu = time.time() - start_cpu
    print(f"CPU degree centrality computed in {time_cpu:.4f} seconds.")

    print("Running GPU implementation...")
    cp.cuda.Stream.null.synchronize()
    start_gpu = time.time()
    degree_gpu = compute_gpu(data, args.threshold)
    cp.cuda.Stream.null.synchronize()
    time_gpu = time.time() - start_gpu
    print(f"GPU degree centrality computed in {time_gpu:.4f} seconds.")

    # Validate results
    correlation = np.corrcoef(degree_cpu, degree_gpu)[0, 1]
    print(f"Correlation between CPU and GPU degree centrality: {correlation:.6f}")

if __name__ == '__main__':
    main()
