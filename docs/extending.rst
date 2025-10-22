Extending gale: Adding New Graph Metrics
=========================================

**gale** is designed with modularity in mind, making it easy to extend with new graph theory metrics beyond degree centrality. This guide demonstrates how to add new metrics while maintaining compatibility with the existing CPU/GPU architecture.

Overview of gale's Modular Design
----------------------------------

The framework is organized into four main modules:

1. **Preprocessing** (``src/preprocessing/``): Data standardization
2. **Correlation** (``src/gpu_engine/``): Pearson correlation computation
3. **Connectivity** (``src/connectivity/``): Graph metrics computation
4. **Experiments** (``src/experiments/``): Pipeline orchestration

To add a new metric, you'll primarily work in the ``connectivity`` module.

Example: Implementing Eigenvector Centrality
---------------------------------------------

Eigenvector centrality measures a node's influence based on its connections to other high-influence nodes. Let's implement both CPU and GPU versions.

Step 1: Create the Module File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create ``src/connectivity/eigenvector_centrality.py``:

.. code-block:: python

    """
    Eigenvector centrality computation for brain connectivity networks.

    Eigenvector centrality measures node importance based on connections
    to other important nodes, using the principal eigenvector of the
    adjacency matrix.
    """

    import numpy as np
    from typing import Union, Tuple

    try:
        import cupy as cp
        CUPY_AVAILABLE = True
    except ImportError:
        CUPY_AVAILABLE = False
        cp = None


Step 2: Implement CPU Version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def eigenvector_centrality_cpu(adj_matrix: np.ndarray,
                                     max_iter: int = 100,
                                     tol: float = 1e-6) -> np.ndarray:
        """
        Compute eigenvector centrality using power iteration (CPU).

        Args:
            adj_matrix: Weighted adjacency matrix (n_voxels, n_voxels).
            max_iter: Maximum number of iterations. Defaults to 100.
            tol: Convergence tolerance. Defaults to 1e-6.

        Returns:
            Eigenvector centrality values of shape (n_voxels,).

        Raises:
            ValueError: If input is not a 2D square matrix.

        Examples:
            >>> import numpy as np
            >>> from src.connectivity.degree_centrality import threshold_matrix_cpu
            >>> corr = np.random.randn(100, 100)
            >>> corr = (corr + corr.T) / 2
            >>> adj = threshold_matrix_cpu(corr, threshold=0.2)
            >>> eigen_cent = eigenvector_centrality_cpu(adj)
            >>> eigen_cent.shape
            (100,)

        References:
            Lohmann G, et al. (2010). Neuroimage. Eigenvector centrality
            mapping for analyzing connectivity patterns in fMRI data.
        """
        if adj_matrix.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {adj_matrix.shape}")

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError(
                f"Expected square matrix, got shape {adj_matrix.shape}"
            )

        n = adj_matrix.shape[0]

        # Initialize with uniform distribution
        x = np.ones(n) / n

        # Power iteration
        for iteration in range(max_iter):
            x_new = adj_matrix @ x

            # Normalize
            norm = np.linalg.norm(x_new)
            if norm < 1e-12:
                # Degenerate case: no connections
                return np.zeros(n)

            x_new = x_new / norm

            # Check convergence
            if np.linalg.norm(x_new - x) < tol:
                break

            x = x_new

        return x


Step 3: Implement GPU Version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def eigenvector_centrality_gpu(adj_matrix: 'cp.ndarray',
                                     max_iter: int = 100,
                                     tol: float = 1e-6) -> 'cp.ndarray':
        """
        Compute eigenvector centrality using power iteration (GPU).

        GPU-accelerated version using CuPy.

        Args:
            adj_matrix: Weighted adjacency matrix (CuPy array).
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance.

        Returns:
            Eigenvector centrality values (CuPy array) of shape (n_voxels,).

        Raises:
            ImportError: If CuPy is not available.
            ValueError: If input is invalid.
        """
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not available.")

        if adj_matrix.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {adj_matrix.shape}")

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError(
                f"Expected square matrix, got shape {adj_matrix.shape}"
            )

        n = adj_matrix.shape[0]
        x = cp.ones(n) / n

        for iteration in range(max_iter):
            x_new = adj_matrix @ x
            norm = cp.linalg.norm(x_new)

            if norm < 1e-12:
                return cp.zeros(n)

            x_new = x_new / norm

            if cp.linalg.norm(x_new - x) < tol:
                break

            x = x_new

        return x


Step 4: Add Convenience Wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def eigenvector_centrality(adj_matrix: Union[np.ndarray, 'cp.ndarray'],
                                use_gpu: bool = False,
                                max_iter: int = 100,
                                tol: float = 1e-6) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Compute eigenvector centrality, automatically selecting CPU or GPU.

        Args:
            adj_matrix: Weighted adjacency matrix.
            use_gpu: If True, use GPU computation. Defaults to False.
            max_iter: Maximum iterations.
            tol: Convergence tolerance.

        Returns:
            Eigenvector centrality values of same type as input.
        """
        if use_gpu:
            if not CUPY_AVAILABLE:
                raise ImportError("CuPy is not available.")
            if not isinstance(adj_matrix, cp.ndarray):
                adj_matrix = cp.asarray(adj_matrix)
            return eigenvector_centrality_gpu(adj_matrix, max_iter, tol)
        else:
            if CUPY_AVAILABLE and isinstance(adj_matrix, cp.ndarray):
                adj_matrix = cp.asnumpy(adj_matrix)
            return eigenvector_centrality_cpu(adj_matrix, max_iter, tol)


Step 5: Integrate into Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Update ``src/experiments/run_experiments.py`` to include the new metric:

.. code-block:: python

    from src.connectivity.eigenvector_centrality import eigenvector_centrality_cpu, eigenvector_centrality_gpu

    def compute_cpu(data, threshold=0.2, compute_eigenvector=False):
        # ... existing code ...
        degree = degree_centrality_cpu(corr_thresh)

        results = {'degree': degree}

        if compute_eigenvector:
            eigen_cent = eigenvector_centrality_cpu(corr_thresh)
            results['eigenvector'] = eigen_cent

        return results


Step 6: Write Tests
~~~~~~~~~~~~~~~~~~~

Create ``tests/test_eigenvector_centrality.py``:

.. code-block:: python

    import numpy as np
    import pytest
    from src.connectivity.eigenvector_centrality import (
        eigenvector_centrality_cpu, CUPY_AVAILABLE
    )

    class TestEigenvectorCentralityCPU:
        def test_star_network(self):
            """Hub node should have highest eigenvector centrality."""
            n = 10
            adj = np.eye(n)
            adj[0, 1:] = 1.0
            adj[1:, 0] = 1.0

            eigen = eigenvector_centrality_cpu(adj)

            assert eigen[0] == max(eigen)  # Hub has highest centrality

        def test_symmetric_result(self):
            """Eigenvector centrality should be non-negative."""
            adj = np.random.randn(50, 50)
            adj = np.abs((adj + adj.T) / 2)

            eigen = eigenvector_centrality_cpu(adj)

            assert np.all(eigen >= 0)


Adding Other Metrics
---------------------

The same pattern applies to other graph metrics:

Betweenness Centrality
~~~~~~~~~~~~~~~~~~~~~~

Measures nodes that lie on many shortest paths:

.. code-block:: python

    # src/connectivity/betweenness_centrality.py
    def betweenness_centrality_cpu(adj_matrix):
        # Implement Floyd-Warshall or Dijkstra's algorithm
        # Calculate shortest paths
        # Count paths through each node
        pass

Closeness Centrality
~~~~~~~~~~~~~~~~~~~~

Measures average distance to all other nodes:

.. code-block:: python

    # src/connectivity/closeness_centrality.py
    def closeness_centrality_cpu(adj_matrix):
        # Calculate shortest path distances
        # Compute average distance for each node
        pass

Clustering Coefficient
~~~~~~~~~~~~~~~~~~~~~~~

Measures local network clustering:

.. code-block:: python

    # src/connectivity/clustering.py
    def clustering_coefficient_cpu(adj_matrix):
        # Count triangles involving each node
        # Normalize by possible triangles
        pass


Best Practices
--------------

1. **Dual Implementation**: Always provide both CPU and GPU versions
2. **Type Hints**: Use type annotations for all functions
3. **Docstrings**: Include comprehensive docstrings with references
4. **Input Validation**: Check array dimensions and values
5. **Error Handling**: Raise descriptive errors for invalid inputs
6. **Testing**: Write tests covering:
   - Basic functionality
   - Edge cases (empty graphs, isolated nodes)
   - Mathematical properties (symmetry, bounds)
   - CPU-GPU consistency
7. **References**: Cite academic papers in docstrings
8. **Performance**: Profile for bottlenecks, optimize hot loops

GPU Optimization Tips
---------------------

**Memory Management**

.. code-block:: python

    # Minimize CPU-GPU transfers
    data_gpu = cp.asarray(data_cpu)  # Transfer once
    result1 = compute_metric1_gpu(data_gpu)
    result2 = compute_metric2_gpu(data_gpu)
    # Transfer results back
    results_cpu = {
        'metric1': cp.asnumpy(result1),
        'metric2': cp.asnumpy(result2)
    }

**Kernel Fusion**

.. code-block:: python

    # Combine operations to reduce kernel launches
    # Bad:
    x = adj_matrix @ vector
    x = x / norm

    # Good:
    x = (adj_matrix @ vector) / norm

**Synchronization**

.. code-block:: python

    # Ensure GPU operations complete before timing
    cp.cuda.Stream.null.synchronize()
    start = time.time()
    result = my_gpu_function(data_gpu)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start

Contributing Your Metric
-------------------------

To contribute a new metric to **gale**:

1. Fork the repository
2. Create a feature branch (``git checkout -b feature/betweenness-centrality``)
3. Implement the metric following the pattern above
4. Write comprehensive tests (aim for >90% coverage)
5. Update documentation
6. Submit a pull request

See the `Contributing Guide <contributing.html>`_ for details.

References
----------

- Lohmann G, et al. (2010). Eigenvector centrality mapping for analyzing connectivity patterns in fMRI data. *NeuroImage*.
- Rubinov M, Sporns O (2010). Complex network measures of brain connectivity. *NeuroImage*.
- Bullmore E, Sporns O (2009). Complex brain networks: graph theoretical analysis. *Nature Reviews Neuroscience*.
