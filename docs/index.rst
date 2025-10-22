GALE: GPU-Accelerated Large-scale Exploration
==============================================

Welcome to the **GALE** framework documentation.

**GALE** (**G**\PU-**A**\ccelerated **L**\arge-scale **E**\xploration) is a high-performance Python framework for computing voxelwise brain connectivity metrics using GPU-accelerated computing.

.. note::
   **Performance**: GALE achieves speedups exceeding **100×** over NumPy (single-core) and **50×** over AFNI (64-core parallelized) across all tested voxel sizes.

Features
--------

* 🚀 **GPU-Accelerated Computing**: Leverage NVIDIA GPUs with CuPy for massive speedups
* 💻 **CPU Fallback**: Graceful degradation for systems without CUDA (e.g., macOS)
* 🧪 **Publication-Quality**: Comprehensive test suite (74 tests), type hints, extensive documentation
* 📊 **Graph Theory Metrics**: Weighted degree centrality for brain connectivity
* 🔬 **Neuroimaging-Ready**: Built for fMRI data with NIfTI support
* 🎯 **Reproducible**: Deterministic results with seed control
* 🛠️ **Modular Design**: Easy to extend with new metrics (see :doc:`extending`)

Quick Links
-----------

* **GitHub**: https://github.com/wizofe/gale
* **Documentation**: https://gale.readthedocs.io
* **Issue Tracker**: https://github.com/wizofe/gale/issues

Getting Started
---------------

.. code-block:: bash

   # Install CPU-only version
   pip install -e .

   # Run analysis
   gale --n_voxels 1000 --n_timepoints 200 --cpu-only

   # Check GPU availability
   gale --show-gpu-info

For detailed installation instructions, see :doc:`installation`.

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   extending

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api

Performance Highlights
----------------------

From synthetic fMRI benchmarks (timings in milliseconds):

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Voxels
     - NumPy (CPU)
     - GALE (GPU)
     - Speedup
     - vs AFNI
   * - 10,000
     - 736.19
     - 5.91
     - 124.6×
     - 54.6×
   * - 50,000
     - 17,766.24
     - 19.31
     - 920.1×
     - 412.1×
   * - 100,000
     - 73,662.84
     - 46.66
     - 1,578.9×
     - 725.8×

Citation
--------

If you use **GALE** in your research, please cite:

.. code-block:: bibtex

   @software{gale2025,
     author = {Valasakis, Ioannis},
     title = {GALE: GPU-Accelerated Large-scale Exploration for Brain Connectivity Analysis},
     year = {2025},
     url = {https://github.com/wizofe/gale},
     version = {0.2.0},
     note = {Documentation available at https://gale.readthedocs.io}
   }

For LaTeX documents:

.. code-block:: latex

   \textsc{gale} (GPU-Accelerated Large-scale Exploration) is available at
   \url{https://github.com/wizofe/gale}. Documentation, including installation
   instructions and usage examples, is available at \url{https://gale.readthedocs.io}.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
