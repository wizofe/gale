gale: GPU-Accelerated Brain Connectivity Analysis
==================================================

Welcome to the **gale** framework documentation.

**gale** is a high-performance Python framework for computing voxelwise brain
connectivity metrics using GPU-accelerated computing.

Features
--------

* GPU-accelerated correlation computation using CuPy
* Graceful CPU fallback for systems without CUDA
* Modular design for preprocessing, connectivity, and graph metrics
* Comprehensive test suite with 74+ tests
* Publication-quality code with type hints and documentation

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples

Installation
------------

See :doc:`installation` for detailed installation instructions.

Quick Start
-----------

.. code-block:: bash

   # Install CPU-only version
   pip install -e .

   # Run analysis
   gale --n_voxels 1000 --n_timepoints 200 --cpu-only

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
