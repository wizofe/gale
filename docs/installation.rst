Installation
============

**neurogale** (GPU-Accelerated Large-scale Exploration) can be installed in multiple ways depending on your needs and environment.

Requirements
------------

Minimum Requirements
~~~~~~~~~~~~~~~~~~~~

- Python 3.8 or higher
- NumPy ≥ 1.20.0
- Nibabel ≥ 3.0.0
- Dask ≥ 2021.0.0
- Matplotlib ≥ 3.3.0
- Seaborn ≥ 0.11.0
- Pandas ≥ 1.2.0

Optional Requirements
~~~~~~~~~~~~~~~~~~~~~

- **CuPy** ≥ 11.0.0 (for GPU acceleration, requires NVIDIA GPU with CUDA)
- **pytest** ≥ 7.0.0 (for running tests)
- **sphinx** ≥ 4.0.0 (for building documentation)

System Requirements
~~~~~~~~~~~~~~~~~~~

- **CPU-only mode**: Any system with Python 3.8+
- **GPU mode**: NVIDIA GPU with CUDA 11.0+ or CUDA 12.x

Quick Installation
------------------

Using uv (Recommended)
~~~~~~~~~~~~~~~~~~~~~~

`uv <https://github.com/astral-sh/uv>`_ is the fastest Python package installer:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/wizofe/neurogale.git
    cd neurogale

    # Create virtual environment
    uv venv .venv
    source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

    # Install CPU-only version
    uv pip install -e .

    # Or install with GPU support (requires NVIDIA GPU)
    uv pip install -e ".[gpu]"

    # Or install with development tools
    uv pip install -e ".[dev]"

Using pip
~~~~~~~~~

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/wizofe/neurogale.git
    cd neurogale

    # Create virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate

    # Install
    pip install -e .

From PyPI (Coming Soon)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install neurogale

Installation Options
--------------------

CPU-Only Installation
~~~~~~~~~~~~~~~~~~~~~

For systems without NVIDIA GPUs (e.g., macOS, AMD GPUs):

.. code-block:: bash

    pip install -e .

This installs all core dependencies except CuPy. **neurogale** will automatically fall back to CPU computation.

GPU-Enabled Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For systems with NVIDIA GPUs:

.. code-block:: bash

    # Install with CUDA 12.x support
    pip install -e ".[gpu]"

    # Or manually specify CuPy version
    pip install cupy-cuda12x  # For CUDA 12.x
    pip install cupy-cuda11x  # For CUDA 11.x

Check your CUDA version:

.. code-block:: bash

    nvcc --version

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For contributors and developers:

.. code-block:: bash

    pip install -e ".[dev]"

This includes:

- pytest for testing
- pytest-cov for coverage
- sphinx for documentation
- black for code formatting
- flake8 for linting
- mypy for type checking

Verify Installation
-------------------

Check Core Installation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from src.preprocessing.standardise import standardise_cpu
    from src.connectivity.degree_centrality import degree_centrality_cpu

    # Generate test data
    data = np.random.randn(100, 200)
    data_std = standardise_cpu(data)
    print("✓ Core functionality working")

Check GPU Support
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.utils.gpu_utils import check_gpu_availability, get_gpu_info

    is_available, error_msg = check_gpu_availability()

    if is_available:
        gpu_info = get_gpu_info()
        print(f"✓ GPU available: {gpu_info['name']}")
        print(f"  Memory: {gpu_info['total_memory_gb']:.1f} GB")
    else:
        print(f"✗ GPU not available: {error_msg}")
        print("  (Will use CPU mode)")

Command-Line Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Show GPU information
    neurogale --show-gpu-info

    # Run small test
    neurogale --n_voxels 100 --n_timepoints 50 --cpu-only --seed 42

Troubleshooting
---------------

CuPy Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: CuPy installation fails

**Solution**: Ensure CUDA toolkit is installed and matches CuPy version:

.. code-block:: bash

    # Check CUDA version
    nvcc --version

    # Install matching CuPy version
    pip install cupy-cuda12x  # For CUDA 12.x
    pip install cupy-cuda11x  # For CUDA 11.x

**Problem**: "CuPy not compatible with CUDA version"

**Solution**: Install the correct CuPy variant for your CUDA version. See the `CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`_.

macOS Installation
~~~~~~~~~~~~~~~~~~

**Note**: NVIDIA CUDA is not available on macOS. **neurogale** will automatically use CPU mode.

.. code-block:: bash

    # Install without GPU support
    pip install -e .

Docker Installation
-------------------

Pre-built Docker Image
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    docker pull wizofe/neurogale:latest
    docker run --gpus all -it wizofe/neurogale:latest

Build from Source
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/wizofe/neurogale.git
    cd neurogale
    docker build -t neurogale:local .
    docker run --gpus all -it neurogale:local

Conda/Mamba Installation
-------------------------

Using conda-forge:

.. code-block:: bash

    conda create -n neurogale python=3.10
    conda activate neurogale
    conda install numpy nibabel dask matplotlib seaborn pandas pytest

    # For GPU support
    conda install -c conda-forge cupy

    # Install neurogale
    pip install -e .

Platform-Specific Notes
-----------------------

Linux
~~~~~

Recommended platform for GPU acceleration. Ensure NVIDIA drivers and CUDA toolkit are installed:

.. code-block:: bash

    # Check NVIDIA driver
    nvidia-smi

    # Install CUDA toolkit (Ubuntu/Debian)
    sudo apt-get install nvidia-cuda-toolkit

Windows
~~~~~~~

GPU support requires NVIDIA drivers and CUDA toolkit:

1. Install `NVIDIA CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_
2. Install Visual Studio Build Tools
3. Follow quick installation steps above

macOS
~~~~~

GPU acceleration not available (no CUDA support). **neurogale** works in CPU mode:

.. code-block:: bash

    # Install with Homebrew Python
    brew install python@3.10
    python3.10 -m pip install -e .

HPC Clusters
~~~~~~~~~~~~

For SLURM-based clusters:

.. code-block:: bash

    # Request GPU node
    srun --gres=gpu:1 --pty bash

    # Load modules
    module load python/3.10 cuda/12.0

    # Install in user directory
    pip install --user -e .

Next Steps
----------

- Read the :doc:`quickstart` guide
- Explore :doc:`examples`
- Learn about :doc:`extending` with new metrics
- Review the :doc:`api` documentation
