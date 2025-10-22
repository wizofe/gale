# DRAGON Framework

DRAGON (Distributed Rapids Analysis of Graph Organization Networks) is a Python framework to optimise voxelwise brain (graph theory) metrics using GPU-accelerated computing. The project is structured for modularity, extensibility and reproducibility.

## Features

- GPU-accelerated correlation computation using CuPy.
- Distributed processing with Dask for large datasets.
- Modular design for data handling, pre-processing, connectivity measures and experiments.
- Reproducible environments with Docker and CI/CD via GitHub Actions.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/dragon.git
    cd dragon
    ```

2. Create a virtual environment and install dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Running the Experiments

Execute the experiments script to run a sample analysis:
```bash
python src/experiments/run_experiments.py --data_path <path_to_nifti_data>

