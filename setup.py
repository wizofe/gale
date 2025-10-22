from setuptools import setup, find_packages

setup(
    name='DRAGON',
    version='0.1.0',
    author='Ioannis Valasakis',
    author_email='code@wizofe.uk',
    description='Optimising voxelwise brain metric calculations using GPU-accelerated computing',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'cupy-cuda12x',
        'dask',
        'nibabel',
        'matplotlib',
        'seaborn',
        'pandas'
    ],
    entry_points={
        'console_scripts': [
            'dragon-run=experiments.run_experiments:main'
        ]
    }
)
