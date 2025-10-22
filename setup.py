from setuptools import setup, find_packages

setup(
    name='gale',  # Renamed from DRAGON
    version='0.2.0',  # Bumped version for major improvements
    author='Ioannis Valasakis',
    author_email='code@wizofe.uk',
    description='GPU-accelerated voxelwise brain connectivity analysis framework',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/wizofe/gale',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.20.0',
        'dask>=2021.0.0',
        'nibabel>=3.0.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'pandas>=1.2.0',
    ],
    extras_require={
        'gpu': [
            'cupy-cuda12x>=11.0.0',  # Optional GPU support
        ],
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'pytest-xdist>=2.5.0',
            'sphinx>=4.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ],
        'test': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'gale=experiments.run_experiments:main',  # Fixed module path
        ]
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='neuroscience fMRI brain-connectivity graph-theory GPU',
)
