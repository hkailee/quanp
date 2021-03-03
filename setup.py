import sys

if sys.version_info < (3, 6):
    sys.exit('quanp requires Python >= 3.6')

from pathlib import Path

from setuptools import setup, find_packages

try:
    from quanp import __author__, __email__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = 'leehongkai@gmail.com'

setup(
    name='quanp',
    version='0.1.22',
    # use_scm_version=True,
    # setup_requires=['setuptools_scm'],
    description='Quantitative Analysis in Python.',
    long_description=Path('README.rst').read_text('utf-8'),
    url='http://github.com/hongkailee/quanp',
    author=__author__,
    author_email=__email__,
    license='BSD',
    python_requires='>=3.6',
    install_requires=[
        l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()
    ],
    extras_require=dict(
        louvain=['python-igraph', 'louvain>=0.6,!=0.6.2'],
        leiden=['python-igraph', 'leidenalg'],
        bbknn=['bbknn'],
        scvi=['scvi>=0.6.5'],
        rapids=['cudf>=0.9', 'cuml>=0.9', 'cugraph>=0.9'],
        magic=['magic-impute>=2.0'],
        skmisc=["scikit-misc>=0.1.3"],
        harmony=['harmonypy'],
        doc=[
            'sphinx',
            'sphinx_rtd_theme',
            'sphinx_autodoc_typehints',
            'scanpydoc>=0.5',
            'typing_extensions; python_version < "3.7"',  # for `Literal`
        ],
        test=[
            'pytest>=4.4',
            'dask[array]!=2.17.0',
            'fsspec',
            'zappy',
            'zarr',
            'black',
            'profimp',
        ],
    ),
    packages=find_packages(),
    include_package_data=True,
    entry_points=dict(console_scripts=['quanp=quanp.cli:console_main']),
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Framework :: Jupyter',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
)
