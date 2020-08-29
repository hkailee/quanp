# some technical stuff
import sys
from ._utils import pkg_version, check_versions, annotate_doc_types

__author__ = ', '.join([
    'Hong Kai Lee',
])
__email__ = ', '.join([
    'leehongkai@gmail.com',
])
try:
    from setuptools_scm import get_version
    __version__ = get_version(root='..', relative_to=__file__)
    del get_version
except (LookupError, ImportError):
    __version__ = str(pkg_version(__name__))

check_versions()
del pkg_version, check_versions

# the actual API
from ._settings import settings, Verbosity  # start with settings as several tools are using it
from . import tools as tl
from . import preprocessing as pp
from . import plotting as pl
from . import datasets, logging, get #, external, queries

from anndata import AnnData
from anndata import read_h5ad, read_csv, read_excel, read_hdf, read_loom, read_mtx, read_text
from .readwrite import read, write
from .neighbors import Neighbors

set_figure_params = settings.set_figure_params

# has to be done at the end, after everything has been imported
annotate_doc_types(sys.modules[__name__], 'quanp')
del sys, annotate_doc_types
