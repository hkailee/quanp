import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import matplotlib  # noqa

# Don’t use tkinter agg when importing quanp → … → matplotlib
matplotlib.use('agg')

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent), str(HERE / 'extensions')]
import quanp  # noqa

on_rtd = os.environ.get('READTHEDOCS') == 'True'

# -- General configuration ------------------------------------------------


nitpicky = True  # Warn about broken links. This is here for a reason: Do not change.
needs_sphinx = '2.0'  # Nicer param docs
suppress_warnings = ['ref.citation']

# General information
project = 'Quanp'
author = quanp.__author__
copyright = f'{datetime.now():%Y}, {author}.'
version = quanp.__version__.replace('.dirty', '')
release = version

# default settings
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
default_role = 'literal'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    # 'plot_generator',
    # 'plot_directive',
    'sphinx_autodoc_typehints',  # needs to be after napoleon
    # 'ipython_directive',
    # 'ipython_console_highlighting',
    'quanpdoc',
    *[p.stem for p in (HERE / 'extensions').glob('*.py')],
]

# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = 'bysource'
# autodoc_default_flags = ['members']
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [('Params', 'Parameters')]
todo_include_todos = False
api_dir = HERE / 'api'  # function_images

quanp_tutorials_url = 'https://quanp-tutorials.readthedocs.io/en/latest/'

intersphinx_mapping = dict(
    anndata=('https://anndata.readthedocs.io/en/stable/', None),
    bbknn=('https://bbknn.readthedocs.io/en/latest/', None),
    cycler=('https://matplotlib.org/cycler/', None),
    h5py=('http://docs.h5py.org/en/stable/', None),
    ipython=('https://ipython.readthedocs.io/en/stable/', None),
    leidenalg=('https://leidenalg.readthedocs.io/en/latest/', None),
    louvain=('https://louvain-igraph.readthedocs.io/en/latest/', None),
    matplotlib=('https://matplotlib.org/', None),
    networkx=('https://networkx.github.io/documentation/networkx-1.10/', None),
    numpy=('https://docs.scipy.org/doc/numpy/', None),
    pandas=('https://pandas.pydata.org/pandas-docs/stable/', None),
    python=('https://docs.python.org/3', None),
    scipy=('https://docs.scipy.org/doc/scipy/reference/', None),
    scvelo=('https://scvelo.readthedocs.io/', None),
    seaborn=('https://seaborn.pydata.org/', None),
    sklearn=('https://scikit-learn.org/stable/', None),
    quanp_tutorials=(quanp_tutorials_url, None),
)


# -- Options for HTML output ----------------------------------------------


html_theme = 'quanpdoc'
html_theme_options = dict(navigation_depth=4, logo_only=True)  # Only show the logo
html_context = dict(
    display_github=True,  # Integrate GitHub
    github_user='hkailee',  # Username
    github_repo='quanp',  # Repo name
    github_version='master',  # Version
    conf_py_path='/docs/',  # Path in the checkout to the docs root
)
html_static_path = ['_static']
html_show_sphinx = False
# html_logo = '_static/img/quanp_Logo_BrightFG.svg'


def setup(app):
    app.warningiserror = on_rtd


# -- Options for other output formats ------------------------------------------

htmlhelp_basename = f'{project}doc'
doc_title = f'{project} Documentation'
latex_documents = [(master_doc, f'{project}.tex', doc_title, author, 'manual')]
man_pages = [(master_doc, project, doc_title, [author], 1)]
texinfo_documents = [
    (
        master_doc,
        project,
        doc_title,
        author,
        project,
        'One line description of project.',
        'Miscellaneous',
    )
]


# -- Suppress link warnings ----------------------------------------------------

qualname_overrides = {
    "sklearn.neighbors.dist_metrics.DistanceMetric": "sklearn.neighbors.DistanceMetric",
    # If the docs are built with an old version of numpy, this will make it work:
    "numpy.random.RandomState": "numpy.random.mtrand.RandomState",
    "quanp.plotting._matrixplot.MatrixPlot": "quanp.pl.MatrixPlot",
    "quanp.plotting._dotplot.DotPlot": "quanp.pl.DotPlot",
    "quanp.plotting._stacked_violin.StackedViolin": "quanp.pl.StackedViolin",
}

nitpick_ignore = [
    # Will probably be documented
    ('py:class', 'quanp._settings.Verbosity'),
    # Currently undocumented: https://github.com/mwaskom/seaborn/issues/1810
    ('py:class', 'seaborn.ClusterGrid'),
    # Won’t be documented
    ('py:class', 'quanp.plotting._utils._AxesSubplot'),
    ('py:class', 'quanp._utils.Empty'),
    ('py:class', 'numpy.random.mtrand.RandomState'),
]
