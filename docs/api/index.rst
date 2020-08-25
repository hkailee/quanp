.. module:: quanp
.. automodule:: quanp
   :noindex:

API
===


Import Quanp as::

   import quanp as qp

.. note::
   Wrappers to external functionality are found in :mod:`quanp.external`.

Preprocessing: `pp`
-------------------

.. module:: quanp.pp
.. currentmodule:: quanp

Filtering of highly-variable features, per-subject normalization.

Any transformation of the data matrix that is not a *tool*. Other than *tools*, preprocessing steps usually don't return an easily interpretable annotation, but perform a basic transformation on the data matrix.

Basic Preprocessing
~~~~~~~~~~~~~~~~~~~

For visual quality control, see :func:`~quanp.pl.highest_expr_features` and
:func:`~quanp.pl.filter_features_dispersion` in :mod:`quanp.plotting`.

.. autosummary::
   :toctree: .

   pp.calculate_qc_metrics
   pp.filter_subjects
   pp.filter_features
   pp.highly_variable_features
   pp.log1p
   pp.pca
   pp.normalize_total
   pp.regress_out
   pp.scale
   pp.subsample
   pp.downsample_counts

Recipes
~~~~~~~

.. autosummary::
   :toctree: .

   pp.recipe_zheng17
   pp.recipe_weinreb17
   pp.recipe_seurat

Batch effect correction
~~~~~~~~~~~~~~~~~~~~~~~

Also see `Data integration`_. Note that a simple batch correction method is available via :func:`pp.regress_out`. Checkout :mod:`quanp.external` for more.

.. autosummary::
   :toctree: .

   pp.combat

Neighbors
~~~~~~~~~

.. autosummary::
   :toctree: .

   pp.neighbors


Tools: `tl`
-----------

.. module:: quanp.tl
.. currentmodule:: quanp

Any transformation of the data matrix that is not *preprocessing*. In contrast to a *preprocessing* function, a *tool* usually adds an easily interpretable annotation to the data matrix, which can then be visualized with a corresponding plotting function.

Embeddings
~~~~~~~~~~

.. autosummary::
   :toctree: .

   tl.pca
   tl.tsne
   tl.umap
   tl.draw_graph
   tl.diffmap

Compute densities on embeddings.

.. autosummary::
   :toctree: .

   tl.embedding_density

Clustering and trajectory inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   tl.leiden
   tl.louvain
   tl.dendrogram
   tl.dpt
   tl.paga

Data integration
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   tl.ingest

Marker features
~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   tl.rank_features_groups
   tl.filter_rank_features_groups
   tl.marker_feature_overlap

Feature scores, Subject cycle
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   tl.score_features
   tl.score_features_subject_cycle

Simulations
~~~~~~~~~~~

.. autosummary::
   :toctree: .

   tl.sim


Plotting: `pl`
--------------

.. module:: quanp.pl
.. currentmodule:: quanp

The plotting module :mod:`quanp.plotting` largely parallels the ``tl.*`` and a few of the ``pp.*`` functions.
For most tools and for some preprocessing functions, you'll find a plotting function with the same name.

.. autosummary::
   :toctree: .

   plotting


Reading
-------

.. note::
   For reading annotation use :ref:`pandas.read_… <pandas:io>`
   and add it to your :class:`anndata.AnnData` object. The following read functions are
   intended for the numeric data in the data matrix `X`.

Read common file formats using

.. autosummary::
   :toctree: .

   read

Read 10x formatted hdf5 files and directories containing `.mtx` files using

.. autosummary::
   :toctree: .

   read_10x_h5
   read_10x_mtx
   read_visium

Read other formats using functions borrowed from :mod:`anndata`

.. autosummary::
   :toctree: .

   read_h5ad
   read_csv
   read_excel
   read_hdf
   read_loom
   read_mtx
   read_text
   read_umi_tools


Get object from `AnnData`: `get`
--------------------------------

.. module:: quanp.get
.. currentmodule:: quanp

The module `sc.get` provides convenience functions for getting values back in
useful formats.

.. autosummary::
   :toctree:

   get.obs_df
   get.var_df
   get.rank_features_groups_df


Queries
-------

.. module:: quanp.queries
.. currentmodule:: quanp

This module provides useful queries for annotation and enrichment.

.. autosummary::
   :toctree: .

   queries.biomart_annotations
   queries.feature_coordinates
   queries.mitochondrial_features
   queries.enrich


Classes
-------

:class:`~anndata.AnnData` is reexported from :mod:`anndata`.

Represent data as a neighborhood structure, usually a knn graph.

.. autosummary::
   :toctree: .

   Neighbors


.. _settings:

Settings
--------

A convenience function for setting some default :obj:`matplotlib.rcParams` and a
high-resolution jupyter display backend useful for use in notebooks.

.. autosummary::
   :toctree: .

   set_figure_params

An instance of the :class:`~quanp._settings.QuanpConfig` is available as `quanp.settings` and allows configuring Quanp.

.. autosummary::
   :toctree: .

   _settings.QuanpConfig

Some selected settings are discussed in the following.

Influence the global behavior of plotting functions. In non-interactive scripts,
you'd usually want to set `settings.autoshow` to ``False``.

.. no :toctree: here because they are linked under the class
.. autosummary::

   ~_settings.QuanpConfig.autoshow
   ~_settings.QuanpConfig.autosave

The default directories for saving figures, caching files and storing datasets.

.. autosummary::

   ~_settings.QuanpConfig.figdir
   ~_settings.QuanpConfig.cachedir
   ~_settings.QuanpConfig.datasetdir

The verbosity of logging output, where verbosity levels have the following
meaning: 0='error', 1='warning', 2='info', 3='hint', 4=more details, 5=even more
details, etc.

.. autosummary::

   ~_settings.QuanpConfig.verbosity

Print versions of packages that might influence numerical results.

.. autosummary::
   :toctree: .

   logging.print_versions


Datasets
--------

.. module:: quanp.datasets
.. currentmodule:: quanp

.. autosummary::
   :toctree: .

   datasets.blobs
   datasets.ebi_expression_atlas
   datasets.krumsiek11
   datasets.moignard15
   datasets.pbmc3k
   datasets.pbmc3k_processed
   datasets.pbmc68k_reduced
   datasets.paul15
   datasets.toggleswitch
   datasets.visium_sge


Further modules
---------------

.. autosummary::
   :toctree: .

   plotting


Deprecated functions
--------------------

.. autosummary::
   :toctree: .

   pp.filter_features_dispersion
   pp.normalize_per_subject
