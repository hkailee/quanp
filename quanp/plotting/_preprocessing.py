from typing import Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as pl
from matplotlib import rcParams
from anndata import AnnData
from . import _utils

# --------------------------------------------------------------------------------
# Plot result of preprocessing functions
# --------------------------------------------------------------------------------


def highly_variable_features(
    adata_or_result: Union[AnnData, pd.DataFrame, np.recarray],
    log: bool = False,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    highly_variable_features: bool = True,
):
    """Plot dispersions or normalized variance versus means for features.

    Produces Supp. Fig. 5c of Zheng et al. (2017) and MeanVarPlot() and
    VariableFeaturePlot() of Seurat.

    Parameters
    ----------
    adata
        Result of :func:`~quanp.pp.highly_variable_features`.
    log
        Plot on logarithmic axes.
    show
         Show the plot, do not return axis.
    save
        If `True` or a `str`, save the figure.
        A string is appended to the default filename.
        Infer the filetype if ending on {{`'.pdf'`, `'.png'`, `'.svg'`}}.
    """
    if isinstance(adata_or_result, AnnData):
        result = adata_or_result.var
        seurat_v3_flavor = adata_or_result.uns["hvg"]["flavor"] == "seurat_v3"
    else:
        result = adata_or_result
        if isinstance(result, pd.DataFrame):
            seurat_v3_flavor = "variances_norm" in result.columns
        else:
            seurat_v3_flavor = False
    if highly_variable_features:
        feature_subset = result.highly_variable
    else:
        feature_subset = result.feature_subset
    means = result.means

    if seurat_v3_flavor:
        var_or_disp = result.variances
        var_or_disp_norm = result.variances_norm
    else:
        var_or_disp = result.dispersions
        var_or_disp_norm = result.dispersions_norm
    size = rcParams['figure.figsize']
    pl.figure(figsize=(2 * size[0], size[1]))
    pl.subplots_adjust(wspace=0.3)
    for idx, d in enumerate([var_or_disp_norm, var_or_disp]):
        pl.subplot(1, 2, idx + 1)
        for label, color, mask in zip(
            ['highly variable features', 'other features'],
            ['black', 'grey'],
            [feature_subset, ~feature_subset],
        ):
            if False:
                means_, var_or_disps_ = np.log10(means[mask]), np.log10(d[mask])
            else:
                means_, var_or_disps_ = means[mask], d[mask]
            pl.scatter(means_, var_or_disps_, label=label, c=color, s=1)
        if log:  # there's a bug in autoscale
            pl.xscale('log')
            pl.yscale('log')
            y_min = np.min(var_or_disp)
            y_min = 0.95 * y_min if y_min > 0 else 1e-1
            pl.xlim(0.95 * np.min(means), 1.05 * np.max(means))
            pl.ylim(y_min, 1.05 * np.max(var_or_disp))
        if idx == 0:
            pl.legend()
        pl.xlabel(('$log_{10}$ ' if False else '') + 'mean expressions of features')
        data_type = 'dispersions' if not seurat_v3_flavor else 'variances'
        pl.ylabel(
            ('$log_{10}$ ' if False else '')
            + '{} of features'.format(data_type)
            + (' (normalized)' if idx == 0 else ' (not normalized)')
        )

    _utils.savefig_or_show('filter_features_dispersion', show=show, save=save)
    if show is False:
        return pl.gca()


# backwards compat
def filter_features_dispersion(
    result: np.recarray,
    log: bool = False,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
):
    """\
    Plot dispersions versus means for features.

    Produces Supp. Fig. 5c of Zheng et al. (2017) and MeanVarPlot() of Seurat.

    Parameters
    ----------
    result
        Result of :func:`~quanp.pp.filter_features_dispersion`.
    log
        Plot on logarithmic axes.
    show
         Show the plot, do not return axis.
    save
        If `True` or a `str`, save the figure.
        A string is appended to the default filename.
        Infer the filetype if ending on {{`'.pdf'`, `'.png'`, `'.svg'`}}.
    """
    highly_variable_features(
        result, log=log, show=show, save=save, highly_variable_features=False
    )
