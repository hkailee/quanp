"""\
Calculate overlaps of rank_features_groups marker features with marker feature dictionaries
"""
import collections.abc as cabc
from typing import Union, Optional, Dict

import numpy as np
import pandas as pd
from anndata import AnnData

from .. import logging as logg
from .._compat import Literal


def _calc_overlap_count(markers1: dict, markers2: dict):
    """\
    Calculate overlap count between the values of two dictionaries

    Note: dict values must be sets
    """
    overlaps = np.zeros((len(markers1), len(markers2)))

    for j, marker_group in enumerate(markers1):
        tmp = [
            len(markers2[i].intersection(markers1[marker_group]))
            for i in markers2.keys()
        ]
        overlaps[j, :] = tmp

    return overlaps


def _calc_overlap_coef(markers1: dict, markers2: dict):
    """\
    Calculate overlap coefficient between the values of two dictionaries

    Note: dict values must be sets
    """
    overlap_coef = np.zeros((len(markers1), len(markers2)))

    for j, marker_group in enumerate(markers1):
        tmp = [
            len(markers2[i].intersection(markers1[marker_group]))
            / max(min(len(markers2[i]), len(markers1[marker_group])), 1)
            for i in markers2.keys()
        ]
        overlap_coef[j, :] = tmp

    return overlap_coef


def _calc_jaccard(markers1: dict, markers2: dict):
    """\
    Calculate jaccard index between the values of two dictionaries

    Note: dict values must be sets
    """
    jacc_results = np.zeros((len(markers1), len(markers2)))

    for j, marker_group in enumerate(markers1):
        tmp = [
            len(markers2[i].intersection(markers1[marker_group]))
            / len(markers2[i].union(markers1[marker_group]))
            for i in markers2.keys()
        ]
        jacc_results[j, :] = tmp

    return jacc_results


_Method = Literal['overlap_count', 'overlap_coef', 'jaccard']


def marker_feature_overlap(
    adata: AnnData,
    reference_markers: Union[Dict[str, set], Dict[str, list]],
    *,
    key: str = 'rank_features_groups',
    method: _Method = 'overlap_count',
    normalize: Optional[Literal['reference', 'data']] = None,
    top_n_markers: Optional[int] = None,
    adj_pval_threshold: Optional[float] = None,
    key_added: str = 'marker_feature_overlap',
    inplace: bool = False,
):
    """\
    Calculate an overlap score between data-deriven marker features and
    provided markers

    Marker feature overlap scores can be quoted as overlap counts, overlap
    coefficients, or jaccard indices. The method returns a pandas dataframe
    which can be used to annotate clusters based on marker feature overlaps.

    This function was written by Malte Luecken.

    Parameters
    ----------
    adata
        The annotated data matrix.
    reference_markers
        A marker feature dictionary object. Keys should be strings with the
        subject identity name and values are sets or lists of strings which match
        format of `adata.var_name`.
    key
        The key in `adata.uns` where the rank_features_groups output is stored.
        By default this is `'rank_features_groups'`.
    method
        (default: `overlap_count`)
        Method to calculate marker feature overlap. `'overlap_count'` uses the
        intersection of the feature set, `'overlap_coef'` uses the overlap
        coefficient, and `'jaccard'` uses the Jaccard index.
    normalize
        Normalization option for the marker feature overlap output. This parameter
        can only be set when `method` is set to `'overlap_count'`. `'reference'`
        normalizes the data by the total number of marker features given in the
        reference annotation per group. `'data'` normalizes the data by the
        total number of marker features used for each cluster.
    top_n_markers
        The number of top data-derived marker features to use. By default all
        calculated marker features are used. If `adj_pval_threshold` is set along
        with `top_n_markers`, then `adj_pval_threshold` is ignored.
    adj_pval_threshold
        A significance threshold on the adjusted p-values to select marker
        features. This can only be used when adjusted p-values are calculated by
        `qp.tl.rank_features_groups()`. If `adj_pval_threshold` is set along with
        `top_n_markers`, then `adj_pval_threshold` is ignored.
    key_added
        Name of the `.uns` field that will contain the marker overlap scores.
    inplace
        Return a marker feature dataframe or store it inplace in `adata.uns`.

    Returns
    -------
    A pandas dataframe with the marker feature overlap scores if `inplace=False`.
    For `inplace=True` `adata.uns` is updated with an additional field
    specified by the `key_added` parameter (default = 'marker_feature_overlap').

    Examples
    --------
    >>> import quanp as qp
    >>> adata = qp.datasets.pbmc68k_reduced()
    >>> qp.pp.pca(adata, svd_solver='arpack')
    >>> qp.pp.neighbors(adata)
    >>> qp.tl.louvain(adata)
    >>> qp.tl.rank_features_groups(adata, groupby='louvain')
    >>> marker_features = {
    ...     'SubjectGroupA': {'feature1'},
    ...     'SubjectGroupB': {'feature2', 'feature3'},
    ...     'SubjectGroupC': {'feature4'},
    ...     'SubjectGroupD': {'feature5'},
    ...     'SubjectGroupE': {'feature6', 'feature7'},
    ...     'SubjectGroupF': {'feature8', 'feature9'},
    ...     'SubjectGroupG': {'feature10', 'feature11'},
    ...     'SubjectGroupH': {'feature12'}
    ... }
    >>> marker_matches = qp.tl.marker_feature_overlap(adata, marker_features)
    """
    # Test user inputs
    if inplace:
        raise NotImplementedError(
            'Writing Pandas dataframes to h5ad is currently under development.'
            '\nPlease use `inplace=False`.'
        )

    if key not in adata.uns:
        raise ValueError(
            'Could not find marker feature data. '
            'Please run `qp.tl.rank_features_groups()` first.'
        )

    avail_methods = {'overlap_count', 'overlap_coef', 'jaccard', 'enrich'}
    if method not in avail_methods:
        raise ValueError(f'Method must be one of {avail_methods}.')

    if normalize == 'None':
        normalize = None

    avail_norm = {'reference', 'data', None}
    if normalize not in avail_norm:
        raise ValueError(f'Normalize must be one of {avail_norm}.')

    if normalize is not None and method != 'overlap_count':
        raise ValueError('Can only normalize with method=`overlap_count`.')

    if not all(isinstance(val, cabc.Set) for val in reference_markers.values()):
        try:
            reference_markers = {
                key: set(val) for key, val in reference_markers.items()
            }
        except Exception:
            raise ValueError(
                'Please ensure that `reference_markers` contains '
                'sets or lists of markers as values.'
            )

    if adj_pval_threshold is not None:
        if 'pvals_adj' not in adata.uns[key]:
            raise ValueError(
                'Could not find adjusted p-value data. '
                'Please run `qp.tl.rank_features_groups()` with a '
                'method that outputs adjusted p-values.'
            )

        if adj_pval_threshold < 0:
            logg.warning(
                '`adj_pval_threshold` was set below 0. Threshold will be set to 0.'
            )
            adj_pval_threshold = 0
        elif adj_pval_threshold > 1:
            logg.warning(
                '`adj_pval_threshold` was set above 1. Threshold will be set to 1.'
            )
            adj_pval_threshold = 1

        if top_n_markers is not None:
            logg.warning(
                'Both `adj_pval_threshold` and `top_n_markers` is set. '
                '`adj_pval_threshold` will be ignored.'
            )

    if top_n_markers is not None and top_n_markers < 1:
        logg.warning(
            '`top_n_markers` was set below 1. `top_n_markers` will be set to 1.'
        )
        top_n_markers = 1

    # Get data-derived marker features in a dictionary of sets
    data_markers = dict()
    cluster_ids = adata.uns[key]['names'].dtype.names

    for group in cluster_ids:
        if top_n_markers is not None:
            n_features = min(top_n_markers, adata.uns[key]['names'].shape[0])
            data_markers[group] = set(adata.uns[key]['names'][group][:n_features])
        elif adj_pval_threshold is not None:
            n_features = (adata.uns[key]['pvals_adj'][group] < adj_pval_threshold).sum()
            data_markers[group] = set(adata.uns[key]['names'][group][:n_features])
            if n_features == 0:
                logg.warning(
                    'No marker features passed the significance threshold of '
                    f'{adj_pval_threshold} for cluster {group!r}.'
                )
        else:
            data_markers[group] = set(adata.uns[key]['names'][group])

    # Find overlaps
    if method == 'overlap_count':
        marker_match = _calc_overlap_count(reference_markers, data_markers)
        if normalize == 'reference':
            # Ensure rows sum to 1
            ref_lengths = np.array(
                [len(reference_markers[m_group]) for m_group in reference_markers]
            )
            marker_match = marker_match / ref_lengths[:, np.newaxis]
            marker_match = np.nan_to_num(marker_match)
        elif normalize == 'data':
            # Ensure columns sum to 1
            data_lengths = np.array(
                [len(data_markers[dat_group]) for dat_group in data_markers]
            )
            marker_match = marker_match / data_lengths
            marker_match = np.nan_to_num(marker_match)
    elif method == 'overlap_coef':
        marker_match = _calc_overlap_coef(reference_markers, data_markers)
    elif method == 'jaccard':
        marker_match = _calc_jaccard(reference_markers, data_markers)

    # Note:
    # Could add an 'enrich' option here
    # (fisher's exact test or hypergeometric test),
    # but that would require knowledge of the size of the space from which
    # the reference marker feature set was taken.
    # This is at best approximately known.

    # Create a pandas dataframe with the results
    marker_groups = list(reference_markers.keys())
    clusters = list(cluster_ids)
    marker_matching_df = pd.DataFrame(
        marker_match, index=marker_groups, columns=clusters
    )

    # Store the results
    if inplace:
        adata.uns[key_added] = marker_matching_df
        logg.hint(f'added\n    {key_added!r}, marker overlap scores (adata.uns)')
    else:
        return marker_matching_df
