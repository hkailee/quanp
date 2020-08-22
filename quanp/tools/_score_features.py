"""Calculate scores based on the expression of feature lists.
"""
from typing import Sequence, Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse

from .. import logging as logg
from .._utils import AnyRandom


def _sparse_nanmean(X, axis):
    """
    np.nanmean equivalent for sparse matrices
    """
    if not issparse(X):
        raise TypeError("X must be a sparse matrix")

    # count the number of nan elements per row/column (dep. on axis)
    Z = X.copy()
    Z.data = np.isnan(Z.data)
    Z.eliminate_zeros()
    n_elements = Z.shape[axis] - Z.sum(axis)

    # set the nans to 0, so that a normal .sum() works
    Y = X.copy()
    Y.data[np.isnan(Y.data)] = 0
    Y.eliminate_zeros()

    # the average
    s = Y.sum(axis)
    m = s / n_elements.astype('float32') # if we dont cast the int32 to float32, this will result in float64...

    return m


def score_features(
    adata: AnnData,
    feature_list: Sequence[str],
    ctrl_size: int = 50,
    feature_pool: Optional[Sequence[str]] = None,
    n_bins: int = 25,
    score_name: str = 'score',
    random_state: AnyRandom = 0,
    copy: bool = False,
    use_raw: bool = None,
) -> Optional[AnnData]:
    """\
    Score a set of features [Satija15]_.

    The score is the average expression of a set of features subtracted with the
    average expression of a reference set of features. The reference set is
    randomly sampled from the `feature_pool` for each binned expression value.

    This reproduces the approach in Seurat [Satija15]_ and has been implemented
    for Quanpy.

    Parameters
    ----------
    adata
        The annotated data matrix.
    feature_list
        The list of feature names used for score calculation.
    ctrl_size
        Number of reference features to be sampled from each bin. If `len(feature_list)` is not too
        low, you can set `ctrl_size=len(feature_list)`.
    feature_pool
        features for sampling the reference set. Default is all features.
    n_bins
        Number of expression level bins for sampling.
    score_name
        Name of the field to be added in `.obs`.
    random_state
        The random seed for sampling.
    copy
        Copy `adata` or modify it inplace.
    use_raw
        Use `raw` attribute of `adata` if present.

        .. versionchanged:: 1.4.5
           Default value changed from `False` to `None`.

    Returns
    -------
    Depending on `copy`, returns or updates `adata` with an additional field
    `score_name`.

    Examples
    --------
    See this `notebook <https://github.com/theislab/scanpy_usage/tree/master/180209_cell_cycle>`__.
    """
    start = logg.info(f'computing score {score_name!r}')
    adata = adata.copy() if copy else adata

    if random_state is not None:
        np.random.seed(random_state)

    feature_list_in_var = []
    var_names = adata.raw.var_names if use_raw else adata.var_names
    features_to_ignore = []
    for feature in feature_list:
        if feature in var_names:
            feature_list_in_var.append(feature)
        else:
            features_to_ignore.append(feature)
    if len(features_to_ignore) > 0:
        logg.warning(f'features are not in var_names and ignored: {features_to_ignore}')
    feature_list = set(feature_list_in_var[:])

    if len(feature_list) == 0:
        logg.warning('provided feature list has length 0, scores as 0')
        adata.obs[score_name] = 0
        return adata if copy else None

    if feature_pool is None:
        feature_pool = list(var_names)
    else:
        feature_pool = [x for x in feature_pool if x in var_names]

    # Trying here to match the Seurat approach in scoring subjects.
    # Basically we need to compare features against random features in a matched
    # interval of expression.

    if use_raw is None:
        use_raw = True if adata.raw is not None else False
    _adata = adata.raw if use_raw else adata

    _adata_subset = _adata[:, feature_pool] if len(feature_pool) < len(_adata.var_names) else _adata
    if issparse(_adata_subset.X):
        obs_avg = pd.Series(
            np.array(_sparse_nanmean(_adata_subset.X, axis=0)).flatten(), index=feature_pool)  # average expression of features
    else:
        obs_avg = pd.Series(
            np.nanmean(_adata_subset.X, axis=0), index=feature_pool)  # average expression of features

    obs_avg = obs_avg[np.isfinite(obs_avg)] # Sometimes (and I don't know how) missing data may be there, with nansfor

    n_items = int(np.round(len(obs_avg) / (n_bins - 1)))
    obs_cut = obs_avg.rank(method='min') // n_items
    control_features = set()

    # now pick `ctrl_size` features from every cut
    for cut in np.unique(obs_cut.loc[feature_list]):
        r_features = np.array(obs_cut[obs_cut == cut].index)
        np.random.shuffle(r_features)
        # uses full r_features if ctrl_size > len(r_features)
        control_features.update(set(r_features[:ctrl_size]))

    # To index, we need a list – indexing implies an order.
    control_features = list(control_features - feature_list)
    feature_list = list(feature_list)

    X_list = _adata[:, feature_list].X
    if issparse(X_list):
        X_list = np.array(_sparse_nanmean(X_list, axis=1)).flatten()
    else:
        X_list = np.nanmean(X_list, axis=1)

    X_control = _adata[:, control_features].X
    if issparse(X_control):
        X_control = np.array(_sparse_nanmean(X_control, axis=1)).flatten()
    else:
        X_control = np.nanmean(X_control, axis=1)

    if len(feature_list) == 0:
        # We shouldn't even get here, but just in case
        logg.hint(
            f'could not add \n'
            f'    {score_name!r}, score of feature set (adata.obs)'
        )
        return adata if copy else None
    elif len(feature_list) == 1:
        if _adata[:, feature_list].X.ndim == 2:
            vector = _adata[:, feature_list].X.toarray()[:, 0] # new anndata
        else:
            vector = _adata[:, feature_list].X  # old anndata
        score = vector - X_control
    else:
        score = X_list - X_control

    adata.obs[score_name] = pd.Series(np.array(score).ravel(), index=adata.obs_names)

    logg.info(
        '    finished',
        time=start,
        deep=(
            'added\n'
            f'    {score_name!r}, score of feature set (adata.obs).\n'
            f'    {len(control_features)} total control features are used.'
        ),
    )
    return adata if copy else None


def score_features_subject_cycle(
    adata: AnnData,
    s_features: Sequence[str],
    g2m_features: Sequence[str],
    copy: bool = False,
    **kwargs,
) -> Optional[AnnData]:
    """\
    Score subject cycle features [Satija15]_.

    Given two lists of features associated to S phase and G2M phase, calculates
    scores and assigns a subject cycle phase (G1, S or G2M). See
    :func:`~quanp.tl.score_features` for more explanation.

    Parameters
    ----------
    adata
        The annotated data matrix.
    s_features
        List of features associated with S phase.
    g2m_features
        List of features associated with G2M phase.
    copy
        Copy `adata` or modify it inplace.
    **kwargs
        Are passed to :func:`~quanp.tl.score_features`. `ctrl_size` is not
        possible, as it's set as `min(len(s_features), len(g2m_features))`.

    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.

    **S_score** : `adata.obs`, dtype `object`
        The score for S phase for each subject.
    **G2M_score** : `adata.obs`, dtype `object`
        The score for G2M phase for each subject.
    **phase** : `adata.obs`, dtype `object`
        The subject cycle phase (`S`, `G2M` or `G1`) for each subject.

    See also
    --------
    score_features

    Examples
    --------
    See this `notebook <https://github.com/theislab/scanpy_usage/tree/master/180209_cell_cycle>`__.
    """
    logg.info('calculating subject cycle phase')

    adata = adata.copy() if copy else adata
    ctrl_size = min(len(s_features), len(g2m_features))
    # add s-score
    score_features(adata, feature_list=s_features, score_name='S_score', ctrl_size=ctrl_size, **kwargs)
    # add g2m-score
    score_features(adata, feature_list=g2m_features, score_name='G2M_score', ctrl_size=ctrl_size, **kwargs)
    scores = adata.obs[['S_score', 'G2M_score']]

    # default phase is S
    phase = pd.Series('S', index=scores.index)

    # if G2M is higher than S, it's G2M
    phase[scores.G2M_score > scores.S_score] = 'G2M'

    # if all scores are negative, it's G1...
    phase[np.all(scores < 0, axis=1)] = 'G1'

    adata.obs['phase'] = phase
    logg.hint('    \'phase\', subject cycle phase (adata.obs)')
    return adata if copy else None
