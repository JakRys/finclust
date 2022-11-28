"""
Utils

This file provides an implementation of helping functions.
"""

from typing import Callable, Dict, List, Union

import pandas as pd


def calculate_affinities(data: pd.DataFrame, func: Callable, fillna_value: float = 0) -> pd.DataFrame:
    if data.columns.nlevels > 1:
        affinities = pd.DataFrame(columns=data.columns)
        for col in data.columns.levels[0]:
            affinities[col] = pd.DataFrame(func(data[col].T),
                                             columns=data.columns.levels[1],
                                             index=data.columns.levels[1]
                                             ).fillna(fillna_value)
        return affinities
    return pd.DataFrame(func(data.T), columns=data.columns, index=data.columns).fillna(fillna_value)


def compose_affinities(affinities: pd.DataFrame, weights: Union[List[float], Dict[str, float]] = None,
                         normalize: bool = True) -> pd.DataFrame:
    """
    Calculate weighted sum of MultiIndex DataFrame.

    Parameters
    ----------
    TODO:

    Returns
    -------
    composed: pd.DataFrame
        Non-MultiIndex DataFrame with weighted summed values.
    """
    if affinities.columns.nlevels == 1 or len(affinities.columns.levels[0]) == 1:
        return affinities
    columns = affinities.columns.levels[0]
    if isinstance(weights, List):
        weights = {c: v for c, v in zip(columns, weights)}
    # composed = affinities.multiply(weights, axis="columns")
    composed = affinities.copy()
    if weights is not None:
        for c, v in weights.items():
            composed[c] *= v
    composed = composed.groupby(level=1, axis="columns").sum()
    if normalize and weights is not None:
        return composed / sum(weights.values())
    return composed
