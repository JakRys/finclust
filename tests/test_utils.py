import pytest
import pandas as pd
from finclust.utils import calculate_affinities, compose_affinities

from sklearn.metrics.pairwise import cosine_similarity


df0 = pd.DataFrame(
    columns = ["a", "b", "c"],
    data = [
        [1, 1, 1], 
        [2, 0, 2],
        ],
)

multiindex = pd.MultiIndex.from_product([["first", "second"], ["a", "b", "c"]])
df1 = pd.DataFrame(columns = multiindex)
df1["first"] = df0
df1["second"] = df0 *2 +1


def test_calculate_affinities_simple_df():
    sim = calculate_affinities(df0, func=cosine_similarity)
    assert sim.shape == (df0.shape[1], df0.shape[1])
    for c in sim.columns:
        assert pytest.approx(sim.loc[c, c]) == 1


def test_calculate_affinities_multiindex_df():
    sim = calculate_affinities(df1, func=cosine_similarity)
    assert sim.shape == (len(df1.columns.levels[1]), df1.shape[1])

###############################################################################

def test_compose_affinities_simple_df():
    assert compose_affinities(df0).equals(df0)


def test_compose_affinities_multiindex_df():
    correct_output = pd.DataFrame(
        columns = ["a", "b", "c"],
        data = [
            [4, 4, 4], 
            [7, 1, 7],
            ],
    )
    assert compose_affinities(df1, normalize=False).equals(correct_output)


def test_compose_affinities_multiindex_df_with_weights():
    assert all(compose_affinities(df1, weights=[2, 0]) == df0)
    assert all(compose_affinities(df1, weights={"first": 0}) == df0 *2 +1)
