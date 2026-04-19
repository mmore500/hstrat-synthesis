import typing

import alifedata_phyloinformatics_convert as apc
import iplotx as ipx
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
from phyloframe import legacy as pfl


def draw_tree(
    phylogeny_df: pd.DataFrame,
    *,
    ax: typing.Optional[mpl.axes.Axes] = None,
    collapse_unifurcations: bool = True,
    ladderize: bool = True,
    mutate: bool = False,
    **kwargs: dict,
) -> mpl.axes.Axes:

    if ax is None:
        ax = plt.gca()

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    if collapse_unifurcations:
        phylogeny_df = pfl.alifestd_collapse_unifurcations(
            phylogeny_df,
            mutate=True,
            drop_topological_sensitivity=True,
        )

    phylogeny_df = pfl.alifestd_try_add_ancestor_list_col(
        phylogeny_df, mutate=True
    )

    dp_tree = apc.RosettaTree(phylogeny_df).as_dendropy
    if ladderize:
        dp_tree.ladderize()

    ipx.plotting.tree(
        dp_tree,
        ax=ax,
        **{
            **dict(
                edge_linewidth=1.5,
                margins=0.0,
            ),
            **kwargs,
        },
    )

    return ax
