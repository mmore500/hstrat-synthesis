import numbers
import types
import typing

import alifedata_phyloinformatics_convert as apc
import iplotx as ipx
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from phyloframe import legacy as pfl
import seaborn as sns


def draw_scatter_tree(
    phylogeny_df: pd.DataFrame,
    *,
    hue: typing.Optional[str] = None,
    size: typing.Optional[str] = None,
    style: typing.Optional[str] = None,
    c: typing.Union[str, typing.Sequence, None] = None,
    ax: typing.Optional[mpl.axes.Axes] = None,
    collapse_unifurcations: bool = True,
    ladderize: bool = True,
    layout: str = "vertical",
    mutate: bool = False,
    scatter_kws: dict = types.MappingProxyType({}),
    scatter_shuffle: typing.Union[bool, int] = False,
    tree_kws: dict = types.MappingProxyType({}),
) -> mpl.axes.Axes:

    if ax is None:
        ax = plt.gca()

    if not mutate:
        phylogeny_df = phylogeny_df.copy()
    elif "label" in phylogeny_df.columns:
        raise ValueError

    if collapse_unifurcations:
        phylogeny_df = pfl.alifestd_collapse_unifurcations(
            phylogeny_df,
            mutate=True,
            drop_topological_sensitivity=True,
        )

    phylogeny_df = pfl.alifestd_try_add_ancestor_list_col(
        phylogeny_df, mutate=True
    )

    phylogeny_df["label"] = np.arange(len(phylogeny_df))

    dp_tree = apc.RosettaTree(phylogeny_df).as_dendropy
    if ladderize:
        dp_tree.ladderize()

    ipx_tree = ipx.plotting.tree(
        dp_tree,
        ax=ax,
        layout=layout,
        **{
            **dict(
                edge_linewidth=1.5,
                margins=0.0,
            ),
            **tree_kws,
        },
    )
    ipx_layout = ipx_tree.get_layout()

    taxa = [node.label for node in ipx_layout.index]
    if layout == "radial":
        xs, ys = ipx_tree.get_nodes().get_offsets().T
    else:
        xs, ys = ipx_layout.T.to_numpy()

    phylogeny_df = phylogeny_df.reset_index(drop=True)

    if "__x__" in phylogeny_df.columns or "__y__" in phylogeny_df.columns:
        raise ValueError

    phylogeny_df["__x__"] = np.nan
    phylogeny_df.loc[taxa, "__x__"] = xs

    phylogeny_df["__y__"] = np.nan
    phylogeny_df.loc[taxa, "__y__"] = ys

    if isinstance(scatter_shuffle, numbers.Integral) or scatter_shuffle:
        random_state = (
            int(scatter_shuffle)
            if isinstance(scatter_shuffle, numbers.Integral)
            and not isinstance(scatter_shuffle, bool)
            else None
        )
        shuf = np.random.RandomState(random_state).permutation(
            len(phylogeny_df),
        )
    else:
        shuf = np.arange(len(phylogeny_df))

    phylogeny_df = phylogeny_df.iloc[shuf]

    if isinstance(c, str):
        c = phylogeny_df[c].fillna("none").tolist()
    elif c is not None:
        c = np.array(c)[shuf]
    elif c is None:
        c = "none"
    else:
        raise ValueError

    sns.scatterplot(
        phylogeny_df,
        x="__x__",
        y="__y__",
        ax=ax,
        hue=hue,
        size=size,
        style=style,
        c=c,
        **{
            "legend": False,
            **scatter_kws,
        },
    )

    return ax
