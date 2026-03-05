import itertools as it
import types
import typing

import matplotlib as mpl
import pandas as pd
from phyloframe import legacy as pfl

from ..tree._draw_scatter_tree import draw_scatter_tree


def draw_ctree(
    phylogeny_df: pd.DataFrame,
    *,
    x: str,
    y: str,
    size: typing.Optional[str] = None,
    style: typing.Optional[str] = None,
    cmap: typing.Callable,
    ax: typing.Optional[mpl.axes.Axes] = None,
    collapse_unifurcations: bool = True,
    xmax: float = 1.0,
    ymax: float = 1.0,
    scatter_kws: dict = types.MappingProxyType({}),
    tree_kws: dict = types.MappingProxyType({}),
    **kwargs,
) -> mpl.axes.Axes:

    phylogeny_df = phylogeny_df.copy()

    if collapse_unifurcations:
        phylogeny_df = pfl.alifestd_collapse_unifurcations(
            phylogeny_df, mutate=True
        )

    def safe_cmap(
        x: float, y: float
    ) -> typing.Tuple[float, float, float, float]:
        if pd.isna(x) or pd.isna(y):
            return (0.0, 0.0, 0.0, 0.0)
        try:
            return (*cmap(x, y), 1.0)
        except Exception:
            return (0.0, 0.0, 0.0, 0.0)

    return draw_scatter_tree(
        phylogeny_df,
        ax=ax,
        size=size,
        style=style,
        c=[
            *it.starmap(
                safe_cmap, zip(phylogeny_df[x] / xmax, phylogeny_df[y] / ymax)
            )
        ],
        collapse_unifurcations=False,
        scatter_kws={
            "edgecolor": "none",
            **scatter_kws,
        },
        tree_kws=tree_kws,
        **kwargs,
    )
