import itertools as it
import types
import typing

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ._draw_cgrid import draw_cgrid


def draw_cscatter(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    cmap: typing.Callable,
    ax: typing.Optional[mpl.axes.Axes] = None,
    border: bool = False,
    despine: bool = True,
    major: typing.Optional[float] = None,
    minor: typing.Optional[float] = None,
    cgrid: bool = True,
    cgrid_kws: dict = types.MappingProxyType({}),
    rect_kws: dict = types.MappingProxyType({}),
    scatter_kws: dict = types.MappingProxyType({}),
    xmax: float = 1.0,
    ymax: float = 1.0,
) -> mpl.axes.Axes:

    if cgrid:
        ax = draw_cgrid(
            ax=ax,
            cmap=cmap,
            xmax=xmax,
            ymax=ymax,
            **{
                "imshow_kws": dict(zorder=-300),
                **cgrid_kws,
            },
        )
    rect = mpl.patches.Rectangle(
        (-0.01, -0.01),
        1.02,
        1.02,
        clip_on=False,
        transform=ax.transAxes,
        **{
            **dict(
                alpha=0.8,
                facecolor="white",
                zorder=-200,
            ),
            **rect_kws,
        },
    )
    ax.add_patch(rect)

    ax = sns.scatterplot(
        data=data,
        x=x,
        y=y,
        ax=ax,
        **{
            **dict(
                alpha=0.3,
                clip_on=False,
                color=[*it.starmap(cmap, zip(data[x] / xmax, data[y] / ymax))],
                edgecolor=None,
                legend=False,
                linewidth=0,
                s=100,
                zorder=-100,
            ),
            **scatter_kws,
        },
    )

    if minor is not None:
        ax.grid(visible=True, which="minor", color="gray", ls="--", lw=0.2)
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minor))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(minor))

    if major is not None:
        ax.grid(visible=True, which="major", color="#222222", lw=0.4)
        ax.xaxis.set_major_locator(plt.MultipleLocator(major))
        ax.yaxis.set_major_locator(plt.MultipleLocator(major))

    if despine:
        ax.tick_params(axis="both", which="both", length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        sns.despine(ax=ax, left=True, bottom=True)
        if major is not None:
            for spine in "left", "bottom":
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_color("#222222")
                ax.spines[spine].set_linewidth(0.4)

    if border:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#222222")
            spine.set_linewidth(0.4)

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)

    return ax
