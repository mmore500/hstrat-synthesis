import itertools as it
import types
import typing

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


def draw_cbar(
    cmap: typing.Callable,
    *,
    ax: typing.Optional[mpl.axes.Axes] = None,
    orient: typing.Literal["horizontal", "vertical"] = "horizontal",
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    imshow_kws: dict = types.MappingProxyType({}),
) -> mpl.axes.Axes:

    if ax is None:
        ax = plt.gca()

    xs = np.linspace(x0, x1, 512)
    ys = np.linspace(y0, y1, 512)

    colors = [*it.starmap(cmap, zip(xs, ys))]

    data = {
        "horizontal": [colors],
        "vertical": [[c] for c in colors],
    }[orient]
    ax.imshow(
        np.array(data),
        aspect="auto",
        origin="lower",
        **imshow_kws,
    )

    return ax
