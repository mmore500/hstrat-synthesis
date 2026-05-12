import types
import typing

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


def draw_cgrid(
    cmap: typing.Callable,
    *,
    ax: typing.Optional[mpl.axes.Axes] = None,
    xmax: float = 1.0,
    ymax: float = 1.0,
    imshow_kws: dict = types.MappingProxyType({}),
) -> mpl.axes.Axes:

    if ax is None:
        ax = plt.gca()

    xs = np.linspace(0, 1, 512)
    ys = np.linspace(0, 1, 512)
    X, Y = np.meshgrid(xs, ys)

    data = np.vectorize(cmap)(X, Y)

    ax.imshow(
        np.dstack(data),
        aspect="auto",
        extent=(0, xmax, 0, ymax),
        origin="lower",
        **imshow_kws,
    )

    return ax
