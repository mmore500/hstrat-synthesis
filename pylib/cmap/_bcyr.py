import matplotlib as mpl


def get_color(x: float, y: float) -> tuple[float, float, float]:
    return (y, x, 1.0 - y)


def get_hex(x: float, y: float) -> str:
    return mpl.colors.to_hex(get_color(x, y))
