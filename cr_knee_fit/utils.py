from typing import Iterable

from matplotlib.artist import Artist
from matplotlib.axes import Axes


def add_log_margin(min: float, max: float, log_margin: float = 0.1) -> tuple[float, float]:
    frac = max / min
    margin = frac**log_margin
    return min / margin, max * margin


E_GEV_LABEL = "$E$ / $\\text{GeV}$"


def label_energy_flux(ax: Axes, scale: float) -> None:
    ax.set_xlabel(E_GEV_LABEL)
    if scale == 0:
        ax.set_ylabel(
            "$ F $ / $ \\text{GeV}^{-1} \\; \\text{m}^{-2} \\; \\text{s}^{-1} \\; \\text{sr}^{-1} $"
        )
    else:
        ax.set_ylabel(
            f"$ E^{{{scale}}} F $ / $ \\text{{GeV}}^{{{scale - 1:.2g}}} \\; \\text{{m}}^{{-2}} \\; \\text{{s}}^{{-1}} \\; \\text{{sr}}^{{-1}} $"
        )


def legend_with_added_items(ax: Axes, items: Iterable[tuple[Artist, str]], **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    for artist, label in items:
        handles.append(artist)
        labels.append(label)
    ax.legend(handles, labels, **kwargs)
