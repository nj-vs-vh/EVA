from typing import Iterable

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


def add_log_margin(min: float, max: float, log_margin: float = 0.1) -> tuple[float, float]:
    frac = max / min
    margin = frac**log_margin
    return min / margin, max * margin


E_GEV_LABEL: str = "$E$ / $\\text{GeV}$"


def label_energy_flux(ax: Axes, scale: float) -> None:
    ax.set_xlabel(E_GEV_LABEL)
    if scale == 0:
        ax.set_ylabel(
            "$ F $ / $ \\text{GeV}^{-1} \\; \\text{m}^{-2} \\; \\text{s}^{-1} \\; \\text{sr}^{-1} $"
        )
    else:
        ax.set_ylabel(
            f"$ E^{{{scale}}} F $ / $ \\text{{GeV}}^{{{scale - 1:.3g}}} \\; \\text{{m}}^{{-2}} \\; \\text{{s}}^{{-1}} \\; \\text{{sr}}^{{-1}} $"
        )


LegendItem = tuple[Artist, str]


def legend_with_added_items(ax: Axes, items: Iterable[LegendItem], **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    for artist, label in items:
        handles.append(artist)
        labels.append(label)
    ax.legend(handles, labels, **kwargs)


def energy_shift_suffix(f: float) -> str:
    if np.isclose(f, 1.0):
        return ""
    shift_percent = abs(100 * (f - 1))
    shift_sign = "+" if f > 1 else "-"
    return f" $(E \\; {shift_sign} {shift_percent:.1f} \\%)$"


def legend_artist_line(color: str) -> Line2D:
    return Line2D([], [], color=color, marker="none")
