import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

from cr_knee_fit.elements import Element


def add_log_margin(min: float, max: float, log_margin: float = 0.05) -> tuple[float, float]:
    if min < 0 or max < 0:
        return (min, max)
    ratio = max / min
    margin = ratio**log_margin
    return min / margin, max * margin


def clamp_log_margin(ax: Axes, reference_lims: tuple[float, float], max_log_margin: float) -> None:
    current = ax.get_ylim()
    extremal = add_log_margin(*reference_lims, max_log_margin)
    ax.set_ylim(
        max(current[0], extremal[0]),
        min(current[1], extremal[1]),
    )


def merged_lims(vals: Sequence[np.ndarray]) -> tuple[float, float]:
    merged = np.hstack(vals)
    return merged.min(), merged.max()


E_GEV_LABEL: str = "$E$ / $\\text{GeV}$"


def label_energy_flux(ax: Axes, scale: float) -> None:
    ax.set_xlabel(E_GEV_LABEL)
    if scale == 0:
        ax.set_ylabel(
            "$ I $ / $ \\text{GeV}^{-1} \\; \\text{m}^{-2} \\; \\text{s}^{-1} \\; \\text{sr}^{-1} $"
        )
    else:
        ax.set_ylabel(
            f"$ E^{{{scale}}} F $ / $ \\text{{GeV}}^{{{scale - 1:.3g}}} \\; \\text{{m}}^{{-2}} \\; \\text{{s}}^{{-1}} \\; \\text{{sr}}^{{-1}} $"
        )


LegendItem = tuple[Artist, str]


def legend_with_added_items(ax: Axes, items: Iterable[LegendItem], **kwargs) -> Legend:
    handles, labels = ax.get_legend_handles_labels()
    for artist, label in items:
        handles.append(artist)
        labels.append(label)
    return ax.legend(handles, labels, **kwargs)


def add_elements_lnA_secondary_axis(ax: Axes) -> None:
    lnA_min, lnA_max = ax.get_ylim()
    tick_elements = [el for el in Element.regular() if lnA_min < el.lnA < lnA_max]
    ax_elements = ax.twinx()
    ax_elements.set_yticks(
        ticks=[e.lnA for e in tick_elements],
        labels=[e.name for e in tick_elements],
        minor=False,
    )
    ax_elements.grid(
        visible=True, which="major", axis="y", color="gray", linestyle="--", linewidth=0.75
    )


def energy_shift_suffix(f: float) -> str:
    if np.isclose(f, 1.0):
        return ""
    shift_percent = abs(100 * (f - 1))
    shift_sign = "+" if f > 1 else "-"
    return f" $(E \\; {shift_sign} {shift_percent:.1f} \\%)$"


def legend_artist_line(color: str) -> Line2D:
    return Line2D([], [], color=color, marker="none")


EXPORT_DIR = Path(__file__).parent / "../../articles/cr-knees-fit/figs"


def export_fig(fig: Figure, filename: str) -> None:
    try:
        assert EXPORT_DIR.exists(), f"Export dir doesn't exist: {EXPORT_DIR.resolve()}"

        path = EXPORT_DIR / filename
        for cmd in (
            ("git", "pull"),
            None,
            ("git", "add", filename),
            ("git", "commit", "-m", "'exported from script'"),
            ("git", "push"),
        ):
            if cmd is None:
                print(f"Saving {filename}")
                fig.savefig(path)
            else:
                print(f"Running {' '.join(cmd)} from {EXPORT_DIR}...")
                subprocess.run(cmd, cwd=EXPORT_DIR)
    except Exception as e:
        print(f"Failed to export figure: {e}")
