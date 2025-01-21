from typing import Callable, Iterable

import numpy as np
from matplotlib.axes import Axes

from cr_knee_fit.model import Model

Observable = Callable[[Model, np.ndarray], np.ndarray]


def plot_credible_band(
    ax: Axes,
    scale: float,
    E_bounds: tuple[float, float],
    model_sample: Iterable[Model],
    observable: Observable,
    color: str,
) -> None:
    Emin, Emax = E_bounds
    E_grid = np.logspace(np.log10(Emin), np.log10(Emax), 100)

    predictions_sample = np.vstack([observable(model, E_grid) for model in model_sample])
    lower = np.quantile(predictions_sample, q=0.05, axis=0)
    upper = np.quantile(predictions_sample, q=0.95, axis=0)

    E_factor = E_grid**scale
    ax.fill_between(
        E_grid,
        E_factor * lower,
        E_factor * upper,
        color=color,
        alpha=0.3,
        edgecolor="none",
    )
