from typing import Callable, Iterable

import numpy as np
from matplotlib.axes import Axes

from cr_knee_fit.model import Model, ModelConfig

Observable = Callable[[Model, np.ndarray], np.ndarray]


def plot_credible_band(
    ax: Axes,
    scale: float,
    E_bounds: tuple[float, float],
    theta_sample: np.ndarray,
    model_config: ModelConfig,
    observable: Observable,
    color: str,
    add_median: bool = False,
) -> None:
    Emin, Emax = E_bounds
    E_grid = np.logspace(np.log10(Emin), np.log10(Emax), 100)
    E_factor = E_grid**scale

    model_sample = [Model.unpack(theta, layout_info=model_config) for theta in theta_sample]
    observable_sample = np.vstack([observable(model, E_grid) for model in model_sample])
    lower = np.quantile(observable_sample, q=0.05, axis=0)
    upper = np.quantile(observable_sample, q=0.95, axis=0)

    ax.fill_between(
        E_grid,
        E_factor * lower,
        E_factor * upper,
        color=color,
        alpha=0.3,
        edgecolor="none",
    )

    if add_median:
        median_model = Model.unpack(np.median(theta_sample, axis=0), layout_info=model_config)
        ax.plot(E_grid, E_factor * observable(median_model, E_grid), color=color)
