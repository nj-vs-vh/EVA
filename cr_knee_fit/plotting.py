import itertools
from typing import Any, Callable

import matplotlib
import matplotlib.colors
import matplotlib.tri
import numpy as np
from matplotlib.axes import Axes

from cr_knee_fit.model_ import Model, ModelConfig

Observable = Callable[[Model, np.ndarray], np.ndarray]


def plot_credible_band(
    ax: Axes,
    scale: float,
    bounds: tuple[float, float],
    theta_sample: np.ndarray,
    model_config: ModelConfig,
    observable: Observable,
    color: str,
    add_median: bool = False,
    label: str | None = None,
) -> None:
    x_min, x_max = bounds
    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    scale_factor = x_grid**scale

    model_sample = [Model.unpack(theta, layout_info=model_config) for theta in theta_sample]
    observable_sample = np.vstack([observable(model, x_grid) for model in model_sample])
    lower = np.quantile(observable_sample, q=0.05, axis=0)
    upper = np.quantile(observable_sample, q=0.95, axis=0)

    ax.fill_between(
        x_grid,
        scale_factor * lower,
        scale_factor * upper,
        color=color,
        alpha=0.3,
        edgecolor="none",
        label=label,
    )

    if add_median:
        median_model = Model.unpack(np.median(theta_sample, axis=0), layout_info=model_config)
        ax.plot(x_grid, scale_factor * observable(median_model, x_grid), color=color)


def plot_posterior_contours(
    ax: Axes,
    scale: float,
    theta_sample: np.ndarray,
    model_config: ModelConfig,
    observable: Observable,
    bounds: tuple[float, float] | None = None,
    x_grid: np.ndarray | None = None,
    tricontourf_kwargs: dict[str, Any] | None = None,
) -> matplotlib.tri.TriContourSet:
    if bounds is not None:
        if x_grid is not None:
            raise ValueError("bounds and x_grid args are mutually exclusive")
        x_min, x_max = bounds
        x_grid = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    if x_grid is None:
        raise ValueError("bounds or x_grid must be specified")

    scale_factor = x_grid**scale
    model_sample = [Model.unpack(theta, layout_info=model_config) for theta in theta_sample]
    observable_sample = np.vstack(
        [scale_factor * observable(model, x_grid) for model in model_sample]
    )

    y_hists: list[np.ndarray] = []
    z_hists: list[np.ndarray] = []
    for sample_at_x in observable_sample.T:
        hist, edges = np.histogram(sample_at_x, bins=30, density=False)
        centers = 0.5 * (edges[1:] + edges[:-1])
        y_hists.append(centers)
        z_hists.append(hist)

    x_pts: list[float] = []
    y_pts: list[float] = []
    z_pts: list[float] = []
    triangles = np.empty(shape=(0, 3), dtype=int)
    for (_, (x_1_value, y_1, z_1)), (
        i_2,
        (x_2_value, y_2, z_2),
    ) in itertools.pairwise(enumerate(zip(x_grid, y_hists, z_hists))):
        x_1 = x_1_value * np.ones_like(y_1)
        x_2 = x_2_value * np.ones_like(y_2)
        tri_layer = matplotlib.tri.Triangulation(
            x=np.hstack((x_1, x_2)),
            y=np.hstack((y_1, y_2)),
        )
        triangles = np.vstack((triangles, len(x_pts) + tri_layer.triangles))
        x_pts.extend(x_1)
        y_pts.extend(y_1)
        z_pts.extend(z_1)
        if i_2 == x_grid.size - 1:
            x_pts.extend(x_2)
            y_pts.extend(y_2)
            z_pts.extend(z_2)

    kwargs = {
        "levels": 10,
        "cmap": "viridis",
    }
    if tricontourf_kwargs:
        kwargs.update(tricontourf_kwargs)
    tri = matplotlib.tri.Triangulation(x_pts, y_pts, triangles=triangles)
    return ax.tricontourf(tri, z_pts, **kwargs)


def tricontourf_kwargs_transparent_colors(
    color: str,
    levels: int = 10,
    alpha_min: float = 0.1,
    alpha_max: float = 0.8,
):
    return {
        "levels": levels,
        "colors": [
            matplotlib.colors.to_rgba(color, alpha=alpha)
            for alpha in np.linspace(alpha_min, alpha_max, levels)
        ],
        "cmap": None,
    }


def plot_ghostly_lines(
    ax: Axes,
    scale: float,
    bounds: tuple[float, float],
    theta_sample: np.ndarray,
    model_config: ModelConfig,
    observable: Observable,
    n_samples: int,
    color: str,
    label: str | None = None,
) -> None:
    x_min, x_max = bounds
    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    scale_factor = x_grid**scale

    n_total = theta_sample.shape[0]
    fraction = n_samples / n_total
    mask = np.random.random(size=theta_sample.shape[0]) < fraction
    for i, theta in enumerate(theta_sample[mask, :]):
        model = Model.unpack(theta, layout_info=model_config)
        obs = observable(model, x_grid)
        ax.plot(
            x_grid,
            scale_factor * obs,
            color=color,
            alpha=max(1 / n_samples, 0.01),
            label=label if i == 0 else None,
        )
