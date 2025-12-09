import dataclasses
import itertools
from typing import Any, Callable, cast

import matplotlib
import matplotlib.colors
import matplotlib.ticker as ticker
import matplotlib.tri
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cr_knee_fit.elements import Element, unresolved_element_names
from cr_knee_fit.fit_data import CRSpectrumData, Data, DataConfig, GenericExperimentData
from cr_knee_fit.inference import (
    get_energy_scale_lg_uncertainty,
)
from cr_knee_fit.model import Model, ModelConfig
from cr_knee_fit.utils import (
    E_GEV_LABEL,
    LN_A_LABEL,
    LegendItem,
    add_elements_lnA_secondary_axis,
    add_log_margin,
    clamp_log_margin,
    legend_artist_line,
    legend_with_added_items,
    merged_lims,
)

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
    cl: float = 0.9,
    alpha: float = 0.3,
    grid_override: np.ndarray | None = None,
) -> None:
    x_min, x_max = bounds
    x_grid = (
        grid_override
        if grid_override is not None
        else np.logspace(np.log10(x_min), np.log10(x_max), 100)
    )
    scale_factor = x_grid**scale

    model_sample = [Model.unpack(theta, layout_info=model_config) for theta in theta_sample]
    observable_sample = np.vstack([observable(model, x_grid) for model in model_sample])
    quantile = (1 - cl) / 2
    lower = np.quantile(observable_sample, q=quantile, axis=0)
    upper = np.quantile(observable_sample, q=1 - quantile, axis=0)

    ax.fill_between(
        x_grid,
        scale_factor * lower,
        scale_factor * upper,
        color=color,
        alpha=alpha,
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
    return ax.tricontourf(tri, z_pts, **kwargs)  # type: ignore


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


@dataclasses.dataclass(frozen=True)
class PosteriorPlotConfig:
    best_fit: bool = True
    contours: bool = False
    band_cl: float | None = 0.90
    max_margin_around_data: None | float = 0.5

    tricontourf_kwargs_override: dict = dataclasses.field(default_factory=dict)

    population_contribs_best_fit: bool = False


@dataclasses.dataclass
class PlotExportOpts:
    main: str | None = None


@dataclasses.dataclass
class PlotsConfig:
    validation_data_config: DataConfig | None = None
    elements: PosteriorPlotConfig = PosteriorPlotConfig()
    all_particle: PosteriorPlotConfig = PosteriorPlotConfig()
    all_particle_elements_contribution: PosteriorPlotConfig | None = PosteriorPlotConfig()
    all_particle_scaled_elements_contribution: PosteriorPlotConfig | None = PosteriorPlotConfig()
    all_particle_unresolved_elements_contribution: PosteriorPlotConfig | None = (
        PosteriorPlotConfig()
    )
    lnA: PosteriorPlotConfig = PosteriorPlotConfig()
    energy_shifts: PosteriorPlotConfig = PosteriorPlotConfig()

    export_opts: PlotExportOpts = dataclasses.field(default_factory=PlotExportOpts)


def plot_everything(
    plots_config: PlotsConfig,
    theta_sample: np.ndarray,
    theta_bestfit: np.ndarray,
    model_config: ModelConfig,
    spectra_scale: float,
    fit_data: Data,
    validation_data: Data,
    axes: dict[str, Axes] | None = None,
    legend_ncol: int = 4,
) -> Figure:
    best_fit_model = Model.unpack(theta_bestfit, layout_info=model_config)

    if axes is None:
        fig, axes = plt.subplot_mosaic(
            [
                ["Elements", "Elements"],
                ["All particle", "lnA"],
                ["Shifts", "Shifts"],
            ],
            figsize=(8, 7),
            height_ratios=[1, 1, 0.3],
        )
    else:
        fig = cast(Figure, next(iter(axes.values())).figure)

    ax_el = axes["Elements"]
    ax_all = axes["All particle"]
    ax_lnA = axes["lnA"]
    ax_shifts = axes["Shifts"]

    POP_CONTRIB_LINEWIDTH = 0.75

    def plot_model_predictions(
        ax: Axes,
        observable: Observable,
        E_bounds: tuple[float, float],
        plot_config: PosteriorPlotConfig,
        color: str,
        scale_override: float | None = None,
    ) -> None:
        scale_ = scale_override if scale_override is not None else spectra_scale
        if plot_config.contours:
            tricontourf_kwargs = tricontourf_kwargs_transparent_colors(
                color=color,
                alpha_max=0.5,
            )
            tricontourf_kwargs.update(plot_config.tricontourf_kwargs_override)
            plot_posterior_contours(
                ax,
                scale=scale_,
                theta_sample=theta_sample,
                model_config=model_config,
                observable=observable,
                bounds=E_bounds,
                tricontourf_kwargs=tricontourf_kwargs,
            )
        if plot_config.band_cl is not None:
            plot_credible_band(
                ax,
                scale=scale_,
                theta_sample=theta_sample,
                model_config=model_config,
                observable=observable,
                bounds=E_bounds,
                color=color,
                alpha=0.2,
                cl=plot_config.band_cl,
            )
        if plot_config.best_fit:
            E_grid = np.geomspace(*E_bounds, 100)
            E_factor = E_grid**scale_
            ax.plot(E_grid, E_factor * observable(best_fit_model, E_grid), color=color)

    # elemental spectra
    element_legend_items: list[LegendItem] = []
    experiment_legend_item_by_label: dict[str, LegendItem] = {}
    plotted_elem_spectra: list[CRSpectrumData] = []
    for data, is_fitted in ((fit_data, True), (validation_data, False)):
        for exp, data_by_particle in data.element_spectra.items():
            f_exp = best_fit_model.energy_shifts.f(exp)
            for _, spec_data in data_by_particle.items():
                spec_data = spec_data.with_shifted_energy_scale(f=f_exp)
                plotted_elem_spectra.append(spec_data)
                spec_data.plot(
                    scale=spectra_scale, ax=ax_el, add_legend_label=False, is_fitted=is_fitted
                )
            experiment_legend_item_by_label.setdefault(
                exp.name, (exp.legend_artist(True), exp.name)
            )
    elements = best_fit_model.layout_info().elements(only_fixed_Z=False)
    comp_data_ylim = merged_lims([sp.scaled_flux(spectra_scale) for sp in plotted_elem_spectra])
    comp_data_Elim = merged_lims([sp.E for sp in plotted_elem_spectra])
    comp_Elim = add_log_margin(*comp_data_Elim)
    for element in elements:
        plot_model_predictions(
            ax=ax_el,
            observable=lambda model, E: model.compute_spectrum(E, element=element),
            E_bounds=comp_Elim,
            plot_config=plots_config.elements,
            color=element.color,
        )
        element_legend_items.append((legend_artist_line(element.color), element.name))

    if plots_config.elements.population_contribs_best_fit and len(best_fit_model.populations) > 1:
        multipop_elements = [
            element
            for element in Element.regular()
            if len([pop for pop in best_fit_model.populations if element in pop.all_elements]) > 1
        ]
        E_grid = np.geomspace(*comp_Elim, 300)
        E_factor = E_grid**spectra_scale
        for pop in best_fit_model.populations:
            for element in pop.resolved_elements:
                if element not in multipop_elements:
                    continue
                ax_el.plot(
                    E_grid,
                    E_factor * pop.compute(E_grid, element=element),
                    color=element.color,
                    linewidth=POP_CONTRIB_LINEWIDTH,
                    linestyle=pop.linestyle,
                )

    if plots_config.elements.max_margin_around_data is not None:
        clamp_log_margin(ax_el, comp_data_ylim, plots_config.elements.max_margin_around_data)
    ax_el.set_xlim(*comp_Elim)

    # all-particle spectra
    if fit_data.all_particle_spectra or validation_data.all_particle_spectra:
        plotted_all_spectra: list[CRSpectrumData] = []
        for data, is_fitted in ((fit_data, True), (validation_data, False)):
            for exp, spec_data in data.all_particle_spectra.items():
                f_exp = best_fit_model.energy_shifts.f(exp)
                spec_data = spec_data.with_shifted_energy_scale(f=f_exp)
                plotted_all_spectra.append(spec_data)
                spec_data.plot(
                    scale=spectra_scale,
                    ax=ax_all,
                    add_legend_label=False,
                    is_fitted=is_fitted,
                )
                experiment_legend_item_by_label.setdefault(
                    exp.name, (exp.legend_artist(True), exp.name)
                )

        all_data_ylim = merged_lims([sp.scaled_flux(spectra_scale) for sp in plotted_all_spectra])
        all_data_Elim = merged_lims([sp.E for sp in plotted_all_spectra])
        all_Elim = add_log_margin(*all_data_Elim)
        ALL_PARTICLE_COLOR = "black"
        plot_model_predictions(
            ax=ax_all,
            observable=lambda model, E: model.compute_spectrum(E, element=None),
            E_bounds=all_Elim,
            plot_config=plots_config.all_particle,
            color=ALL_PARTICLE_COLOR,
        )
        element_legend_items.append((legend_artist_line(ALL_PARTICLE_COLOR), "All particle"))

        if plots_config.all_particle_elements_contribution is not None:
            for element in elements:
                plot_model_predictions(
                    ax=ax_all,
                    observable=lambda model, E: model.compute_spectrum(E, element=element),
                    E_bounds=all_Elim,
                    plot_config=plots_config.all_particle_elements_contribution,
                    color=element.color,
                )

        if plots_config.all_particle_scaled_elements_contribution and any(
            pop_conf.rescale_all_particle
            or any(comp.scale_contrib_to_allpart for comp in pop_conf.component_configs)
            for pop_conf in model_config.population_configs
        ):
            plot_model_predictions(
                ax=ax_all,
                observable=lambda model, E: sum(
                    (pop.compute_extra_all_particle_contribution(E) for pop in model.populations),
                    np.zeros_like(E),
                ),
                E_bounds=all_Elim,
                plot_config=plots_config.all_particle_scaled_elements_contribution,
                color="gray",
            )
            element_legend_items.append((legend_artist_line("gray"), "Extra contribution"))

        if plots_config.all_particle_unresolved_elements_contribution and any(
            pop_conf.add_unresolved_elements for pop_conf in model_config.population_configs
        ):
            plot_model_predictions(
                ax=ax_all,
                observable=lambda model, E: sum(
                    (
                        sum(
                            (
                                pop.compute(E, element=element)
                                for element in unresolved_element_names
                            ),
                            np.zeros_like(E),
                        )
                        for pop in model.populations
                    ),
                    np.zeros_like(E),
                ),
                E_bounds=all_Elim,
                plot_config=plots_config.all_particle_unresolved_elements_contribution,
                color="magenta",
            )
            element_legend_items.append((legend_artist_line("magenta"), "Unresolved elements"))

        if (
            plots_config.all_particle.population_contribs_best_fit
            and len(best_fit_model.populations) > 1
        ):
            E_grid = np.geomspace(*all_Elim, 300)
            E_factor = E_grid**spectra_scale
            for pop in best_fit_model.populations:
                ax_all.plot(
                    E_grid,
                    E_factor * pop.compute_all_particle(E_grid),
                    color=ALL_PARTICLE_COLOR,
                    linewidth=POP_CONTRIB_LINEWIDTH,
                    linestyle=pop.linestyle,
                )

        if plots_config.all_particle.max_margin_around_data is not None:
            clamp_log_margin(
                ax_all, all_data_ylim, plots_config.all_particle.max_margin_around_data
            )
        ax_all.set_xlim(*all_Elim)

    # <lnA>
    if fit_data.lnA or validation_data.lnA:
        LN_A_COLOR = "tab:green"
        plotted_lnA_data: list[GenericExperimentData] = []
        for data, is_fitted in ((fit_data, True), (validation_data, False)):
            for exp, lnA_data in data.lnA.items():
                f_exp = best_fit_model.energy_shifts.f(exp)
                lnA_data = dataclasses.replace(lnA_data, x=lnA_data.x * f_exp)
                plotted_lnA_data.append(lnA_data)
                lnA_data.plot(
                    scale=0,
                    ax=ax_lnA,
                    add_legend_label=False,
                    color=LN_A_COLOR,
                    is_fitted=is_fitted,
                )
                experiment_legend_item_by_label.setdefault(
                    exp.name, (exp.legend_artist(True), exp.name)
                )

        lnA_data_ylim = merged_lims([s.y for s in plotted_lnA_data])
        lnA_data_Elim = merged_lims([s.x for s in plotted_lnA_data])
        lnA_Elim = add_log_margin(*lnA_data_Elim)
        plot_model_predictions(
            ax=ax_lnA,
            observable=lambda model, E: model.compute_lnA(E),
            E_bounds=lnA_Elim,
            plot_config=plots_config.lnA,
            color=LN_A_COLOR,
            scale_override=0,
        )
        ax_lnA.set_xlabel(E_GEV_LABEL)
        ax_lnA.set_ylabel(LN_A_LABEL)
        add_elements_lnA_secondary_axis(ax_lnA)
        if plots_config.lnA.max_margin_around_data is not None:
            clamp_log_margin(ax_lnA, lnA_data_ylim, plots_config.lnA.max_margin_around_data)
        ax_lnA.set_xlim(*lnA_Elim)

    # experimental energy scale shifts
    exp_indices = np.arange(len(model_config.shifted_experiments))
    SHIFTS_COLOR = "black"
    for i, exp in enumerate(model_config.shifted_experiments):
        ax_shifts.errorbar(
            [i],
            y=[0],
            yerr=[
                [100 * (10 ** get_energy_scale_lg_uncertainty(exp) - 1)],
                [100 * (1 - 10 ** (-get_energy_scale_lg_uncertainty(exp)))],
            ],
            marker=exp.marker,
            color=SHIFTS_COLOR,
            markersize=3.0,
            elinewidth=0.5,
            capsize=1.5,
        )
        observable: Observable = lambda model, grid: (  # noqa: E731
            100 * (model.energy_shifts.f(exp) - 1)
        ) * np.ones_like(grid)
        bounds = (i - 0.5, i + 0.5)
        if plots_config.energy_shifts.contours:
            tricontourf_kwargs = tricontourf_kwargs_transparent_colors(
                color=SHIFTS_COLOR,
                alpha_max=0.5,
            )
            tricontourf_kwargs.update(plots_config.energy_shifts.tricontourf_kwargs_override)
            plot_posterior_contours(
                ax=ax_shifts,
                scale=0,
                theta_sample=theta_sample,
                model_config=model_config,
                observable=observable,
                bounds=bounds,
                tricontourf_kwargs=tricontourf_kwargs,
            )
        if plots_config.energy_shifts.band_cl is not None:
            plot_credible_band(
                ax=ax_shifts,
                scale=0,
                theta_sample=theta_sample,
                model_config=model_config,
                observable=observable,
                bounds=bounds,
                color=SHIFTS_COLOR,
                alpha=0.2,
                cl=plots_config.energy_shifts.band_cl,
                grid_override=np.array(bounds),
            )
        if plots_config.energy_shifts.best_fit:
            ax_shifts.plot(bounds, observable(best_fit_model, np.array(bounds)), color=SHIFTS_COLOR)

    ax_shifts.axhline(0, linestyle="--", color="gray")
    ax_shifts.set_ylabel("$ \\delta E $ / %")
    ax_shifts.yaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax_shifts.set_xticks(
        exp_indices,
        [exp.name for exp in model_config.shifted_experiments],
        # rotation=30,
        fontsize="x-small",
    )
    ax_shifts.set_xlim(-0.5, exp_indices[-1] + 0.5)

    # legending and general plot formatting
    for ax in (ax_el, ax_all):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())  # never caption minor ticks!
    ax_lnA.set_xscale("log")

    legend_items = element_legend_items.copy()
    if (
        plots_config.elements.population_contribs_best_fit
        or plots_config.all_particle.population_contribs_best_fit
    ):
        for pop in best_fit_model.populations:
            legend_items.append(
                (
                    legend_artist_line(
                        color="gray", linestyle=pop.linestyle, linewidth=POP_CONTRIB_LINEWIDTH
                    ),
                    (pop.population_meta.name if pop.population_meta else "Unnamed") + " pop.",
                )
            )
    legend_items += list(experiment_legend_item_by_label.values())
    legend_with_added_items(
        ax_el,
        legend_items,
        fontsize="small",
        bbox_to_anchor=(0.00, 1.05, 1.0, 0.0),
        loc="lower left",
        fancybox=True,
        shadow=True,
        ncol=legend_ncol,
    )

    fig.tight_layout()
    return fig
