import dataclasses
import itertools
import random
from collections.abc import Callable
from typing import Any, cast

import matplotlib
import matplotlib.colors
import matplotlib.tri
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cr_knee_fit.elements import Element
from cr_knee_fit.experiments import Experiment
from cr_knee_fit.fit_data import (
    CRSpectrumData,
    Data,
    DataConfig,
    GenericExperimentData,
    SpectrumDataSpec,
    spectrum_data_spec_label,
)
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
    energy_shift_suffix,
    legend_artist_line,
    legend_with_added_items,
    merged_lims,
)

Observable = Callable[[Model, np.ndarray], np.ndarray]


def _to_model_sample(theta_sample: np.ndarray | list[Model], model_config: ModelConfig):
    if isinstance(theta_sample, list):
        return theta_sample
    else:
        return [Model.unpack(theta, layout_info=model_config) for theta in theta_sample]


def plot_credible_band(
    ax: Axes,
    scale: float,
    bounds: tuple[float, float],
    model_sample: list[Model],
    observable: Observable,
    color: str,
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


def plot_posterior_contours(
    ax: Axes,
    scale: float,
    model_sample: list[Model],
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

    observable_sample = np.vstack(
        [scale_factor * observable(model, x_grid) for model in model_sample]
    )

    x_grid_: list[float] = []
    y_hists: list[np.ndarray] = []
    z_hists: list[np.ndarray] = []
    for x, sample_at_x in zip(x_grid, observable_sample.T):
        sample_at_x = sample_at_x[np.isfinite(sample_at_x)]
        if sample_at_x.size == 0:
            continue
        hist, edges = np.histogram(sample_at_x, bins=30, density=False)
        centers = 0.5 * (edges[1:] + edges[:-1])
        x_grid_.append(x)
        y_hists.append(centers)
        z_hists.append(hist)
    x_grid = np.array(x_grid_)

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
    theta_sample: np.ndarray | list[Model],
    model_config: ModelConfig,
    observable: Observable,
    n_samples: int,
    color: str,
    label: str | None = None,
    randomized: bool = False,
) -> None:
    x_min, x_max = bounds
    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    scale_factor = x_grid**scale

    model_sample = _to_model_sample(theta_sample, model_config).copy()
    if randomized:
        random.shuffle(model_sample)

    for i, model in enumerate(model_sample[:n_samples]):
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
    margin_around_fitted_data_only: bool = False

    tricontourf_kwargs_override: dict = dataclasses.field(default_factory=dict)

    population_contribs_best_fit: bool = False

    ylim_override: tuple[float | None, float | None] | None = None
    xlim_override: tuple[float | None, float | None] | None = None

    def apply_limits(
        self, ax: Axes, data_ylim: tuple[float, float], data_xlim: tuple[float, float]
    ) -> None:
        if self.ylim_override is not None:
            ax.set_ylim(*self.ylim_override)
        elif self.max_margin_around_data is not None:
            clamp_log_margin(ax, data_ylim, self.max_margin_around_data)

        if self.xlim_override:
            ax.set_xlim(*self.xlim_override)
        else:
            ax.set_xlim(data_xlim)

    def get_model_xlim(self, data_xlim: tuple[float, float]) -> tuple[float, float]:
        default = add_log_margin(*data_xlim)
        override = self.xlim_override if self.xlim_override is not None else (None, None)
        return (override[0] or default[0], override[1] or default[1])


@dataclasses.dataclass
class PlotExportOpts:
    main: str | None = None


@dataclasses.dataclass
class PlotsConfig:
    validation_data_config: DataConfig | None = None

    datasets: bool = True
    observables: bool = True
    corner: bool = True
    energy_density: bool = True

    # main plot detailed settings
    elements: PosteriorPlotConfig = PosteriorPlotConfig()
    all_particle: PosteriorPlotConfig = PosteriorPlotConfig()
    all_particle_elements_contribution: PosteriorPlotConfig | None = PosteriorPlotConfig()
    all_particle_scaled_elements_contribution: PosteriorPlotConfig | None = PosteriorPlotConfig()
    all_particle_unresolved_elements_contribution: PosteriorPlotConfig | None = (
        PosteriorPlotConfig()
    )
    lnA: PosteriorPlotConfig = PosteriorPlotConfig()
    energy_shifts: PosteriorPlotConfig = PosteriorPlotConfig()

    observables_posterior: PosteriorPlotConfig = PosteriorPlotConfig()

    export_opts: PlotExportOpts = dataclasses.field(default_factory=PlotExportOpts)


def plot_model_predictions(
    ax: Axes,
    observable: Observable,
    best: Model,
    model_sample: list[Model] | None,
    E_bounds: tuple[float, float],
    plot_config: PosteriorPlotConfig,
    color: str,
    scale: float,
) -> None:
    if plot_config.contours:
        assert model_sample is not None, "Contrours requested, but no model sample passed"
        tricontourf_kwargs = tricontourf_kwargs_transparent_colors(
            color=color,
            alpha_max=0.5,
        )
        tricontourf_kwargs.update(plot_config.tricontourf_kwargs_override)
        plot_posterior_contours(
            ax,
            scale=scale,
            model_sample=model_sample,
            observable=observable,
            bounds=E_bounds,
            tricontourf_kwargs=tricontourf_kwargs,
        )
    if plot_config.band_cl is not None:
        assert model_sample is not None, "Credible band requested, but no model sample passed"
        plot_credible_band(
            ax,
            scale=scale,
            model_sample=model_sample,
            observable=observable,
            bounds=E_bounds,
            color=color,
            alpha=0.2,
            cl=plot_config.band_cl,
        )
    if plot_config.best_fit:
        E_grid = np.geomspace(*E_bounds, 100)
        E_factor = E_grid**scale
        ax.plot(E_grid, E_factor * observable(best, E_grid), color=color)


POP_CONTRIB_LINEWIDTH = 0.75
ALL_PARTICLE_COLOR = "black"


def plot_everything(
    plots_config: PlotsConfig,
    model_sample: list[Model],
    center: Model,
    spectra_scale: float,
    fit_data: Data,
    validation_data: Data,
    axes: dict[str, Axes] | None = None,
    legend_ncol: int = 4,
) -> Figure:
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

    model_config = center.layout_info()

    ax_el = axes["Elements"]
    ax_all = axes["All particle"]
    ax_lnA = axes["lnA"]
    ax_shifts = axes["Shifts"]

    # elemental spectra
    element_legend_items: list[LegendItem] = []
    experiment_legend_item_by_label: dict[str, LegendItem] = {}
    plotted_elem_spectra: list[CRSpectrumData] = []
    elements_to_plot = set[Element]()
    for data, is_fitted in ((fit_data, True), (validation_data, False)):
        for exp, data_by_particle in data.element_spectra.items():
            f_exp = center.energy_shifts.f(exp)
            for spec_data in data_by_particle.values():
                spec_data = spec_data.with_shifted_energy_scale(f=f_exp)
                plotted_elem_spectra.append(spec_data)
                spec_data.plot(
                    scale=spectra_scale, ax=ax_el, add_legend_label=False, is_fitted=is_fitted
                )
                if isinstance(spec_data.spec, Element):
                    elements_to_plot.add(spec_data.spec)
            experiment_legend_item_by_label.setdefault(
                exp.name, (exp.legend_artist(True), exp.name)
            )

    elem_data_ylim = merged_lims([sp.scaled_flux(spectra_scale) for sp in plotted_elem_spectra])
    elem_data_Elim = merged_lims([sp.E for sp in plotted_elem_spectra])
    elem_Elim = plots_config.elements.get_model_xlim(elem_data_Elim)
    for element in sorted(elements_to_plot):
        plot_model_predictions(
            ax=ax_el,
            best=center,
            model_sample=model_sample,
            observable=lambda model, E: model.compute_spectrum(E, element=element, quantity="E"),  # noqa: B023
            E_bounds=elem_Elim,
            plot_config=plots_config.elements,
            color=element.color,
            scale=spectra_scale,
        )
        element_legend_items.append((legend_artist_line(element.color), element.name))

    if plots_config.elements.population_contribs_best_fit and center.n_populations_eff() > 1:
        E_grid = np.geomspace(*elem_Elim, 300)
        E_factor = E_grid**spectra_scale
        for element in center.multipopulation_elements():
            for pop in center.populations:
                if element not in pop.resolved_elements:
                    continue
                ax_el.plot(
                    E_grid,
                    E_factor * pop.compute_spectrum(E_grid, element=element, quantity="E"),
                    color=element.color,
                    linewidth=POP_CONTRIB_LINEWIDTH,
                    linestyle=pop.linestyle,
                )
            if crams := center.crams:
                ax_el.plot(
                    E_grid,
                    E_factor * crams.compute_spectrum(E_grid, element=element, quantity="E"),
                    color=element.color,
                    linewidth=POP_CONTRIB_LINEWIDTH,
                    linestyle=crams.linestyle,
                )

    plots_config.elements.apply_limits(ax_el, data_ylim=elem_data_ylim, data_xlim=elem_Elim)

    # all-particle spectra
    if fit_data.all_particle_spectra or validation_data.all_particle_spectra:
        plotted_all_spectra: list[CRSpectrumData] = []
        for data, is_fitted in ((fit_data, True), (validation_data, False)):
            for exp, spec_data in data.all_particle_spectra.items():
                f_exp = center.energy_shifts.f(exp)
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
        all_Elim = plots_config.all_particle.get_model_xlim(all_data_Elim)
        plot_model_predictions(
            ax=ax_all,
            best=center,
            model_sample=model_sample,
            observable=lambda model, E: model.compute_spectrum(E, element=None, quantity="E"),
            E_bounds=all_Elim,
            plot_config=plots_config.all_particle,
            color=ALL_PARTICLE_COLOR,
            scale=spectra_scale,
        )
        element_legend_items.append((legend_artist_line(ALL_PARTICLE_COLOR), "All particle"))

        if plots_config.all_particle_elements_contribution is not None:
            for element in elements_to_plot:
                plot_model_predictions(
                    ax=ax_all,
                    best=center,
                    model_sample=model_sample,
                    observable=lambda model, E: model.compute_spectrum(
                        E,
                        element=element,  # noqa: B023
                        quantity="E",
                    ),
                    E_bounds=all_Elim,
                    plot_config=plots_config.all_particle_elements_contribution,
                    color=element.color,
                    scale=spectra_scale,
                )

        if plots_config.all_particle_scaled_elements_contribution and any(
            pop_conf.rescale_all_particle
            or any(comp.scale_contrib_to_allpart for comp in pop_conf.component_configs)
            for pop_conf in model_config.population_configs
        ):
            plot_model_predictions(
                ax=ax_all,
                best=center,
                model_sample=model_sample,
                observable=lambda model, E: sum(
                    (pop.compute_extra_all_particle_contribution(E) for pop in model.populations),
                    np.zeros_like(E),
                ),
                E_bounds=all_Elim,
                plot_config=plots_config.all_particle_scaled_elements_contribution,
                color="gray",
                scale=spectra_scale,
            )
            element_legend_items.append((legend_artist_line("gray"), "Extra contribution"))

        if (
            plots_config.all_particle.population_contribs_best_fit
            and center.n_populations_eff() > 1
        ):
            E_grid = np.geomspace(*all_Elim, 300)
            E_factor = E_grid**spectra_scale
            for pop in center.populations:
                ax_all.plot(
                    E_grid,
                    E_factor * pop.compute_all_particle_spectrum(E_grid),
                    color=ALL_PARTICLE_COLOR,
                    linewidth=POP_CONTRIB_LINEWIDTH,
                    linestyle=pop.linestyle,
                )
            if crams := center.crams:
                ax_el.plot(
                    E_grid,
                    E_factor * crams.compute_all_particle_spectrum(E_grid),
                    color=ALL_PARTICLE_COLOR,
                    linewidth=POP_CONTRIB_LINEWIDTH,
                    linestyle=crams.linestyle,
                )

        plots_config.all_particle.apply_limits(ax_all, data_ylim=all_data_ylim, data_xlim=all_Elim)

    # <lnA>
    if fit_data.lnA or validation_data.lnA:
        LN_A_COLOR = "tab:green"
        plotted_lnA_data: list[GenericExperimentData] = []
        for data, is_fitted in ((fit_data, True), (validation_data, False)):
            for lnA_data in data.lnA:
                exp = lnA_data.experiment
                f_exp = center.energy_shifts.f(exp)
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
        lnA_Elim = plots_config.lnA.get_model_xlim(lnA_data_Elim)
        plot_model_predictions(
            ax=ax_lnA,
            best=center,
            model_sample=model_sample,
            observable=lambda model, E: model.compute_lnA(E),
            E_bounds=lnA_Elim,
            plot_config=plots_config.lnA,
            color=LN_A_COLOR,
            scale=0,
        )
        ax_lnA.set_xlabel(E_GEV_LABEL)
        ax_lnA.set_ylabel(LN_A_LABEL)
        add_elements_lnA_secondary_axis(ax_lnA)
        plots_config.lnA.apply_limits(ax_lnA, data_ylim=lnA_data_ylim, data_xlim=lnA_Elim)

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
        observable: Observable = lambda model, grid: (
            (100 * (model.energy_shifts.f(exp) - 1)) * np.ones_like(grid)  # noqa: B023
        )
        bounds = (i - 0.5, i + 0.5)
        if plots_config.energy_shifts.contours:
            tricontourf_kwargs = tricontourf_kwargs_transparent_colors(
                color=SHIFTS_COLOR,
                alpha_max=0.5,
            )
            tricontourf_kwargs.update(plots_config.energy_shifts.tricontourf_kwargs_override)
            plot_posterior_contours(
                ax=ax_shifts,
                model_sample=model_sample,
                observable=observable,
                bounds=bounds,
                tricontourf_kwargs=tricontourf_kwargs,
                scale=0,
            )
        if plots_config.energy_shifts.band_cl is not None:
            plot_credible_band(
                ax=ax_shifts,
                scale=0,
                model_sample=model_sample,
                observable=observable,
                bounds=bounds,
                color=SHIFTS_COLOR,
                alpha=0.2,
                cl=plots_config.energy_shifts.band_cl,
                grid_override=np.array(bounds),
            )
        if plots_config.energy_shifts.best_fit:
            ax_shifts.plot(bounds, observable(center, np.array(bounds)), color=SHIFTS_COLOR)

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
        if crams := center.crams:
            legend_items.append(
                (
                    legend_artist_line(
                        color="gray", linestyle=crams.linestyle, linewidth=POP_CONTRIB_LINEWIDTH
                    ),
                    (
                        crams.config.population_meta.name
                        if crams.config.population_meta
                        else "Unnamed"
                    )
                    + " pop.",
                )
            )
        for pop in center.populations:
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


def plot_spectrum(
    spec: SpectrumDataSpec,
    config: PosteriorPlotConfig,
    center: Model,
    model_sample: list[Model],
    scale: float,
    fit_data: Data,
    validation_data: Data,
    legend_ncol: int = 4,
    axes: Axes | None = None,
) -> Figure:
    if axes is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        ax = axes
        fig = cast(Figure, ax.figure)

    experiment_legend_items: dict[Experiment, LegendItem] = {}
    plotted_data: list[CRSpectrumData] = []
    fit_data_for_spec = [d for d in fit_data.spectra if d.spec == spec]
    val_data_for_spec = [d for d in validation_data.spectra if d.spec == spec]
    for data_, is_fitted in ((fit_data_for_spec, True), (val_data_for_spec, False)):
        for d in data_:
            exp = d.experiment
            f_exp = center.energy_shifts.f(exp)
            d = d.with_shifted_energy_scale(f=f_exp)
            d.plot(scale=scale, ax=ax, add_legend_label=False, is_fitted=is_fitted)
            experiment_legend_items.setdefault(
                exp,
                (
                    exp.legend_artist(is_fitted=is_fitted),
                    exp.name + energy_shift_suffix(f_exp),
                ),
            )
            if (not config.margin_around_fitted_data_only) or (is_fitted or not fit_data_for_spec):
                plotted_data.append(d)

    if not plotted_data:
        raise RuntimeError(f"No plotted data found for {spec=}")

    match spec:
        case Element():
            plot_elements = [spec]
        case tuple():
            plot_elements = list(spec)
        case None:
            plot_elements = Element.regular()
    plot_allpart = spec is None

    model_legend_items: list[LegendItem] = []
    data_ylim = merged_lims([sp.scaled_flux(scale) for sp in plotted_data])
    elem_data_Elim = merged_lims([sp.E for sp in plotted_data])
    data_Elim = config.get_model_xlim(elem_data_Elim)
    for element in sorted(plot_elements):
        plot_model_predictions(
            ax=ax,
            best=center,
            model_sample=model_sample,
            observable=lambda model, E: model.compute_spectrum(E, element=element, quantity="E"),  # noqa: B023
            E_bounds=data_Elim,
            plot_config=config,
            color=element.color,  # NOTE: change to a neutral color for better readability?
            scale=scale,
        )
        if not plot_allpart:
            model_legend_items.append((legend_artist_line(element.color), element.name))

    if config.population_contribs_best_fit and center.n_populations_eff() > 1:
        E_grid = np.geomspace(*data_Elim, 300)
        E_factor = E_grid**scale
        for element in center.multipopulation_elements():
            for pop in center.populations:
                if element not in pop.resolved_elements:
                    continue
                ax.plot(
                    E_grid,
                    E_factor * pop.compute_spectrum(E_grid, element=element, quantity="E"),
                    color=element.color,
                    linewidth=POP_CONTRIB_LINEWIDTH,
                    linestyle=pop.linestyle,
                )
            if crams := center.crams:
                ax.plot(
                    E_grid,
                    E_factor * crams.compute_spectrum(E_grid, element=element, quantity="E"),
                    color=element.color,
                    linewidth=POP_CONTRIB_LINEWIDTH,
                    linestyle=crams.linestyle,
                )

    if plot_allpart:
        plot_model_predictions(
            ax=ax,
            best=center,
            model_sample=model_sample,
            observable=lambda model, E: model.compute_spectrum(E, element=None, quantity="E"),
            E_bounds=data_Elim,
            plot_config=config,
            color=ALL_PARTICLE_COLOR,
            scale=scale,
        )
        model_legend_items.append((legend_artist_line(ALL_PARTICLE_COLOR), "All particle"))

        if config.population_contribs_best_fit and center.n_populations_eff() > 1:
            E_grid = np.geomspace(*data_Elim, 300)
            E_factor = E_grid**scale
            for pop in center.populations:
                ax.plot(
                    E_grid,
                    E_factor * pop.compute_all_particle_spectrum(E_grid),
                    color=ALL_PARTICLE_COLOR,
                    linewidth=POP_CONTRIB_LINEWIDTH,
                    linestyle=pop.linestyle,
                )
            if crams := center.crams:
                ax.plot(
                    E_grid,
                    E_factor * crams.compute_all_particle_spectrum(E_grid),
                    color=ALL_PARTICLE_COLOR,
                    linewidth=POP_CONTRIB_LINEWIDTH,
                    linestyle=crams.linestyle,
                )

    config.apply_limits(ax, data_ylim=data_ylim, data_xlim=data_Elim)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())  # never caption minor ticks!

    legend_items = model_legend_items.copy()
    if config.population_contribs_best_fit:
        if crams := center.crams:
            legend_items.append(
                (
                    legend_artist_line(
                        color="gray", linestyle=crams.linestyle, linewidth=POP_CONTRIB_LINEWIDTH
                    ),
                    (
                        crams.config.population_meta.name
                        if crams.config.population_meta
                        else "Unnamed"
                    )
                    + " pop.",
                )
            )
        for pop in center.populations:
            legend_items.append(
                (
                    legend_artist_line(
                        color="gray", linestyle=pop.linestyle, linewidth=POP_CONTRIB_LINEWIDTH
                    ),
                    (pop.population_meta.name if pop.population_meta else "Unnamed") + " pop.",
                )
            )
    legend_items += list(experiment_legend_items.values())

    legend_with_added_items(
        ax,
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


def plot_all_observables(
    config: PosteriorPlotConfig,
    center: Model,
    model_sample: list[Model],
    spectra_scale: float,
    fit_data: Data,
    validation_data: Data,
    axes: dict[str, Axes] | None = None,
    legend_ncol: int = 4,
) -> dict[str, Figure]:
    res: dict[str, Figure] = {}

    spectrum_specs = {s.spec for s in fit_data.spectra + validation_data.spectra}
    for spec in spectrum_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_spectrum(
            config=config,
            spec=spec,
            center=center,
            model_sample=model_sample,
            scale=spectra_scale,
            fit_data=fit_data,
            validation_data=validation_data,
            legend_ncol=legend_ncol,
            axes=ax,
        )
        res[spectrum_data_spec_label(spec)] = fig

    # TODO: also plot posterior flux ratios
    # flux_ratios = {frd.ratio for frd in fit_data.flux_ratios + validation_data.flux_ratios}
    # for fr in flux_ratios:
    #     for quantity in ("R", "E_n"):
    #         fig, ax = plt.subplots()
    #         if self._plot_flux_ratios_per_quantity(
    #             ax=ax,
    #             fit_data=fit_data,
    #             validation_data=validation_data,
    #             only_quantity=quantity,
    #             only_ratio=fr,
    #         ):
    #             res[f"ratio_{fr.num.name}_over_{fr.denom.name}_vs_{quantity}"] = fig

    return res
