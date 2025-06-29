import argparse
import contextlib
import dataclasses
import datetime
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Callable, cast
from warnings import warn

import corner  # type: ignore
import emcee  # type: ignore
import matplotlib.ticker as ticker
import numpy as np
import pydantic
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic_numpy.typing import Np2DArrayFp64  # type: ignore
from scipy import optimize  # type: ignore

from cr_knee_fit.elements import Element, unresolved_element_names
from cr_knee_fit.fit_data import CRSpectrumData, Data, DataConfig, GenericExperimentData
from cr_knee_fit.inference import (
    get_energy_scale_lg_uncertainty,
    loglikelihood,
    logposterior,
    set_global_fit_data,
)
from cr_knee_fit.model_ import Model, ModelConfig
from cr_knee_fit.plotting import (
    Observable,
    plot_credible_band,
    plot_posterior_contours,
    tricontourf_kwargs_transparent_colors,
)
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts
from cr_knee_fit.utils import (
    E_GEV_LABEL,
    LegendItem,
    add_elements_lnA_secondary_axis,
    add_log_margin,
    clamp_log_margin,
    export_fig,
    legend_artist_line,
    legend_with_added_items,
    merged_lims,
)

# as recommended by emceee parallelization guide
# see https://emcee.readthedocs.io/en/stable/tutorials/parallel/#parallelization
os.environ["OMP_NUM_THREADS"] = "1"

IS_CLUSTER = os.environ.get("CRKNEES_CLUSTER") == "1"


@dataclasses.dataclass
class McmcConfig:
    n_steps: int
    n_walkers: int
    processes: int
    reuse_saved: bool = True


@dataclasses.dataclass(frozen=True)
class PosteriorPlotConfig:
    best_fit: bool = True
    contours: bool = False
    band_cl: float | None = 0.90
    max_margin_around_data: None | float = 0.5

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


class FitConfig(pydantic.BaseModel):
    name: str
    fit_data_config: DataConfig
    mcmc: McmcConfig | None
    model: ModelConfig
    plots: PlotsConfig
    initial_guesses: Np2DArrayFp64  # n_sample x n_model_dim

    reuse_saved_models: bool = False

    def __post_init__(self) -> None:
        model_elements = set(self.model.elements(only_fixed_Z=True))
        data_elements = set(self.fit_data_config.elements)
        unconstrained_elements = model_elements - data_elements
        if unconstrained_elements:
            warn(
                f"Some elements in the model are not contstrained by data: {sorted(unconstrained_elements)}"
            )

    @classmethod
    def from_guessing_func(
        cls,
        name: str,
        fit_data: DataConfig,
        mcmc: McmcConfig | None,
        plots: PlotsConfig,
        generate_guess: Callable[[], Model],
        n_guesses: int = 100,
    ) -> "FitConfig":
        guesses = [generate_guess() for _ in range(n_guesses)]
        assert len({g.ndim() for g in guesses}) == 1, (
            "guess generation function generates different-dimensional models"
        )

        return FitConfig(
            name=name,
            fit_data_config=fit_data,
            mcmc=mcmc,
            plots=plots,
            model=guesses[0].layout_info(),
            initial_guesses=np.array([guess.pack() for guess in guesses]),
        )

    def generate_initial_guess(self, data: Data) -> Model:
        n_try = 1000
        for _ in range(n_try):
            # initial guess are not supposed to sample a specific distribution,
            # it's enough to generate some points in the region of the parameter
            # space defined as the convex hull of user-provided samples
            n_sample = self.initial_guesses.shape[0]
            a = self.initial_guesses[np.random.choice(n_sample), :]
            b = self.initial_guesses[np.random.choice(n_sample), :]
            guess = a + np.random.random() * (b - a)
            m = Model.unpack(guess, layout_info=self.model)
            if np.isfinite(logposterior(m, data, self.model)):
                return m
        else:
            raise ValueError(f"Failed to generate valid model in {n_try} tries")


def print_delim():
    print("\n" + "=" * 15 + "\n" + datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))


@dataclasses.dataclass
class GoodnessOfFit:
    max_logpost: float
    loglike_at_map: float
    ndim: int
    aic: float


def run_ml_analysis(
    config: FitConfig,
    fit_data: Data,
    freeze_shifts: bool,
    initial_model: Model | None = None,
) -> tuple[Model, GoodnessOfFit]:
    model_config = config.model
    initial_model = initial_model or config.generate_initial_guess(fit_data)
    if freeze_shifts:
        model_config = dataclasses.replace(model_config, shifted_experiments=[])
        initial_model = dataclasses.replace(
            initial_model,
            energy_shifts=ExperimentEnergyScaleShifts(dict()),
        )

    def to_minimize(v: np.ndarray) -> float:
        # technically it should be -loglikelihood, but as we're using mostly flat priors
        # plus gaussian priors (L2 regularization) for experimental scale shifts
        # so, let us use logposterior instead
        return -logposterior(v, fit_data, model_config)

    res = optimize.minimize(
        to_minimize,
        x0=initial_model.pack(),
        method="Nelder-Mead",
        options={
            "maxiter": 100_000,
        },
    )
    print(res)
    map_model = Model.unpack(res.x, layout_info=model_config)

    max_logpost = -res.fun
    gof = GoodnessOfFit(
        max_logpost=max_logpost,
        loglike_at_map=loglikelihood(map_model, fit_data, model_config),
        ndim=map_model.ndim(),
        # we're actually maximizing posterior not likelihood, so this is not strictly AIC
        # but as mentioned above, this shouldn't matter too much dues to our choice of mostly
        # trivial priors
        aic=2 * initial_model.ndim() - 2 * max_logpost,
    )
    print(f"Goodness of fit: {gof}")
    return map_model, gof


def run_bayesian_analysis(config: FitConfig, outdir: Path) -> None:
    print(f"Output dir: {outdir}")

    Path(outdir / "config-dump.json").write_text(config.model_dump_json(indent=2))

    def load_saved(path: Path) -> Model | None:
        if config.reuse_saved_models:
            return Model.load(path, layout_info=config.model)
        else:
            return None

    print_delim()
    print("Loading fit data...")
    fit_data = Data.load(config.fit_data_config)
    set_global_fit_data(fit_data)

    validation_data = Data.load(
        config.plots.validation_data_config
        or DataConfig(
            experiments_all_particle=[],
            experiments_elements=[],
            experiments_lnA=[],
            elements=[],
        )
    )

    scale = 2.8 if fit_data.E_max() > 2e6 else 2.6
    fit_data.plot(scale=scale, describe=True).savefig(outdir / "data.png")
    validation_data.plot(scale=scale, describe=True).savefig(outdir / "data-validation.png")

    print_delim()
    print("Initial guess model (example):")
    initial_guess = config.generate_initial_guess(fit_data)
    initial_guess.plot_spectra(fit_data, scale=scale, validation_data=validation_data).savefig(
        outdir / "initial_guess.png"
    )
    initial_guess.print_params()
    print(
        "Logposterior value:",
        logposterior(
            initial_guess,
            fit_data=fit_data,
            config=config.model,
        ),
    )

    print_delim()
    print("Running preliminary ML analysis...")
    mle_model_dump = outdir / "preliminary-ml.txt"

    if loaded := load_saved(mle_model_dump):
        mle_model = loaded
    else:
        mle_model, gof = run_ml_analysis(
            config=config,
            fit_data=fit_data,
            freeze_shifts=False,
            initial_model=None,
        )
        mle_model.save(mle_model_dump, header=[f"GoF: {gof}"])

    mle_model.print_params()
    mle_model.plot_spectra(fit_data, scale=scale, validation_data=validation_data).savefig(
        outdir / "preliminary-mle-result.png"
    )
    mle_model.plot_lnA(fit_data, validation_data=validation_data).savefig(
        outdir / "preliminary-mle-result-lnA.png"
    )

    if config.mcmc is None:
        print("Not running bayesian analysis, mcmc config is None")
        return

    print_delim()
    print("Running bayesian analysis...")
    print(f"MCMC config: {config.mcmc}")
    ndim = config.generate_initial_guess(fit_data).ndim()
    print(f"N dim = {ndim}")

    sample_path = outdir / "theta.txt"

    if config.mcmc.reuse_saved and sample_path.exists():
        print("Loading saved theta sample")
        theta_sample = np.loadtxt(sample_path)
        assert theta_sample.ndim == 2, "Saved theta sample has the wrong number of dimensions"
        assert theta_sample.shape[1] == ndim, "Saved theta sample has wrong dimensions"
    else:
        print("Sampling theta...")
        sampling_start = time.time()
        pool_ctx = (
            multiprocessing.Pool(processes=config.mcmc.processes)
            if config.mcmc.processes > 1
            else contextlib.nullcontext(enter_result=None)
        )
        with pool_ctx as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers=config.mcmc.n_walkers,
                ndim=ndim,
                log_prob_fn=logposterior,
                args=(
                    # on macos passing fit data through a global variable doesn't work
                    # probably because it's using "spawn" multiprocessing method
                    fit_data if sys.platform == "darwin" else None,
                    config.model,
                ),
                pool=pool,
            )
            initial_state = np.array(
                [
                    config.generate_initial_guess(fit_data).pack()
                    for _ in range(config.mcmc.n_walkers)
                ]
            )
            sampler.run_mcmc(initial_state, nsteps=config.mcmc.n_steps, progress=not IS_CLUSTER)

        print("Sampling done")
        print(f"Acceptance fraction: {sampler.acceptance_fraction.mean()}")
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"{tau = }")

        burn_in = 5 * int(tau.max())
        thin = 2 * int(tau.max())

        print(f"Burn in: {burn_in}; Thinning: {thin}")

        theta_sample: np.ndarray = sampler.get_chain(flat=True, discard=burn_in, thin=thin)  # type: ignore

        sampling_time_sec = time.time() - sampling_start
        sampling_time_msg = (
            f"Sampling time: {sampling_time_sec / 60:.0f} min ~ {sampling_time_sec / 3600:.1f} hrs"
        )
        print(sampling_time_msg)

        model_example = Model.unpack(theta_sample[0, :], layout_info=config.model)
        tau_labels = model_example.labels(latex=False)
        np.savetxt(
            sample_path,
            theta_sample,
            header="\n".join(
                [
                    f"Generated on: {datetime.datetime.now()}",
                    sampling_time_msg,
                    f"MCMC config: {config.mcmc}",
                    f"Estimated autocorrelation lengths: {', '.join(f'{label}: {t}' for label, t in zip(tau_labels, tau))}",
                    f"Burn-in, steps: {burn_in}",
                    f"Thinning, steps: {thin}",
                    f"Sample shape: {theta_sample.shape}",
                ]
            ),
        )

    print(f"MCMC sample ready, shape: {theta_sample.shape}")
    median_model = Model.unpack(np.median(theta_sample, axis=0), layout_info=config.model)
    print("Median model:")
    median_model.print_params()

    model_sample = [Model.unpack(theta, layout_info=config.model) for theta in theta_sample]
    loglike_values = [loglikelihood(model, fit_data, config=config.model) for model in model_sample]
    best_fit_idx = np.argmax(loglike_values)
    print(f"Best-fitting model idx: {best_fit_idx}; loglike = {loglike_values[best_fit_idx]}")
    posterior_best_model = model_sample[best_fit_idx]

    print_delim()
    print("Plotting corner plot of the posterior")
    sample_to_plot = theta_sample
    sample_labels = ["$" + label + "$" for label in initial_guess.labels(latex=True)]
    fig_corner: Figure = corner.corner(
        sample_to_plot,
        labels=sample_labels,
        show_titles=True,
        quantiles=[0.05, 0.5, 0.95],
    )
    fig_corner.savefig(outdir / "corner.png")

    print_delim()
    print("Plotting best-fitting model from the posterior sample")

    posterior_best_model.plot_spectra(
        fit_data, scale=scale, validation_data=validation_data
    ).savefig(outdir / "best-fitting-posterior-point.png")
    posterior_best_model.plot_abundances().savefig(outdir / "abundances.png")

    print_delim()
    print("Running ML analysis from the best-fitting posterior point")
    posterior_ml_dump = outdir / "posterior-ml.txt"
    if loaded := load_saved(posterior_ml_dump):
        posterior_ml_best = loaded
    else:
        posterior_ml_best, gof = run_ml_analysis(
            config=config,
            fit_data=fit_data,
            freeze_shifts=False,
            initial_model=posterior_best_model,
        )
        posterior_ml_best.save(posterior_ml_dump, header=[f"GoF: {gof}"])

    posterior_ml_best.print_params()
    posterior_ml_best.plot_spectra(fit_data, scale=scale, validation_data=validation_data).savefig(
        outdir / "mle-from-posterior-best.png"
    )
    posterior_ml_best.plot_lnA(fit_data, validation_data=validation_data).savefig(
        outdir / "mle-from-posterior-best-lnA.png"
    )

    print_delim()
    print("Plotting final model plot")

    best_fit_model = posterior_ml_best or posterior_best_model

    fig, axes = plt.subplot_mosaic(
        [
            ["Elements", "Elements"],
            ["All particle", "lnA"],
            ["Shifts", "Shifts"],
        ],
        figsize=(8, 10),
        height_ratios=[1, 1, 0.3],
    )
    fig = cast(Figure, fig)
    axes = cast(dict[str, Axes], axes)
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
        scale_ = scale_override if scale_override is not None else scale
        if plot_config.contours:
            plot_posterior_contours(
                ax,
                scale=scale_,
                theta_sample=theta_sample,
                model_config=config.model,
                observable=observable,
                bounds=E_bounds,
                tricontourf_kwargs=tricontourf_kwargs_transparent_colors(
                    color=color,
                    alpha_max=0.5,
                ),
            )
        if plot_config.band_cl is not None:
            plot_credible_band(
                ax,
                scale=scale_,
                theta_sample=theta_sample,
                model_config=config.model,
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
                spec_data.plot(scale=scale, ax=ax_el, add_label=False, is_fitted=is_fitted)
            experiment_legend_item_by_label.setdefault(
                exp.name, (exp.legend_artist(True), exp.name)
            )
    elements = best_fit_model.layout_info().elements(only_fixed_Z=False)
    comp_data_ylim = merged_lims([sp.scaled_flux(scale) for sp in plotted_elem_spectra])
    comp_data_Elim = merged_lims([sp.E for sp in plotted_elem_spectra])
    comp_Elim = add_log_margin(*comp_data_Elim)
    for element in elements:
        plot_model_predictions(
            ax=ax_el,
            observable=lambda model, E: model.compute_spectrum(E, element=element),
            E_bounds=comp_Elim,
            plot_config=config.plots.elements,
            color=element.color,
        )
        element_legend_items.append((legend_artist_line(element.color), element.name))

    if config.plots.elements.population_contribs_best_fit and len(best_fit_model.populations) > 1:
        multipop_elements = [
            element
            for element in Element.regular()
            if len([pop for pop in best_fit_model.populations if element in pop.all_elements]) > 1
        ]
        E_grid = np.geomspace(*comp_Elim, 300)
        E_factor = E_grid**scale
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

    if config.plots.elements.max_margin_around_data is not None:
        clamp_log_margin(ax_el, comp_data_ylim, config.plots.elements.max_margin_around_data)
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
                    scale=scale,
                    ax=ax_all,
                    add_label=False,
                    is_fitted=is_fitted,
                )
                experiment_legend_item_by_label.setdefault(
                    exp.name, (exp.legend_artist(True), exp.name)
                )

        all_data_ylim = merged_lims([sp.scaled_flux(scale) for sp in plotted_all_spectra])
        all_data_Elim = merged_lims([sp.E for sp in plotted_all_spectra])
        all_Elim = add_log_margin(*all_data_Elim)
        ALL_PARTICLE_COLOR = "black"
        plot_model_predictions(
            ax=ax_all,
            observable=lambda model, E: model.compute_spectrum(E, element=None),
            E_bounds=all_Elim,
            plot_config=config.plots.all_particle,
            color=ALL_PARTICLE_COLOR,
        )
        element_legend_items.append((legend_artist_line(ALL_PARTICLE_COLOR), "All particle"))

        if config.plots.all_particle_elements_contribution is not None:
            for element in elements:
                plot_model_predictions(
                    ax=ax_all,
                    observable=lambda model, E: model.compute_spectrum(E, element=element),
                    E_bounds=all_Elim,
                    plot_config=config.plots.all_particle_elements_contribution,
                    color=element.color,
                )

        if config.plots.all_particle_scaled_elements_contribution and any(
            pop_conf.rescale_all_particle
            or any(comp.scale_contrib_to_allpart for comp in pop_conf.component_configs)
            for pop_conf in config.model.population_configs
        ):
            plot_model_predictions(
                ax=ax_all,
                observable=lambda model, E: sum(
                    (pop.compute_extra_all_particle_contribution(E) for pop in model.populations),
                    np.zeros_like(E),
                ),
                E_bounds=all_Elim,
                plot_config=config.plots.all_particle_scaled_elements_contribution,
                color="gray",
            )
            element_legend_items.append((legend_artist_line("gray"), "Extra contribution"))

        if config.plots.all_particle_unresolved_elements_contribution and any(
            pop_conf.add_unresolved_elements for pop_conf in config.model.population_configs
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
                plot_config=config.plots.all_particle_unresolved_elements_contribution,
                color="magenta",
            )
            element_legend_items.append((legend_artist_line("magenta"), "Unresolved elements"))

        if (
            config.plots.all_particle.population_contribs_best_fit
            and len(best_fit_model.populations) > 1
        ):
            E_grid = np.geomspace(*all_Elim, 300)
            E_factor = E_grid**scale
            for pop in best_fit_model.populations:
                ax_all.plot(
                    E_grid,
                    E_factor * pop.compute_all_particle(E_grid),
                    color=ALL_PARTICLE_COLOR,
                    linewidth=POP_CONTRIB_LINEWIDTH,
                    linestyle=pop.linestyle,
                )

        if config.plots.all_particle.max_margin_around_data is not None:
            clamp_log_margin(
                ax_all, all_data_ylim, config.plots.all_particle.max_margin_around_data
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
                    add_label=False,
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
            plot_config=config.plots.lnA,
            color=LN_A_COLOR,
            scale_override=0,
        )
        ax_lnA.set_xlabel(E_GEV_LABEL)
        ax_lnA.set_ylabel("$ \\langle \\ln A \\rangle $")
        add_elements_lnA_secondary_axis(ax_lnA)
        if config.plots.lnA.max_margin_around_data is not None:
            clamp_log_margin(ax_lnA, lnA_data_ylim, config.plots.lnA.max_margin_around_data)
        ax_lnA.set_xlim(*lnA_Elim)

    # experimental energy scale shifts
    exp_indices = np.arange(len(config.model.shifted_experiments))
    SHIFTS_COLOR = "black"
    for i, exp in enumerate(config.model.shifted_experiments):
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
        if config.plots.energy_shifts.contours:
            plot_posterior_contours(
                ax=ax_shifts,
                scale=0,
                theta_sample=theta_sample,
                model_config=config.model,
                observable=observable,
                bounds=bounds,
                tricontourf_kwargs=tricontourf_kwargs_transparent_colors(
                    color=SHIFTS_COLOR,
                    alpha_max=0.5,
                ),
            )
        if config.plots.energy_shifts.band_cl is not None:
            plot_credible_band(
                ax=ax_shifts,
                scale=0,
                theta_sample=theta_sample,
                model_config=config.model,
                observable=observable,
                bounds=bounds,
                color=SHIFTS_COLOR,
                alpha=0.2,
                cl=config.plots.energy_shifts.band_cl,
                grid_override=np.array(bounds),
            )
        if config.plots.energy_shifts.best_fit:
            ax_shifts.plot(bounds, observable(best_fit_model, np.array(bounds)), color=SHIFTS_COLOR)

    ax_shifts.axhline(0, linestyle="--", color="gray")
    ax_shifts.set_ylabel("$ \\delta E $ / %")
    ax_shifts.yaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax_shifts.set_xticks(
        exp_indices,
        [exp.name for exp in config.model.shifted_experiments],
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
        config.plots.elements.population_contribs_best_fit
        or config.plots.all_particle.population_contribs_best_fit
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
        ncol=4,
    )

    fig.tight_layout()
    fig.savefig(outdir / "model.pdf")
    if config.plots.export_opts.main is not None:
        export_fig(fig, filename=config.plots.export_opts.main)
    fig.savefig(outdir / "model.png")


if __name__ == "__main__":
    # CLI for cluster run; use run_local.py wrapper script to run the analysis locally
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Configuration JSON file; output files will be placed in the same directory",
    )
    args = parser.parse_args()
    config_path: Path = args.config
    print(f"Reading fit config from {config_path}")

    fit_config = FitConfig.model_validate_json(config_path.read_text())
    run_bayesian_analysis(fit_config, outdir=config_path.parent)
