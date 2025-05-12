import argparse
import contextlib
import dataclasses
import datetime
import itertools
import multiprocessing
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Sequence, cast
from warnings import warn

import corner  # type: ignore
import emcee  # type: ignore
import numpy as np
import pydantic
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic_numpy.typing import Np2DArrayFp64  # type: ignore
from scipy import optimize  # type: ignore

from cr_knee_fit.elements import unresolved_element_names
from cr_knee_fit.fit_data import Data, DataConfig
from cr_knee_fit.inference import loglikelihood, logposterior, set_global_fit_data
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
    add_log_margin,
    energy_shift_suffix,
    legend_artist_line,
    legend_with_added_items,
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
    best_fit: bool
    contours: bool
    band_cl: float | None

    @classmethod
    def default(cls) -> "PosteriorPlotConfig":
        return PosteriorPlotConfig(best_fit=True, contours=False, band_cl=0.68)


@dataclasses.dataclass
class PlotsConfig:
    validation_data_config: DataConfig | None = None
    elements: PosteriorPlotConfig = PosteriorPlotConfig.default()
    all_particle: PosteriorPlotConfig = PosteriorPlotConfig.default()
    all_particle_elements_contribution: PosteriorPlotConfig | None = PosteriorPlotConfig.default()
    all_particle_scaled_elements_contribution: PosteriorPlotConfig | None = (
        PosteriorPlotConfig.default()
    )
    all_particle_unresolved_elements_contribution: PosteriorPlotConfig | None = (
        PosteriorPlotConfig.default()
    )
    lnA: PosteriorPlotConfig = PosteriorPlotConfig.default()


class FitConfig(pydantic.BaseModel):
    name: str
    fit_data_config: DataConfig
    mcmc: McmcConfig | None
    model: ModelConfig
    plots: PlotsConfig
    initial_guesses: Np2DArrayFp64  # n_sample x n_model_dim

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


def run_ml_analysis(
    config: FitConfig,
    fit_data: Data,
    freeze_shifts: bool,
    initial_model: Model | None = None,
) -> Model | None:
    try:
        model_config = config.model
        initial_model = initial_model or config.generate_initial_guess(fit_data)
        if freeze_shifts:
            model_config = dataclasses.replace(model_config, shifted_experiments=[])
            initial_model = dataclasses.replace(
                initial_model,
                energy_shifts=ExperimentEnergyScaleShifts(dict()),
            )

        def negloglike(v: np.ndarray) -> float:
            return -loglikelihood(v, fit_data, model_config)

        res = optimize.minimize(
            negloglike,
            x0=initial_model.pack(),
            method="Nelder-Mead",
            options={
                "maxiter": 100_000,
            },
        )
        print(res)
        return Model.unpack(res.x, layout_info=model_config)
    except Exception as e:
        print(f"Error running ML analysis, ignoring: {e}")
        traceback.print_exc()
        return None


def run_bayesian_analysis(config: FitConfig, outdir: Path) -> None:
    print(f"Output dir: {outdir}")

    Path(outdir / "config-dump.json").write_text(config.model_dump_json(indent=2))

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
    initial_guess.plot(fit_data, scale=scale).savefig(outdir / "initial_guess.png")
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
    mle_model = run_ml_analysis(
        config=config,
        fit_data=fit_data,
        freeze_shifts=True,
        initial_model=None,
    )
    if mle_model is not None:
        mle_model.plot(fit_data, scale=scale).savefig(outdir / "preliminary-mle-result.png")
        mle_model.print_params()

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
        np.savetxt(sample_path, theta_sample)

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
    print("Plotting model predictions over the data")

    best_fit_model = posterior_best_model  # ML?

    fig, axes = plt.subplots(figsize=(18, 6), ncols=3)
    axes = cast(Sequence[Axes], axes)

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
            )
        if plot_config.best_fit:
            E_grid = np.geomspace(*E_bounds, 100)
            E_factor = E_grid**scale_
            ax.plot(E_grid, E_factor * observable(best_fit_model, E_grid), color=color)

    ax_comp = axes[0]
    ax_all = axes[1]
    ax_lnA = axes[2]

    ax_comp.set_title("Elements")
    legend_items: list[LegendItem] = []
    for data, is_fitted in ((fit_data, True), (validation_data, False)):
        for exp, data_by_particle in data.element_spectra.items():
            f_exp = best_fit_model.energy_shifts.f(exp)
            for _, spec_data in data_by_particle.items():
                spec_data.with_shifted_energy_scale(f=f_exp).plot(
                    scale=scale, ax=ax_comp, add_label=False, is_fitted=is_fitted
                )
            legend_items.append(
                (exp.legend_artist(is_fitted=is_fitted), exp.name + energy_shift_suffix(f_exp))
            )

    elements = best_fit_model.layout_info().elements(only_fixed_Z=False)
    E_merged_comp = np.hstack(
        [
            spectrum.E
            for spectrum in itertools.chain.from_iterable(
                spectra_by_element.values()
                for spectra_by_element in itertools.chain(
                    fit_data.element_spectra.values(), validation_data.element_spectra.values()
                )
            )
        ]
    )
    for element in elements:
        plot_model_predictions(
            ax=ax_comp,
            observable=lambda model, E: model.compute_spectrum(E, element=element),
            E_bounds=add_log_margin(E_merged_comp.min(), E_merged_comp.max()),
            plot_config=config.plots.elements,
            color=element.color,
        )
        legend_items.append((element.legend_artist(), element.name))
    legend_with_added_items(ax_comp, legend_items, fontsize="x-small")

    if fit_data.all_particle_spectra or validation_data.all_particle_spectra:
        ax_all.set_title("All particle")
        legend_items = []

        for data, is_fitted in ((fit_data, True), (validation_data, False)):
            for exp, spec_data in data.all_particle_spectra.items():
                f_exp = best_fit_model.energy_shifts.f(exp)
                spec_data.with_shifted_energy_scale(f=f_exp).plot(
                    scale=scale,
                    ax=ax_all,
                    add_label=False,
                    is_fitted=is_fitted,
                )
                legend_items.append(
                    (exp.legend_artist(is_fitted=is_fitted), exp.name + energy_shift_suffix(f_exp))
                )

        E_merged_all = np.hstack(
            [
                spectrum.E
                for spectrum in itertools.chain(
                    fit_data.all_particle_spectra.values(),
                    validation_data.all_particle_spectra.values(),
                )
            ]
        )
        E_bounds_all = add_log_margin(E_merged_all.min(), E_merged_all.max())
        ALL_PARTICLE_COLOR = "black"
        plot_model_predictions(
            ax=ax_all,
            observable=lambda model, E: model.compute_spectrum(E, element=None),
            E_bounds=E_bounds_all,
            plot_config=config.plots.all_particle,
            color=ALL_PARTICLE_COLOR,
        )
        legend_items.append((legend_artist_line(ALL_PARTICLE_COLOR), "All particle"))

        if config.plots.all_particle_elements_contribution is not None:
            for element in elements:
                plot_model_predictions(
                    ax=ax_all,
                    observable=lambda model, E: model.compute_spectrum(E, element=element),
                    E_bounds=E_bounds_all,
                    plot_config=config.plots.all_particle_elements_contribution,
                    color=element.color,
                )
                legend_items.append((element.legend_artist(), element.name))

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
                E_bounds=E_bounds_all,
                plot_config=config.plots.all_particle_scaled_elements_contribution,
                color="gray",
            )
            legend_items.append((legend_artist_line("gray"), "Extra contribution"))

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
                E_bounds=E_bounds_all,
                plot_config=config.plots.all_particle_unresolved_elements_contribution,
                color="magenta",
            )
            legend_items.append((legend_artist_line("magenta"), "Unresolved elements"))

        legend_with_added_items(ax_all, legend_items, fontsize="x-small")

    if fit_data.lnA or validation_data.lnA:
        LN_A_COLOR = "red"
        ax_lnA.set_title("$ \\langle \\ln A \\rangle $")
        legend_items = []
        for data, is_fitted in ((fit_data, True), (validation_data, False)):
            for exp, lnA_data in data.lnA.items():
                f_exp = best_fit_model.energy_shifts.f(exp)
                lnA_data = dataclasses.replace(lnA_data, x=lnA_data.x * f_exp)
                lnA_data.plot(
                    scale=0,
                    ax=ax_lnA,
                    add_label=False,
                    color=LN_A_COLOR,
                    is_fitted=False,
                )
                legend_items.append(
                    (exp.legend_artist(is_fitted), exp.name + energy_shift_suffix(f_exp))
                )

        E_merged_lnA = np.hstack(
            [s.x for s in itertools.chain(fit_data.lnA.values(), validation_data.lnA.values())]
        )
        plot_model_predictions(
            ax=ax_lnA,
            observable=lambda model, E: model.compute_lnA(E),
            E_bounds=add_log_margin(E_merged_lnA.min(), E_merged_lnA.max()),
            plot_config=config.plots.lnA,
            color=LN_A_COLOR,
            scale_override=0,
        )
        legend_with_added_items(ax_lnA, legend_items, fontsize="x-small")
        ax_lnA.set_xscale("log")
        ax_lnA.set_xlabel(E_GEV_LABEL)
        ax_lnA.set_ylabel("$ \\langle \\ln A \\rangle $")

    for ax in (ax_comp, ax_all):
        ax.set_xscale("log")
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(outdir / "model.pdf")
    fig.savefig(outdir / "model.png")

    print_delim()
    print("Plotting best-fitting model from the posterior sample")

    posterior_best_model.plot(fit_data, scale=scale).savefig(
        outdir / "best-fitting-posterior-point.png"
    )
    posterior_best_model.plot_abundances().savefig(outdir / "abundances.png")

    print_delim()
    print("Running ML analysis from the best-fitting posterior point")
    posterior_ml_best = run_ml_analysis(
        config=config,
        fit_data=fit_data,
        freeze_shifts=True,
        initial_model=posterior_best_model,
    )
    if posterior_ml_best is not None:
        posterior_ml_best.print_params()
        posterior_ml_best.plot(fit_data, scale=scale).savefig(
            outdir / "mle-from-posterior-best.png"
        )


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
