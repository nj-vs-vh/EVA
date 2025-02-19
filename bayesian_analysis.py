import argparse
import contextlib
import dataclasses
import datetime
import itertools
import multiprocessing
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Callable, Sequence, cast

import corner  # type: ignore
import emcee  # type: ignore
import numpy as np
import pydantic
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic_numpy.typing import Np2DArrayFp64
from scipy import optimize  # type: ignore

from cr_knee_fit.experiments import Experiment
from cr_knee_fit.fit_data import FitData
from cr_knee_fit.inference import loglikelihood, logposterior, set_global_fit_data
from cr_knee_fit.model_ import Model, ModelConfig
from cr_knee_fit.plotting import (
    plot_posterior_contours,
    tricontourf_kwargs_transparent_colors,
)
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts
from cr_knee_fit.utils import E_GEV_LABEL, add_log_margin, legend_with_added_items

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


class FitConfig(pydantic.BaseModel):
    name: str
    experiments_detailed: list[Experiment]
    experiments_all_particle: list[Experiment]
    experiments_lnA: list[Experiment]
    mcmc: McmcConfig | None

    model: ModelConfig
    initial_guesses: Np2DArrayFp64  # n_sample x n_model_dim

    @classmethod
    def from_guessing_func(
        cls,
        name: str,
        experiments_detailed: list[Experiment],
        experiments_all_particle: list[Experiment],
        experiments_lnA: list[Experiment],
        mcmc: McmcConfig | None,
        generate_guess: Callable[[], Model],
        n_guesses: int = 100,
    ) -> "FitConfig":
        guesses = [generate_guess() for _ in range(n_guesses)]
        assert (
            len({g.ndim() for g in guesses}) == 1
        ), "guess generation function generates different-dimensional models"

        return FitConfig(
            name=name,
            experiments_detailed=experiments_detailed,
            experiments_all_particle=experiments_all_particle,
            experiments_lnA=experiments_lnA,
            mcmc=mcmc,
            model=guesses[0].layout_info(),
            initial_guesses=np.array([guess.pack() for guess in guesses]),
        )

    def generate_initial_guess(self, fit_data: FitData) -> Model:
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
            if np.isfinite(logposterior(m, fit_data, self.model)):
                return m
        else:
            raise ValueError(f"Failed to generate valid model in {n_try} tries")


def print_delim():
    print(
        "\n\n"
        + "=" * 15
        + "\n"
        + datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
        + "\n\n"
    )


def load_fit_data(config: FitConfig) -> FitData:
    fit_data = FitData.load(
        experiments_detailed=config.experiments_detailed,
        experiments_all_particle=config.experiments_all_particle,
        experiments_lnA=config.experiments_lnA,
        primaries=config.model.primaries(),
        R_bounds=(7e2, 1e8),
    )
    set_global_fit_data(fit_data)
    return fit_data


def run_bayesian_analysis(config: FitConfig, outdir: Path) -> None:
    print(f"Output dir: {outdir}")

    Path(outdir / "config-dump.json").write_text(config.model_dump_json(indent=2))

    print_delim()
    print("Loading fit data...")
    fit_data = load_fit_data(config)
    scale = 2.8 if fit_data.E_max() > 2e6 else 2.6
    fit_data.plot(scale=scale, describe=True).savefig(outdir / "data.png")

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
    print("Running preliminary MLE analysis...")
    try:
        mle_config = dataclasses.replace(config.model, shifted_experiments=[])
        initial_mle_model = dataclasses.replace(
            config.generate_initial_guess(fit_data),
            energy_shifts=ExperimentEnergyScaleShifts(dict()),
        )

        def negloglike(v: np.ndarray) -> float:
            return -loglikelihood(v, fit_data, mle_config)

        res = optimize.minimize(
            negloglike,
            x0=initial_mle_model.pack(),
            method="Nelder-Mead",
            options={
                "maxiter": 100_000,
            },
        )
        print(res)
        mle_model = Model.unpack(res.x, layout_info=mle_config)
        fig = mle_model.plot(fit_data, scale=scale)
        fig.savefig(outdir / "mle-result.png")
        mle_model.print_params()
    except Exception as e:
        print(f"Error running MLE analysis, ignoring: {e}")
        traceback.print_exc()

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

    print_delim()
    print("Plotting posterior")
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
    print("Plotting posterior contours on model predictions")

    fig, axes = plt.subplots(figsize=(18, 6), ncols=3)
    axes = cast(Sequence[Axes], axes)

    ax_comp = axes[0]
    ax_all = axes[1]
    ax_lnA = axes[2]

    ax_comp.set_title("Composition")
    for exp, data_by_particle in fit_data.spectra.items():
        for _, spectrum in data_by_particle.items():
            spectrum.with_shifted_energy_scale(f=median_model.energy_shifts.f(exp)).plot(
                scale=scale,
                ax=ax_comp,
                add_label=False,
            )
    primaries = median_model.layout_info().primaries(observed_only=False)
    E_comp_all = np.hstack(
        [
            s.E
            for s in itertools.chain.from_iterable(ps.values() for ps in fit_data.spectra.values())
        ]
    )
    Emin, Emax = add_log_margin(E_comp_all.min(), E_comp_all.max())
    for primary in primaries:
        plot_posterior_contours(
            ax_comp,
            scale=scale,
            theta_sample=theta_sample,
            model_config=config.model,
            observable=lambda model, E: model.compute_spectrum(E, primary=primary),
            bounds=(Emin, Emax),
            tricontourf_kwargs=tricontourf_kwargs_transparent_colors(
                color=primary.color, levels=15
            ),
        )
    legend_with_added_items(
        ax_comp,
        (
            [(p.legend_artist(), p.name) for p in primaries]
            + [(exp.legend_artist(), exp.name) for exp in sorted(fit_data.spectra.keys())]
        ),
        fontsize="x-small",
    )

    if fit_data.all_particle_spectra:
        ax_all.set_title("All particle")
        E_all_all = np.hstack([s.E for s in fit_data.all_particle_spectra.values()])
        E_bounds_all = add_log_margin(E_all_all.min(), E_all_all.max())
        for exp, spectrum in fit_data.all_particle_spectra.items():
            spectrum.with_shifted_energy_scale(f=median_model.energy_shifts.f(exp)).plot(
                scale=scale,
                ax=ax_all,
                add_label=False,
            )

        plot_posterior_contours(
            ax_all,
            scale=scale,
            theta_sample=theta_sample,
            model_config=config.model,
            observable=lambda model, E: model.compute_spectrum(E, primary=None),
            bounds=E_bounds_all,
            tricontourf_kwargs={"levels": 15},
        )
        ylim = ax_all.get_ylim()
        for primary in primaries:
            plot_posterior_contours(
                ax_all,
                scale=scale,
                theta_sample=theta_sample,
                model_config=config.model,
                observable=lambda model, E: model.compute_spectrum(E, primary=primary),
                bounds=E_bounds_all,
                tricontourf_kwargs=tricontourf_kwargs_transparent_colors(
                    color=primary.color, levels=15
                ),
            )
        ax_all.set_ylim(*ylim)
        legend_with_added_items(
            ax_all,
            (
                [(p.legend_artist(), p.name) for p in primaries]
                + [
                    (exp.legend_artist(), exp.name)
                    for exp in (sorted(fit_data.all_particle_spectra.keys()))
                ]
            ),
            fontsize="x-small",
        )

    if fit_data.lnA:
        ax_lnA.set_title("$ \\langle \\ln A \\rangle $")
        for exp, data in fit_data.lnA.items():
            data = dataclasses.replace(data, x=data.x * median_model.energy_shifts.f(exp))
            data.plot(
                scale=0,
                ax=ax_lnA,
                add_label=False,
                color="black",
            )
        E_lnA_all = np.hstack([s.x for s in fit_data.lnA.values()])
        plot_posterior_contours(
            ax_lnA,
            scale=0,
            theta_sample=theta_sample,
            model_config=config.model,
            observable=lambda model, E: model.compute_lnA(E),
            bounds=add_log_margin(E_lnA_all.min(), E_lnA_all.max()),
        )
        legend_with_added_items(
            ax_lnA,
            [
                (exp.legend_artist(), exp.name)
                for exp in sorted(fit_data.lnA.keys(), key=lambda e: e.name)
            ],
            fontsize="x-small",
        )
        ax_lnA.set_xscale("log")
        ax_lnA.set_xlabel(E_GEV_LABEL)
        ax_lnA.set_ylabel("$ \\langle \\ln A \\rangle $")

    for ax in (ax_comp, ax_all):
        ax.set_xscale("log")
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(outdir / "posterior_contours.pdf")


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
