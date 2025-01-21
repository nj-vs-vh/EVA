import contextlib
import dataclasses
import datetime
import logging
import multiprocessing
import os
from pathlib import Path

import corner
import emcee
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy import optimize, stats

from cr_knee_fit.cr_model import (
    CosmicRaysModel,
    CosmicRaysModelConfig,
    RigidityBreak,
    SharedPowerLaw,
)
from cr_knee_fit.fit_data import FitData
from cr_knee_fit.inference import loglikelihood, logposterior
from cr_knee_fit.model import Model, ModelConfig
from cr_knee_fit.plotting import plot_credible_band
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts
from cr_knee_fit.types_ import Experiment, Primary
from cr_knee_fit.utils import add_log_margin

# as recommended by emceee parallelization guide https://emcee.readthedocs.io/en/stable/tutorials/parallel/#parallelization
os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.INFO)

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(exist_ok=True)

E_SCALE = 2.6


def initial_guess_model(config: ModelConfig) -> Model:
    return Model(
        cr_model=CosmicRaysModel(
            base_spectra=[
                (
                    SharedPowerLaw(
                        lgI_per_primary={
                            primary: stats.norm.rvs(loc=-4, scale=0.5) - 2.6 * np.log10(primary.Z)
                            for primary in component
                        },
                        alpha=stats.norm.rvs(loc=2.7, scale=0.1),
                    )
                )
                for component in config.cr_model_config.components
            ],
            breaks=[
                # dampe softening
                RigidityBreak(
                    lg_R=stats.norm.rvs(loc=4.2, scale=0.1),
                    d_alpha=stats.norm.rvs(loc=0.3, scale=0.05),
                    lg_sharpness=stats.norm.rvs(loc=np.log10(5), scale=0.01),
                ),
                # grapes hardening
                RigidityBreak(
                    lg_R=stats.norm.rvs(loc=5.3, scale=0.1),
                    d_alpha=stats.norm.rvs(loc=-0.3, scale=0.05),
                    lg_sharpness=stats.norm.rvs(loc=np.log10(10), scale=0.01),
                ),
                # knee
                RigidityBreak(
                    lg_R=stats.norm.rvs(loc=6.5, scale=0.2),
                    d_alpha=stats.norm.rvs(loc=0.5, scale=0.05),
                    lg_sharpness=stats.norm.rvs(loc=np.log10(5), scale=0.01),
                ),
            ][: config.cr_model_config.n_breaks],
            all_particle_lg_shift=(
                stats.norm.rvs(scale=0.2) if config.cr_model_config.rescale_all_particle else None
            ),
        ),
        energy_shifts=ExperimentEnergyScaleShifts(
            lg_shifts={exp: stats.norm.rvs(loc=0, scale=0.01) for exp in config.shifted_experiments}
        ),
    )


@dataclasses.dataclass
class McmcConfig:
    n_steps: int
    n_walkers: int
    processes: int


def print_delim():
    print(
        "\n\n"
        + "=" * 15
        + "\n"
        + datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
        + "\n\n"
    )


def main(
    name: str,
    experiments: list[Experiment],
    config: ModelConfig,
    mcmc_config: McmcConfig,
) -> None:
    outdir = OUT_DIR / name
    outdir.mkdir(exist_ok=True)

    logfile = outdir / "log.txt"
    with logfile.open("w") as log, contextlib.redirect_stdout(log):
        print(f"Output dir: {outdir}")
        print_delim()
        print("Loading fit data...")
        fit_data = FitData.load(
            experiments_detailed=[e for e in experiments if e.available_primaries()],
            experiments_all_particle=[e for e in experiments if not e.available_primaries()],
            primaries=config.cr_model_config.primaries,
            R_bounds=(7e2, 1e8),
        )

        print("Data by primary:")
        for exp, ps in fit_data.spectra.items():
            print(exp)
            for p, s in ps.items():
                print(f"  {p.name}: {s.E.size} points from {s.E.min():.1e} to {s.E.max():.1e} GeV")

        print("All particle data:")
        for exp, s in fit_data.all_particle_spectra.items():
            print(f"{exp}: {s.E.size} points from {s.E.min():.1e} to {s.E.max():.1e} GeV")

        print_delim()
        print("Initial guess model (example):")
        initial_guess_model(config).print_params()

        print_delim()
        print("Running preliminary MLE analysis...")
        mle_config = dataclasses.replace(config, shifted_experiments={})

        def negloglike(v: np.ndarray) -> float:
            return -loglikelihood(v, fit_data, mle_config)

        res = optimize.minimize(
            negloglike,
            x0=initial_guess_model(mle_config).pack(),
            method="Nelder-Mead",
            options={
                "maxiter": 100_000,
            },
        )
        print(res)
        mle_model = Model.unpack(res.x, layout_info=mle_config)
        fig = mle_model.plot(fit_data, scale=E_SCALE)
        fig.savefig(outdir / "mle-result.png")
        mle_model.print_params()

        print_delim()
        print("Running bayesian analysis...")
        print(f"MCMC config: {mcmc_config}")
        ndim = initial_guess_model(config).ndim()
        print(f"N dim = {ndim}")

        pool_ctx = (
            multiprocessing.Pool(processes=mcmc_config.processes)
            if mcmc_config.processes > 1
            else contextlib.nullcontext(enter_result=None)
        )
        with pool_ctx as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers=mcmc_config.n_walkers,
                ndim=ndim,
                log_prob_fn=logposterior,
                args=(fit_data, config),
                pool=pool,
            )
            initial_state = np.array(
                [initial_guess_model(config).pack() for _ in range(mcmc_config.n_walkers)]
            )
            sampler.run_mcmc(
                initial_state,
                nsteps=mcmc_config.n_steps,
                progress=True,
            )
            print(f"Acceptance fraction: {sampler.acceptance_fraction.mean()}")

            tau = sampler.get_autocorr_time(quiet=True)
            print(f"{tau = }")

            burn_in = 5 * int(tau.max())
            thin = 2 * int(tau.max())

            print(f"Burn in: {burn_in}; Thinning: {thin}")

            theta_sample: np.ndarray = sampler.get_chain(flat=True, discard=burn_in, thin=thin)  # type: ignore
            print(f"MCMC sample ready, shape: {theta_sample.shape}")

        np.savetxt(outdir / "theta.txt", theta_sample)

        print_delim()
        print("Plotting posterior")
        sample_to_plot = theta_sample
        sample_labels = [
            "$" + label + "$" for label in initial_guess_model(config).labels(latex=True)
        ]
        fig: Figure = corner.corner(
            sample_to_plot,
            labels=sample_labels,
            show_titles=True,
            quantiles=[0.05, 0.5, 0.95],
        )
        fig.savefig(outdir / "corner.pdf")

        print_delim()
        print("Plotting primary fluxes with credible bands")
        fig, ax = plt.subplots(figsize=(10, 8))

        model_sample = [Model.unpack(theta, layout_info=config) for theta in theta_sample]
        median_model = Model.unpack(np.median(theta_sample, axis=0), layout_info=config)
        mle_model.print_params()

        for exp, data_by_particle in fit_data.spectra.items():
            for _, spectrum in data_by_particle.items():
                spectrum.with_shifted_energy_scale(f=median_model.energy_shifts.f(exp)).plot(
                    scale=E_SCALE, ax=ax
                )

        for exp, spectrum in fit_data.all_particle_spectra.items():
            spectrum.with_shifted_energy_scale(f=median_model.energy_shifts.f(exp)).plot(
                scale=E_SCALE, ax=ax
            )

        primaries = median_model.cr_model.layout_info().primaries
        Emin, Emax = add_log_margin(fit_data.E_min(), fit_data.E_max())
        median_model.cr_model.plot(Emin, Emax, scale=E_SCALE, axes=ax)

        for p in primaries:
            plot_credible_band(
                ax,
                scale=E_SCALE,
                model_sample=model_sample,
                observable=lambda model, E: model.cr_model.compute(E, p),
                color=p.color,
                E_bounds=(Emin, Emax),
            )

        ax.legend(fontsize="xx-small")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(Emin, Emax)
        fig.savefig(outdir / "fluxes.pdf")


if __name__ == "__main__":
    # experiments = [e for e in Experiment if e.available_primaries()]
    experiments = [Experiment.DAMPE, Experiment.AMS02]
    main(
        name="pre-knee-composition",
        experiments=experiments,
        config=ModelConfig(
            cr_model_config=CosmicRaysModelConfig(
                components=[
                    [Primary.H],
                    [Primary.He],
                    # sorted(p for p in Primary if p not in {Primary.H, Primary.He}),
                ],
                n_breaks=1,
                rescale_all_particle=False,
            ),
            shifted_experiments=[e for e in experiments if e is not Experiment.AMS02],
        ),
        mcmc_config=McmcConfig(
            n_steps=10_000,
            n_walkers=128,
            processes=1,
        ),
    )
