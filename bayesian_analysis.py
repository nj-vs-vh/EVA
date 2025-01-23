import contextlib
from matplotlib.axes import Axes
import pydantic
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
    RigidityBreakConfig,
    SharedPowerLaw,
)
from cr_knee_fit.experiments import Experiment
from cr_knee_fit.fit_data import FitData
from cr_knee_fit.inference import loglikelihood, logposterior, set_global_fit_data
from cr_knee_fit.model import Model, ModelConfig
from cr_knee_fit.plotting import plot_credible_band
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts
from cr_knee_fit.types_ import Primary
from cr_knee_fit.utils import add_log_margin

# as recommended by emceee parallelization guide
# see https://emcee.readthedocs.io/en/stable/tutorials/parallel/#parallelization
os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.INFO)

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(exist_ok=True)

E_SCALE = 2.6  # for plots only


def initial_guess_break(bc: RigidityBreakConfig, break_idx: int) -> RigidityBreak:
    if bc.fixed_lg_sharpness:
        lg_s = bc.fixed_lg_sharpness
    else:
        lg_s = stats.norm.rvs(loc=np.log10(5), scale=0.01)

    break_pos_guesses = [4.2, 5.3, 6.5]
    d_alpha_guesses = [0.3, -0.3, 0.5]

    return RigidityBreak(
        lg_R=stats.norm.rvs(loc=break_pos_guesses[break_idx], scale=0.1),
        d_alpha=stats.norm.rvs(loc=d_alpha_guesses[break_idx], scale=0.05),
        lg_sharpness=lg_s,
        fix_sharpness=bc.fixed_lg_sharpness is not None,
    )


initial_guess_lgI = {
    Primary.H: -4,
    Primary.He: -4.65,
    Primary.C: -6.15,
    Primary.O: -6.1,
    Primary.Mg: -6.85,
    Primary.Si: -6.9,
    Primary.Fe: -6.9,
}
initial_guess_alpha = {
    Primary.H: 2.6,
    Primary.He: 2.5,
}


def initial_guess_model(config: ModelConfig) -> Model:
    return Model(
        cr_model=CosmicRaysModel(
            base_spectra=[
                (
                    SharedPowerLaw(
                        lgI_per_primary={
                            primary: stats.norm.rvs(loc=initial_guess_lgI[primary], scale=0.05)
                            for primary in component
                        },
                        alpha=stats.norm.rvs(
                            loc=2.6 if component == [Primary.H] else 2.5,
                            scale=0.05,
                        ),
                    )
                )
                for component in config.cr_model_config.components
            ],
            breaks=[
                initial_guess_break(bc, break_idx=i)
                for i, bc in enumerate(config.cr_model_config.breaks)
            ],
            all_particle_lg_shift=(
                np.log10(stats.uniform.rvs(loc=1.1, scale=1.3))
                if config.cr_model_config.rescale_all_particle
                else None
            ),
        ),
        energy_shifts=ExperimentEnergyScaleShifts(
            lg_shifts={exp: stats.norm.rvs(loc=0, scale=0.01) for exp in config.shifted_experiments}
        ),
    )


def safe_initial_guess_model(config: ModelConfig, fit_data: FitData) -> Model:
    n_try = 1000
    for _ in range(n_try):
        m = initial_guess_model(config)
        if np.isfinite(logposterior(m, fit_data, config)):
            return m
    else:
        raise ValueError(f"Failed to generate valid model in {n_try} tries")


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
    model: ModelConfig
    mcmc: McmcConfig


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
        primaries=config.model.cr_model_config.primaries,
        R_bounds=(7e2, 1e8),
    )
    set_global_fit_data(fit_data)
    return fit_data


def main(config: FitConfig) -> None:
    outdir = OUT_DIR / config.name
    outdir.mkdir(exist_ok=True)

    logfile = outdir / "log.txt"
    with logfile.open("w") as log, contextlib.redirect_stdout(new_target=log):
        print(f"Output dir: {outdir}")

        config_path = Path(outdir / "config.json")
        if config_path.exists():
            previous_config = FitConfig.model_validate_json(config_path.read_text())
            is_config_updated = config != previous_config
            print(f"Found previous config in output dir, is updated = {is_config_updated}")
        else:
            is_config_updated = True
        config_path.write_text(config.model_dump_json(indent=2))

        print_delim()
        print("Loading fit data...")
        fit_data = load_fit_data(config)
        fig, ax = plt.subplots()
        print("Data by primary:")
        for exp, ps in fit_data.spectra.items():
            print(exp.name)
            for p, s in ps.items():
                print(f"  {p.name}: {s.E.size} points from {s.E.min():.1e} to {s.E.max():.1e} GeV")
                s.plot(scale=E_SCALE, ax=ax)
        print("All particle data:")
        for exp, s in fit_data.all_particle_spectra.items():
            print(f"{exp.name}: {s.E.size} points from {s.E.min():.1e} to {s.E.max():.1e} GeV")
            s.plot(scale=E_SCALE, ax=ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize="xx-small")
        fig.savefig(outdir / "data.png")

        print_delim()
        print("Initial guess model (example):")
        initial_guess = safe_initial_guess_model(config.model, fit_data)
        initial_guess.plot(fit_data, scale=E_SCALE).savefig(outdir / "initial_guess.png")
        initial_guess.print_params()
        print(
            "Logposterior value:",
            logposterior(
                initial_guess,
                fit_data=fit_data,
                config=config.model,
            ),
        )

        # safe_initial_guess_model(config.model, fit_data).plot(fit_data, scale=E_SCALE).savefig(outdir / "test.png")

        print_delim()
        print("Running preliminary MLE analysis...")
        mle_config = dataclasses.replace(config.model, shifted_experiments={})

        def negloglike(v: np.ndarray) -> float:
            return -loglikelihood(v, fit_data, mle_config)

        res = optimize.minimize(
            negloglike,
            x0=safe_initial_guess_model(mle_config, fit_data).pack(),
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
        print(f"MCMC config: {config.mcmc}")
        ndim = safe_initial_guess_model(config.model, fit_data).ndim()
        print(f"N dim = {ndim}")

        sample_path = outdir / "theta.txt"

        if config.mcmc.reuse_saved and not is_config_updated and sample_path.exists():
            print("Loading saved theta sample")
            theta_sample = np.loadtxt(sample_path)
            assert theta_sample.ndim == 2
            assert theta_sample.shape[1] == ndim
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
                    args=(None, config.model),
                    pool=pool,
                )
                initial_state = np.array(
                    [
                        safe_initial_guess_model(config.model, fit_data).pack()
                        for _ in range(config.mcmc.n_walkers)
                    ]
                )
                sampler.run_mcmc(
                    initial_state,
                    nsteps=config.mcmc.n_steps,
                    progress=True,
                )

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

        print_delim()
        print("Plotting posterior")
        sample_to_plot = theta_sample
        sample_labels = [
            "$" + label + "$" for label in initial_guess_model(config.model).labels(latex=True)
        ]
        fig: Figure = corner.corner(
            sample_to_plot,
            labels=sample_labels,
            show_titles=True,
            quantiles=[0.05, 0.5, 0.95],
        )
        fig.savefig(outdir / "corner.png")

        print_delim()
        print("Plotting primary fluxes with credible bands")
        fig, axes = plt.subplots(figsize=(15, 7), ncols=2)

        median_model = Model.unpack(np.median(theta_sample, axis=0), layout_info=config.model)
        print("Median model:")
        median_model.print_params()

        ax_comp: Axes = axes[0]
        ax_comp.set_title("Composition")
        for exp, data_by_particle in fit_data.spectra.items():
            for _, spectrum in data_by_particle.items():
                spectrum.with_shifted_energy_scale(f=median_model.energy_shifts.f(exp)).plot(
                    scale=E_SCALE,
                    ax=ax_comp,
                    add_label=False,
                )
        primaries = median_model.cr_model.layout_info().primaries
        Emin, Emax = add_log_margin(fit_data.E_min(), fit_data.E_max())
        for p in primaries:
            plot_credible_band(
                ax_comp,
                scale=E_SCALE,
                theta_sample=theta_sample,
                model_config=config.model,
                observable=lambda model, E: model.cr_model.compute(E, p),
                color=p.color,
                bounds=(Emin, Emax),
                add_median=True,
                label=p.name,
            )
        handles, labels = ax_comp.get_legend_handles_labels()
        exps = sorted(fit_data.spectra.keys(), key=lambda e: e.name)
        handles.extend([exp.legend_handle() for exp in exps])
        labels.extend([exp.name for exp in exps])
        ax_comp.legend(handles, labels, fontsize="xx-small")

        ax_all: Axes = axes[1]
        ax_all.set_title("All particle")
        for exp, spectrum in fit_data.all_particle_spectra.items():
            spectrum.with_shifted_energy_scale(f=median_model.energy_shifts.f(exp)).plot(
                scale=E_SCALE,
                ax=ax_all,
                add_label=False,
            )
        plot_credible_band(
            ax_all,
            scale=E_SCALE,
            theta_sample=theta_sample,
            model_config=config.model,
            observable=lambda model, E: model.cr_model.compute_all_particle(E),
            color="red",
            bounds=(Emin, Emax),
            add_median=True,
            label="Sum of primaries $\\times$ K"
        )
        handles, labels = ax_all.get_legend_handles_labels()
        exps = sorted(fit_data.all_particle_spectra.keys(), key=lambda e: e.name)
        handles.extend([exp.legend_handle() for exp in exps])
        labels.extend([exp.name for exp in exps])
        ax_all.legend(handles, labels, fontsize="xx-small")

        for ax in axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(Emin, Emax)

        fig.savefig(outdir / "fluxes.pdf")


if __name__ == "__main__":
    from cr_knee_fit import experiments

    experiments_detailed = experiments.direct_experiments + [experiments.grapes]
    experiments_all_particle = [experiments.hawc, experiments.dampe]
    # experiments_all_particle = []

    main(
        FitConfig(
            name="below-knee",
            experiments_detailed=experiments_detailed,
            experiments_all_particle=experiments_all_particle,
            model=ModelConfig(
                cr_model_config=CosmicRaysModelConfig(
                    components=[
                        [Primary.H],
                        [Primary.He],
                        sorted(p for p in Primary if p not in {Primary.H, Primary.He}),
                    ],
                    breaks=[
                        RigidityBreakConfig(fixed_lg_sharpness=np.log10(5)),
                        RigidityBreakConfig(fixed_lg_sharpness=np.log10(10)),
                        # RigidityBreakConfig(fixed_lg_sharpness=np.log10(5)),
                    ],
                    rescale_all_particle=True,
                ),
                shifted_experiments=[e for e in experiments_detailed if e != experiments.ams02],
            ),
            mcmc=McmcConfig(
                n_steps=30_000,
                n_walkers=256,
                processes=8,
                reuse_saved=True,
            ),
        )
    )
