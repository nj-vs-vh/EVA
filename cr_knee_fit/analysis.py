import argparse
import contextlib
import dataclasses
import datetime
import math
import multiprocessing
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from warnings import warn

import corner  # type: ignore
import emcee  # type: ignore
import iminuit  # type: ignore
import numpy as np
import pydantic
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pydantic_numpy.typing import Np2DArrayFp64  # type: ignore
from scipy import optimize  # type: ignore

from cr_knee_fit.constants import SPEED_OF_LIGHT_M_SEC
from cr_knee_fit.elements import Element
from cr_knee_fit.fit_data import Data, DataConfig
from cr_knee_fit.inference import (
    DEFAULT_CHI2_METHOD,
    DEFAULT_LOGNORMAL_CHI2,
    GoodnessOfFit,
    loglikelihood,
    logposterior,
    set_global_fit_data,
)
from cr_knee_fit.model import Model, ModelConfig
from cr_knee_fit.plotting import (  # noqa: F401
    PlotExportOpts,
    PlotsConfig,
    PosteriorPlotConfig,
    plot_everything,
)
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts
from cr_knee_fit.utils import E_N_GEV_LABEL, export_fig

# as recommended by emceee parallelization guide
# see https://emcee.readthedocs.io/en/stable/tutorials/parallel/#parallelization
os.environ["OMP_NUM_THREADS"] = "1"

IS_CLUSTER = os.environ.get("CRKNEES_CLUSTER") == "1"


@dataclasses.dataclass
class McmcConfig:
    n_steps: int
    n_walkers: int
    processes: int
    runtime_thinning: int | None = None
    reuse_saved: bool = True
    tau_eff_override: float | None = None


class FitConfig(pydantic.BaseModel):
    name: str
    fit_data_config: DataConfig
    mcmc: McmcConfig | None
    model: ModelConfig
    plots: PlotsConfig
    initial_guesses: Np2DArrayFp64  # n_sample x n_model_dim

    optimizer: Literal["scipy", "minuit"] = "scipy"

    # skips most analyses, just loading model dumps from disk whenever possible
    # useful to regenerate plots without rerunning the actual analysis
    reuse_saved_models: bool = False

    # Pydantic struff: nan/inf values must be written to JSON properly as Infinity/NaN
    model_config = pydantic.ConfigDict(ser_json_inf_nan="constants")

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
        generate_guess().validate_packing()
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

    def generate_initial_guess(self, ensure_finite_on_data: Data | None) -> Model:
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
            if ensure_finite_on_data is None or np.isfinite(
                logposterior(m, ensure_finite_on_data, self.model)
            ):
                return m
        raise ValueError(f"Failed to generate valid model in {n_try} tries")


startup_time = time.time()


def print_delim():
    dt = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")  # noqa: DTZ005
    elapsed_min = (time.time() - startup_time) / 60
    elapsed_hrs = elapsed_min / 60
    print("\n" + "=" * 15 + "\n" + f"{dt}; runtime: {elapsed_min:.4g} min = {elapsed_hrs:.4g} hrs")


def run_ml_analysis(
    config: FitConfig,
    fit_data: Data,
    freeze_shifts: bool,
    initial_model: Model | None = None,
) -> Model:
    model_config = config.model
    initial_model = initial_model or config.generate_initial_guess(fit_data)
    if freeze_shifts:
        model_config = dataclasses.replace(model_config, shifted_experiments=[])
        initial_model = dataclasses.replace(
            initial_model,
            energy_shifts=ExperimentEnergyScaleShifts({}),
        )

    def to_minimize(v: np.ndarray) -> float:
        # technically it should be -loglikelihood, but as we're using mostly flat priors
        # plus gaussian priors (L2 regularization) for experimental scale shifts
        # so, let us use logposterior instead
        return -logposterior(v, fit_data, model_config)

    print(f"Running optimization with {config.optimizer}...")
    start = time.time()
    match config.optimizer:
        case "scipy":
            res: optimize.OptimizeResult = optimize.minimize(
                to_minimize,
                x0=initial_model.pack(),
                bounds=initial_model.bounds(),
                method="Nelder-Mead",
                options={
                    "maxiter": 100_000,
                },
            )
        case "minuit":
            res = iminuit.minimize(
                to_minimize,
                x0=initial_model.pack(),
                bounds=initial_model.bounds(),
                options={"disp": True, "param_names": initial_model.labels(latex=False)},
            )
    total = time.time() - start
    print(f"Optimization done in {total:.2g} sec = {total / 60:.2g} min")
    print("Optimization result:")
    print(res)
    map_model = Model.unpack(res.x, layout_info=model_config)
    map_model.set_errcov(res.hess_inv)
    return map_model


def run_mcmc(
    config: FitConfig,
    mcmc_conf: McmcConfig,
    fit_data: Data,
    outdir: Path,
    sample_path: Path,
    mle_model: Model | None,
) -> np.ndarray:
    ndim = config.generate_initial_guess(fit_data).ndim()

    chain_path = outdir / "chain.h5"
    if chain_path.exists() and not mcmc_conf.reuse_saved:
        backup_chain_path = chain_path.with_suffix(f".bck-{datetime.datetime.now().isoformat()}.h5")  # noqa: DTZ005
        print(f"Not reusing the existing chain, backing up as {backup_chain_path.name}")
        shutil.move(chain_path, backup_chain_path)

    backend = emcee.backends.HDFBackend(filename=str(chain_path))

    sampler_pool_ctx = (
        multiprocessing.Pool(
            processes=mcmc_conf.processes,
            initializer=set_global_fit_data,
            initargs=(fit_data,),
        )
        if mcmc_conf.processes > 1
        else contextlib.nullcontext(enter_result=None)
    )
    with sampler_pool_ctx as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers=mcmc_conf.n_walkers,
            ndim=ndim,
            log_prob_fn=logposterior,
            args=(
                None,  # fit data is set in the global variable by the initializer
                config.model,
            ),
            pool=pool,
            backend=backend,
        )

        thin_by = mcmc_conf.runtime_thinning or 1

        try:
            sampler.get_log_prob()
            initial_state = None
            steps_already_run = sampler.iteration * thin_by
            # NOTE: this assumes thin_by doesn change across sampling runs, which might not be true
            steps_to_run = mcmc_conf.n_steps - steps_already_run
            print(f"Continuing previously started sampling saved in {backend.filename}")
        except AttributeError:
            print("Saved state not found, starting run from scratch")
            initial_state_samples: list[np.ndarray] = []
            if mle_model is not None and mle_model.sample_errcov() is not None:
                print("Initializing sampler around MLE model...")
                for _i_walker in range(mcmc_conf.n_walkers):
                    n_try = 1_000
                    for _i_try in range(n_try):
                        theta = mle_model.sample_errcov()
                        if theta is None:
                            continue
                        if np.isfinite(logposterior(theta, fit_data, config.model)):
                            initial_state_samples.append(theta)
                            break
                    else:
                        print(
                            f"Couldn't sample a valid model from errcov in {n_try} tries :( Initializing this walker with initial guess..."
                        )
                        initial_state_samples.append(config.generate_initial_guess(fit_data).pack())
            else:
                print("Initializing sampler initial guesses...")
                initial_state_samples = [
                    config.generate_initial_guess(fit_data).pack()
                    for _ in range(mcmc_conf.n_walkers)
                ]

            initial_state = np.array(initial_state_samples)

            steps_already_run = 0
            steps_to_run = mcmc_conf.n_steps

        if steps_to_run <= 0:
            print(
                "Seems like enough samples has been drawn: "
                + f"{steps_already_run} >= {mcmc_conf.n_steps}"
            )
            sampling_time_msg = "Existing chain reused"
        else:
            thinned_steps = math.ceil(steps_to_run / thin_by)
            print(
                f"Actual sampling steps to run: {steps_to_run}, thinned by {thin_by} = {thinned_steps}"
            )
            sampling_start = time.time()
            sampler.run_mcmc(
                initial_state,
                nsteps=thinned_steps,
                progress=not IS_CLUSTER,
                thin_by=thin_by,
            )
            sampling_time_sec = time.time() - sampling_start
            sampling_time_msg = f"Sampling done in {sampling_time_sec / 60:.0f} min ~ {sampling_time_sec / 3600:.1f} hrs"
            print(sampling_time_msg)

        print(f"Acceptance fraction: {sampler.acceptance_fraction.mean()}")

        if mcmc_conf.tau_eff_override is not None:
            print(f"Tau overridden: {mcmc_conf.tau_eff_override}")
            tau = None
            tau_eff = mcmc_conf.tau_eff_override
        else:
            print("Computing autocorr time...")
            tau = sampler.get_autocorr_time(quiet=True)
            print(f"{tau = }")
            # tau_eff = float(np.quantile(tau[np.isfinite(tau)], q=0.95))
            tau_eff = np.median(tau)

        print(f"Effective tau = {tau_eff}...")
        burn_in = int(5 * tau_eff)
        thin = int(2 * tau_eff)

        print(f"Burn in: {burn_in}; Thinning: {thin}")

        theta_sample: np.ndarray = sampler.get_chain(flat=True, discard=burn_in, thin=thin)  # type: ignore

        model_example = Model.unpack(theta_sample[0, :], layout_info=config.model)
        tau_labels = model_example.labels(latex=False)
        np.savetxt(
            sample_path,
            theta_sample,
            header="\n".join(
                [
                    f"Generated on: {datetime.datetime.now()}",  # noqa: DTZ005
                    sampling_time_msg,
                    f"MCMC config: {mcmc_conf}",
                    (
                        f"Estimated autocorrelation lengths: {', '.join(f'{label}: {t}' for label, t in zip(tau_labels, tau))}"
                        if tau is not None
                        else "<overriden>"
                    ),
                    f"Effective autocorrelation length: {tau_eff}",
                    f"Burn-in, steps: {burn_in}",
                    f"Thinning, steps: {thin}",
                    f"Sample shape: {theta_sample.shape}",
                ]
            ),
        )

    return theta_sample


def plot_and_print_model(
    outdir: Path,
    dirname: str,
    model: Model,
    fit_data: Data,
    validation_data: Data,
    scale: float,
    config: FitConfig,
):
    dest = outdir / dirname
    dest.mkdir(exist_ok=True)

    model.print_params()
    model.plot_spectra(fit_data, scale=scale, validation_data=validation_data).savefig(
        dest / "spectra.png"
    )

    if fig := model.plot_lnA(fit_data, validation_data):
        fig.savefig(dest / "lnA.png")
    if fig := model.plot_flux_ratios(fit_data, validation_data):
        fig.savefig(dest / "flux-ratios.png")

    if config.plots.datasets:
        datasetes_dir = dest / "datasets"
        datasetes_dir.mkdir(exist_ok=True)
        for (exp, observable), fig in model.plot_all_datasets(
            fit_data, spectra_scale=scale, validation_data=validation_data
        ).items():
            fig.savefig(datasetes_dir / f"{exp.filename_prefix}_{observable}.png")

    if config.plots.observables:
        observables_dir = dest / "observables"
        observables_dir.mkdir(exist_ok=True)
        for (observable), fig in model.plot_all_observables(
            fit_data, spectra_scale=scale, validation_data=validation_data
        ).items():
            fig.savefig(observables_dir / f"{observable}.png")

    if config.plots.energy_density:
        fig, ax = plt.subplots()
        E_n = np.geomspace(1e2, 1e7, 300)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(
            E_n, E_n**2 * model.compute_energy_density(E_n), color="k", linewidth=2, label="Total"
        )
        for elements in ([Element.H], [Element.He], Element.nuclei()):
            ax.plot(
                E_n,
                E_n**2 * model.compute_energy_density(E_n, contributing_elements=elements),
                color=elements[0].color if len(elements) == 1 else "magenta",
                linewidth=1,
                label=elements[0].name if len(elements) == 1 else "nuclei",
            )

        ax.set_xlim(E_n[0], E_n[-1])
        ax.set_xlabel(E_N_GEV_LABEL)
        ax.set_ylabel("$ E^2 n_\\text{CR} $ / $ \\text{GeV} \\; \\text{m}^{-3} $")
        ax.legend()
        fig.tight_layout()  # type: ignore
        fig.savefig(dest / "energy-density.png")

        density2spec = 1e6 / (
            4 * np.pi / SPEED_OF_LIGHT_M_SEC
        )  # GeV^-1 m^-3 density -> GeV^-1 cm^-2 s^-1 sr^-1 intensity

        components = [
            model.compute_energy_density(E_n, contributing_elements=[Element.H]),
            model.compute_energy_density(E_n, contributing_elements=[Element.He]),
            model.compute_energy_density(E_n, contributing_elements=Element.nuclei()),
        ]
        np.savetxt(
            dest / "energy-density.txt",
            np.vstack(
                (
                    E_n,
                    *[density2spec * comp for comp in components],
                    *components,
                )
            ).T,
            header="\n".join(
                [
                    f"Model dir: {outdir}",
                    f"Dumped on: {datetime.datetime.now()}",  # noqa: DTZ005
                    "Columns: (1) E per nucleon [GeV], (2-4) proton, helium and nuclei spectra [GeV^-1 cm^-2 s^-1 sr^-1], "
                    + "(5-7) proton, helium and nuclei energy densities [GeV^-1 m^-3]",
                ]
            ),
        )

    plt.close("all")


def run_analysis(config: FitConfig, outdir: Path) -> None:
    print(f"Output dir: {outdir}")
    print(f"Default chi2 method: {DEFAULT_CHI2_METHOD}")
    print(f"Default lognormal chi2: {DEFAULT_LOGNORMAL_CHI2}")

    Path(outdir / "config-dump.json").write_text(config.model_dump_json(indent=2))

    def load_saved(path: Path) -> Model | None:
        if config.reuse_saved_models:
            return Model.load(path, layout_info=config.model)
        else:
            return None

    print_delim()
    print("Loading fit data...")
    fit_data = Data.load(config.fit_data_config)
    assert not fit_data.is_empty(), "Fit data cannot be empty"
    set_global_fit_data(fit_data)
    scale = 2.75 if fit_data.E_max() > 2e6 else 2.6
    if fig := fit_data.plot(scale=scale, describe=True):
        fig.savefig(outdir / "data.png")

    if config.plots.validation_data_config is not None:
        print("\nLoading validation data...")
        validation_data = Data.load(config.plots.validation_data_config)
        if fig := validation_data.plot(scale=scale, describe=True):
            fig.savefig(outdir / "data-validation.png")
        if validation_data.is_empty():
            print("    (empty)")
    else:
        validation_data = Data.empty()

    print_delim()
    print("Initial guess model:")
    initial_guess = config.generate_initial_guess(ensure_finite_on_data=None)
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
    mle_model_dump = outdir / "mle-prelim.txt"

    if loaded := load_saved(mle_model_dump):
        print(f"Loaded ML analysis result from {mle_model_dump}")
        mle_model = loaded
        loglike = loglikelihood(mle_model, fit_data, mle_model.layout_info())
        print(f"Loglike: {loglike}")
    else:
        print("Running preliminary ML analysis...")
        mle_model = run_ml_analysis(
            config=config,
            fit_data=fit_data,
            freeze_shifts=False,
            initial_model=None,
        )
        mle_model.save(
            mle_model_dump, header=[f"GoF: {GoodnessOfFit.compute(mle_model, fit_data)}"]
        )

    # TODO: remove after 3pop significance tests are done
    gof_dump = mle_model_dump.with_name(mle_model_dump.stem + ".gof.json")
    if not gof_dump.exists():
        GoodnessOfFit.compute(mle_model, fit_data).save(gof_dump)

    plot_and_print_model(
        outdir=outdir,
        dirname="mle-prelim",
        model=mle_model,
        fit_data=fit_data,
        validation_data=validation_data,
        scale=scale,
        config=config,
    )

    print_delim()
    print("Running bayesian analysis...")
    if config.mcmc is None:
        print("Skipping, config is None")
        return
    print(f"MCMC config: {config.mcmc}")

    ndim = config.generate_initial_guess(None).ndim()
    print(f"N dim = {ndim}")

    sample_path = outdir / "theta.txt"
    if config.reuse_saved_models and sample_path.exists():
        print("Loading and reusing saved model sample")
        theta_sample = np.loadtxt(sample_path)
        assert theta_sample.ndim == 2, "Saved theta sample has the wrong number of dimensions"
        assert theta_sample.shape[1] == ndim, "Saved theta sample has wrong dimensions"
    else:
        theta_sample = run_mcmc(
            config,
            config.mcmc,
            fit_data=fit_data,
            outdir=outdir,
            sample_path=sample_path,
            mle_model=mle_model,
        )

    print(f"MCMC sample ready, shape: {theta_sample.shape}")
    median_model = Model.unpack(np.median(theta_sample, axis=0), layout_info=config.model)
    print("Median model:")
    median_model.print_params()

    samples_to_search_for_best = 1000
    print(f"Computing loglikelihood for the first {samples_to_search_for_best} samples")
    model_sample = [
        Model.unpack(theta, layout_info=config.model)
        for theta in theta_sample[:samples_to_search_for_best, :]
    ]
    loglike_values = [logposterior(model, fit_data, config=config.model) for model in model_sample]
    best_fit_idx = np.argmax(loglike_values)
    print(f"Best-fitting model idx: {best_fit_idx}; loglike = {loglike_values[best_fit_idx]}")
    posterior_best_model = model_sample[best_fit_idx]

    if config.plots.corner:
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
    posterior_ml_dump = outdir / "mle-map.txt"
    if loaded := load_saved(posterior_ml_dump):
        print(f"Loaded ML analysis result from {posterior_ml_dump}")
        posterior_ml_best = loaded
        loglike = loglikelihood(posterior_ml_best, fit_data, posterior_ml_best.layout_info())
        print(f"Loglike: {loglike}")
    else:
        print("Running ML analysis from the best-fitting posterior point")
        posterior_ml_best = run_ml_analysis(
            config=config,
            fit_data=fit_data,
            freeze_shifts=False,
            initial_model=posterior_best_model,
        )
        posterior_ml_best.save(
            posterior_ml_dump, header=[f"GoF: {GoodnessOfFit.compute(posterior_ml_best, fit_data)}"]
        )

    plot_and_print_model(
        outdir=outdir,
        dirname="mle-map",
        model=posterior_ml_best,
        fit_data=fit_data,
        validation_data=validation_data,
        scale=scale,
        config=config,
    )

    print_delim()
    print("Plotting final model plot")

    best_fit_model = posterior_ml_best or posterior_best_model
    fig = plot_everything(
        plots_config=config.plots,
        theta_sample=model_sample,
        theta_bestfit=best_fit_model.pack(),
        model_config=config.model,
        spectra_scale=scale,
        fit_data=fit_data,
        validation_data=validation_data,
    )
    fig.savefig(outdir / "model.pdf")
    if config.plots.export_opts.main is not None:
        export_fig(fig, filename=config.plots.export_opts.main)
    fig.savefig(outdir / "model.png")

    print_delim()
    print("Bye!")


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
    run_analysis(fit_config, outdir=config_path.parent)
