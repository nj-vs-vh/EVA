import argparse
import contextlib
import dataclasses
import datetime
import math
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable
from warnings import warn

import corner  # type: ignore
import emcee  # type: ignore
import numpy as np
import pydantic
from matplotlib.figure import Figure
from pydantic_numpy.typing import Np2DArrayFp64  # type: ignore
from scipy import optimize  # type: ignore

from cr_knee_fit.fit_data import Data, DataConfig
from cr_knee_fit.inference import (
    CHI2_METHOD,
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
from cr_knee_fit.utils import export_fig

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

    # skips most analyses, just loading model dumps from disk whenever possible
    # useful to regenerate plots without rerunning the actual analysis
    reuse_saved_models: bool = False

    def __post_init__(self) -> None:
        model_elements = set(self.model.elements(only_fixed_Z=True))
        data_elements = set(self.fit_data_config.default_elements)
        unconstrained_elements = model_elements - data_elements
        if unconstrained_elements:
            warn(
                f"Some elements in the model are not contstrained by data: {sorted(unconstrained_elements)}"
            )

        if self.fit_data_config.aux_data:
            warn(
                "Aux data are ignored in fit data; use validation data "
                + f"for posterior plotting: {self.fit_data_config.aux_data}"
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
        # but as mentioned above, this shouldn't matter too much due to our choice of mostly
        # trivial flat priors
        aic=2 * initial_model.ndim() - 2 * max_logpost,
    )
    print(f"Goodness of fit: {gof}")
    return map_model, gof


def run_mcmc(
    config: FitConfig, mcmc_conf: McmcConfig, fit_data: Data, outdir: Path, sample_path: Path
) -> np.ndarray:
    ndim = config.generate_initial_guess(fit_data).ndim()

    chain_path = outdir / "chain.h5"
    if chain_path.exists() and not mcmc_conf.reuse_saved:
        backup_chain_path = chain_path.with_suffix(f".bck-{datetime.datetime.now().isoformat()}.h5")
        print(f"Not reusing the existing chain, backing up as {backup_chain_path.name}")
        shutil.move(chain_path, backup_chain_path)

    backend = emcee.backends.HDFBackend(filename=str(chain_path))

    sampler_pool_ctx = (
        multiprocessing.Pool(processes=mcmc_conf.processes)
        if mcmc_conf.processes > 1
        else contextlib.nullcontext(enter_result=None)
    )
    with sampler_pool_ctx as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers=mcmc_conf.n_walkers,
            ndim=ndim,
            log_prob_fn=logposterior,
            args=(
                # on macos passing fit data through a global variable doesn't work
                # probably because it's using "spawn" multiprocessing method
                fit_data if sys.platform == "darwin" else None,
                config.model,
            ),
            pool=pool,
            backend=backend,
        )

        thin_by = mcmc_conf.runtime_thinning or 1

        try:
            sampler.get_log_prob()
            initial_state = None
            steps_to_run = (
                mcmc_conf.n_steps - sampler.iteration * thin_by
            )  # NOTE: this assumes thin_by doesn change across sampling runs, which might not be true
            print(f"Continuing previously started sampling saved in {backend.filename}")
        except AttributeError:
            print("Saved state not found, initializing sampler near ML solution")
            initial_state = np.array(
                [config.generate_initial_guess(fit_data).pack() for _ in range(mcmc_conf.n_walkers)]
            )
            steps_to_run = mcmc_conf.n_steps

        if steps_to_run <= 0:
            print(f"Seems like enough samples has been drawn (>={mcmc_conf.n_steps})")
            sampling_time_msg = "Existing chain reused"
        else:
            thinned_steps = int(math.ceil(steps_to_run / thin_by))
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
            tau_eff = float(np.quantile(tau[np.isfinite(tau)], q=0.95))

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
                    f"Generated on: {datetime.datetime.now()}",
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


def run_analysis(config: FitConfig, outdir: Path) -> None:
    print(f"Output dir: {outdir}")
    print(f"chi2 method: {CHI2_METHOD}")

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

    validation_data = (
        Data.load(config.plots.validation_data_config)
        if config.plots.validation_data_config is not None
        else Data.empty()
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
    if validation_data.aux_data:
        mle_model.plot_aux_data(spectra_scale=scale, validation_data=validation_data).savefig(
            outdir / "preliminary-mle-aux-data.png"
        )
    mle_model.plot_lnA(fit_data, validation_data=validation_data).savefig(
        outdir / "preliminary-mle-result-lnA.png"
    )

    print_delim()
    print("Running bayesian analysis...")
    if config.mcmc is None:
        print("Skipping, config is None")
        return
    print(f"MCMC config: {config.mcmc}")

    ndim = config.generate_initial_guess(fit_data).ndim()
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
    fig = plot_everything(
        plots_config=config.plots,
        theta_sample=theta_sample,
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
