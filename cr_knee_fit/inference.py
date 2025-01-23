import numpy as np
from scipy import stats

from cr_knee_fit.cr_model import RigidityBreak
from cr_knee_fit.fit_data import FitData
from cr_knee_fit.model import Model, ModelConfig


def break_prior(b: RigidityBreak, lg_R_min: float, lg_R_max: float, is_softening: bool) -> float:
    if not (lg_R_min < b.lg_R < lg_R_max):
        return -np.inf
    if (b.d_alpha > 0) != is_softening:
        return -np.inf
    return 0


def logprior(model: Model) -> float:
    res = 0

    # breaks must be ordered in R to avoid ambiguity
    breaks_lgR = [m.lg_R for m in model.cr_model.breaks]
    if breaks_lgR != sorted(breaks_lgR):
        return -np.inf

    # dampe break
    res += break_prior(model.cr_model.breaks[0], lg_R_min=3, lg_R_max=5, is_softening=True)
    if len(model.cr_model.breaks) > 1:
        # grapes hardening
        res += break_prior(model.cr_model.breaks[1], lg_R_min=4.5, lg_R_max=6, is_softening=False)
    if len(model.cr_model.breaks) > 2:
        # all-particle knee
        res += break_prior(model.cr_model.breaks[2], lg_R_min=5.5, lg_R_max=7, is_softening=True)

    lgK = model.cr_model.all_particle_lg_shift
    if lgK is not None:
        if not (1 <= 10**lgK <= 1.5):
            return -np.inf

    # energy shift priors
    # TODO: realistic priors from experiments' energy scale systematic uncertainties
    for _, lg_shift in model.energy_shifts.lg_shifts.items():
        resolution_percent = 10
        lg_sigma = np.abs(np.log10(1 + resolution_percent / 100))
        res += stats.norm.logpdf(lg_shift, loc=0, scale=lg_sigma)

    return res  # type: ignore


def to_model(
    model_or_theta: Model | np.ndarray,
    config: ModelConfig,
) -> Model:
    return (
        model_or_theta
        if isinstance(model_or_theta, Model)
        else Model.unpack(model_or_theta, layout_info=config)
    )


def loglikelihood(
    model_or_theta: Model | np.ndarray,
    fit_data: FitData,
    config: ModelConfig,
) -> float:
    m = to_model(model_or_theta, config)
    res = 0
    for experiment, particle_data in fit_data.spectra.items():
        for particle, data in particle_data.items():
            data = data.with_shifted_energy_scale(f=m.energy_shifts.f(experiment))
            prediction = m.cr_model.compute(E=data.E, primary=particle)
            loglike_per_bin = -0.5 * (
                np.where(
                    prediction > data.F,
                    ((prediction - data.F) / data.F_errhi) ** 2,
                    ((prediction - data.F) / data.F_errlo) ** 2,
                )
            )
            if np.any(np.isnan(loglike_per_bin)):
                return -np.inf
            res += float(np.sum(loglike_per_bin))
    for experiment, data in fit_data.all_particle_spectra.items():
        data = data.with_shifted_energy_scale(f=m.energy_shifts.f(experiment))
        prediction = m.cr_model.compute_all_particle(E=data.E)
        loglike_per_bin = -0.5 * (
            np.where(
                prediction > data.F,
                ((prediction - data.F) / data.F_errhi) ** 2,
                ((prediction - data.F) / data.F_errlo) ** 2,
            )
        )
        if np.any(np.isnan(loglike_per_bin)):
            return -np.inf
        res += float(np.sum(loglike_per_bin))
    return res


# to optimize logposterior evaluation in a multiprocessing setup
# see https://emcee.readthedocs.io/en/stable/tutorials/parallel/#pickling-data-transfer-arguments
fit_data_global: FitData | None = None


def set_global_fit_data(fit_data: FitData):
    global fit_data_global
    fit_data_global = fit_data


def logposterior(
    model_or_theta: Model | np.ndarray, fit_data: FitData | None, config: ModelConfig
) -> float:
    model = to_model(model_or_theta, config)
    logpi = logprior(model)
    if not np.isfinite(logpi):
        return logpi
    fit_data_ = fit_data or fit_data_global
    if fit_data_ is None:
        raise ValueError("fit data must be either passed directly or through a global variable")
    return logpi + loglikelihood(model, fit_data_, config)
