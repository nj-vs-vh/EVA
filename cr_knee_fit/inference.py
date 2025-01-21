from typing import Callable

import numpy as np
from scipy import stats

from cr_knee_fit.fit_data import FitData
from cr_knee_fit.model import Model, ModelConfig


def logprior(model: Model) -> float:
    res = 0

    # breaks must be ordered in R to avoid ambiguity
    breaks_lgR = [m.lg_R for m in model.cr_model.breaks]
    if breaks_lgR != sorted(breaks_lgR):
        return -np.inf

    dampe_break = model.cr_model.breaks[0]
    if not (3 < dampe_break.lg_R < 5):
        return -np.inf
    # enforce softening
    if dampe_break.d_alpha < 0:
        return -np.inf
    # "fixing" break sharpness with a very narrow prior
    res += stats.norm.logpdf(dampe_break.lg_sharpness, loc=np.log10(5), scale=0.01)

    if len(model.cr_model.breaks) > 1:
        grapes_hardening = model.cr_model.breaks[1]
        if not (4.5 < grapes_hardening.lg_R < 6):
            return -np.inf
        if grapes_hardening.d_alpha > 0:
            return -np.inf
        res += stats.norm.logpdf(grapes_hardening.lg_sharpness, loc=np.log10(10), scale=0.01)

    if len(model.cr_model.breaks) > 2:
        knee = model.cr_model.breaks[2]
        if not (5.5 < knee.lg_R < 7):
            return -np.inf
        if knee.d_alpha < 0:
            return -np.inf
        res += stats.norm.logpdf(knee.lg_sharpness, loc=np.log10(5), scale=0.01)

    lgK = model.cr_model.all_particle_lg_shift
    if lgK is not None:
        res += stats.norm.logpdf(lgK, scale=0.1)

    # energy shift priors
    # TODO: realistic priors from experiments' energy scale systematic uncertainties
    for _, lg_shift in model.energy_shifts.lg_shifts.items():
        resolution_percent = 10
        lg_sigma = np.abs(np.log10(1 + resolution_percent / 100))
        res += stats.norm.logpdf(lg_shift, loc=0, scale=lg_sigma)
    return float(res)


def make_loglikelihood(fit_data: FitData, config: ModelConfig) -> Callable[[np.ndarray], float]:
    def loglike(theta: np.ndarray) -> float:
        m = Model.unpack(theta, layout_info=config)
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
        if config.cr_model_config.fit_all_particle:
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

    return loglike


def make_logposterior(fit_data: FitData, config: ModelConfig) -> Callable[[np.ndarray], float]:
    loglike = make_loglikelihood(fit_data, config)

    def logpost(theta: np.ndarray) -> float:
        model = Model.unpack(theta, layout_info=config)
        logpi = logprior(model)
        if not np.isfinite(logpi):
            return logpi
        return logpi + loglike(theta)

    return logpost
