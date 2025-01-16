from typing import Callable
from scipy import stats
import numpy as np

from cr_knee_fit.galactic import GalacticCR, RigidityBreak
from cr_knee_fit.model import ModelConfig, Model
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts
from cr_knee_fit.spectrum import PowerLaw
from cr_knee_fit.types_ import Experiment, FitData, Primary


def logprior(model: Model) -> float:
    res = 0

    # DAMPE break
    if not 3 < model.cr.dampe_break.lg_R < 5:
        return -np.inf
    # hardening
    if model.cr.dampe_break.d_alpha < 0:
        return -np.inf
    # "fixing" break sharpness with a very narrow prior
    res += stats.norm.logpdf(model.cr.dampe_break.lg_sharpness, loc=np.log10(5), scale=0.01)

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
                prediction = m.cr.compute(E=data.E, particle=particle)
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


def initial_guess_model(config: ModelConfig) -> Model:
    return Model(
        cr=GalacticCR(
            components={
                p: PowerLaw(
                    lgI=stats.norm.rvs(loc=-4, scale=0.5) - 2.6 * np.log10(p.Z),
                    alpha=stats.norm.rvs(loc=2.7, scale=0.1),
                )
                for p in config.primaries
            },
            dampe_break=RigidityBreak(
                lg_R=stats.norm.rvs(loc=4.0, scale=0.5),  # DAMPE break at ~1 TeV
                d_alpha=stats.norm.rvs(loc=0, scale=0.1) ** 2,
                lg_sharpness=stats.norm.rvs(loc=np.log10(5), scale=0.01),
            ),
        ),
        energy_shifts=ExperimentEnergyScaleShifts(
            lg_shifts={exp: stats.norm.rvs(loc=0, scale=0.01) for exp in config.shifted_experiments}
        ),
    )
