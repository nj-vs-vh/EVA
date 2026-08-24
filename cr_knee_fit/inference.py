import dataclasses
import itertools
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import stats  # type: ignore

from cr_knee_fit import experiments
from cr_knee_fit.fit_data import Data
from cr_knee_fit.model import Model, ModelConfig

energy_scale_uncertainties = {
    # 10.1103/PhysRevLett.113.121101
    # > This comparison limits the uncertainty of the absolute energy scale to 2% in the range covered by
    #   the beam test results, 10–290 GeV. It increases to 5% at 0.5 GeV and to 3% at 500 GeV.
    experiments.ams02: 3.0,
    # 10.1103/PhysRevLett.119.181101
    # > It is found that the average ratio of the expected to measured cutoff position in the electron flux
    #   is 1.035 +/- 0.009 (stat). As a result, a correction of the energy scale by 3.5% was implemented in
    #   the analysis.
    experiments.calet: 0.9,
    # 10.22323/1.301.0197
    # > we provide an estimation on absolute energy scale of DAMPE 1.25% higher than expected at about 13GeV
    #   energy with uncertainty about ±1.75%(stat)±1.34%(sys)
    # combining 1.75 and 1.34 in lg quadratures, we get 2.2
    experiments.dampe: 2.2,
    # no direct statement found on energy scale uncertainty found, but arXiV:2004.10371 lists energy resolution
    # at a level of 6.4% at 150 GeV, so we take the same value for energy scale uncertainty
    experiments.cream: 6.4,
    experiments.iss_cream: 6.4,
    # 10.1016/j.astropartphys.2024.103077
    # > Fig. 9 [...] The uncertainty in the energy spectrum induced by a systematic error in the energy scale
    # equal to deltaE = 9% is displayed in the upper-right corner of the figure using arrows.
    experiments.hawc: 9.0,
    # 10.1016/j.nima.2004.11.025
    # no energy scale data found, but energy resolution is said to be ~10%
    experiments.grapes: 10.0,
    # 10.1103/PhysRevD.104.062007 and 10.1051/epjconf/202328302002
    # > The uncertainty of 30% is statistics dominant in the measurement of the shift.
    # however, we tentatively set the uncertainty to ~10 to roughly match other indirect experiments
    experiments.lhaaso_epos: 10.0,
    experiments.lhaaso_qgsjet: 10.0,
    experiments.lhaaso_sibyll: 10.0,
}

default_energy_scale_uncertainty = 10.0


def percent2lg(percent: float) -> float:
    upper = np.log10(1 + percent / 100)
    lower = -np.log10(1 - percent / 100)
    return 0.5 * (upper + lower)


energy_scale_lg_uncertainties = {
    exp: percent2lg(energy_scale_uncertainties.get(exp, default_energy_scale_uncertainty))
    for exp in experiments.ALL
}


def get_energy_scale_lg_uncertainty(exp: experiments.Experiment) -> float:
    return energy_scale_lg_uncertainties[exp]


def logprior(model: Model, model_packed: np.ndarray | None) -> float:
    res = 0.0

    # parameter bounds check
    for param, bound in zip(
        (model_packed if model_packed is not None else model.pack()),
        model.bounds(),
        strict=True,
    ):
        if bound is None:
            continue
        min, max = bound
        if not (min <= param <= max):
            return -np.inf

    # breaks ordering check to avoid sampling all the permutations
    for population in model.populations:
        break_positions = [m.lg_break for m in population.breaks]
        if break_positions != sorted(break_positions):
            return -np.inf

    # sigmoid prior on ratio of populations' energy densities
    # TODO: tunable params
    Emin = 1.0
    Emax = 1e8
    npoints = 200
    energy_dominant_pop_densities = [
        pop._compute_energy_density_simpson(Emin=Emin, Emax=Emax, npoints=npoints)
        for pop in model.populations
        if pop.population_meta and pop.population_meta.is_apriori_energy_dominant is True
    ]
    energy_inferior_pop_densities = [
        pop._compute_energy_density_simpson(Emin=Emin, Emax=Emax, npoints=npoints)
        for pop in model.populations
        if pop.population_meta and pop.population_meta.is_apriori_energy_dominant is False
    ]
    for dom, inf in itertools.product(energy_dominant_pop_densities, energy_inferior_pop_densities):
        ratio = dom / inf
        print(dom, inf, ratio)
        res += -np.log(1 + ratio**-1)  # "smooth step" preferring dom >> inf

    # Gaussian priors on experimental energy scale shifts
    for exp, lg_shift in model.energy_shifts.lg_shifts.items():
        lg_sigma = model.energy_scale_lg_uncertainty_override.get(
            exp
        ) or get_energy_scale_lg_uncertainty(exp)
        res += stats.norm.logpdf(lg_shift, loc=0, scale=lg_sigma)

    return res  # type: ignore


def saturated_logprior(model: Model) -> float:
    res = 0.0
    for exp in model.energy_shifts.lg_shifts:
        lg_sigma = model.energy_scale_lg_uncertainty_override.get(
            exp
        ) or get_energy_scale_lg_uncertainty(exp)
        res += stats.norm.logpdf(0.0, loc=0, scale=lg_sigma)
    return res  # type: ignore


def ensure_model(model_or_theta: Model | np.ndarray, config: ModelConfig) -> Model:
    if isinstance(model_or_theta, Model):
        return model_or_theta
    else:
        return Model.unpack(model_or_theta, layout_info=config)


Chi2Method = Literal["correlated", "dimidated"]
DEFAULT_CHI2_METHOD = os.environ.get("CRKNEE_CHI2_METHOD", "correlated")


def set_default_chi2_method(new: Chi2Method) -> None:
    global DEFAULT_CHI2_METHOD
    DEFAULT_CHI2_METHOD = new


def chi_squared_loglikelihood(
    prediction: np.ndarray,
    y: np.ndarray,
    err_stat: np.ndarray,
    err_syst: np.ndarray,
    inv_err_cov: np.ndarray,
    method: Chi2Method | None = None,
) -> float:
    residual = prediction - y
    match method or DEFAULT_CHI2_METHOD:
        case "dimidated":
            # Chi2 based on dimidated Gaussian, i.e. using upper and lower errors on corresponding sides. Statistical and systematic
            # uncertainties are assumed to be uncorrelated between bins and are added in quadratures. Further sophistication is
            # possible e.g. by using asymmetric error summation from Barlow, R. Asymmetric Systematic Errors. arXiv:physics/0306138
            residual_sq = residual**2
            err_squared_total = err_stat**2 + err_syst**2
            loglike_per_bin = -0.5 * (
                np.where(
                    residual > 0,
                    residual_sq / err_squared_total[:, 1],
                    residual_sq / err_squared_total[:, 0],
                )
            )
            return float(np.sum(loglike_per_bin))
        case "correlated":
            # Chi2 accounting for error correlation, see err_cov method. Note that errors are symmetrized in this case.
            residual_vec = residual.reshape((-1, 1))
            result = -0.5 * (residual_vec.T @ inv_err_cov @ residual_vec)
            return float(np.squeeze(result))
        case unexpected:
            raise RuntimeError(
                f"Unexpected chi2 method specified in {'argument' if method is None else 'env var'}: {unexpected}"
            )


DEFAULT_LOGNORMAL_CHI2 = os.environ.get("CRKNEE_LOGNORMAL_CHI2", "1") == "1"


def set_default_lognormal_chi2(new: bool) -> None:
    global DEFAULT_LOGNORMAL_CHI2
    DEFAULT_LOGNORMAL_CHI2 = new


def loglikelihood(
    model_or_theta: Model | np.ndarray,
    fit_data: Data,
    config: ModelConfig,
    chi2_method: Chi2Method | None = None,
    lognormal: bool | None = None,
) -> float:
    model = ensure_model(model_or_theta, config)
    res = 0.0

    use_lognormal_chi2 = lognormal or DEFAULT_LOGNORMAL_CHI2

    for spectrum in fit_data.spectra:
        f = model.energy_shifts.f(spectrum.experiment)
        if use_lognormal_chi2:
            d = spectrum.data_for_lognormal_chi2(f)
            prediction = np.log10(model.compute_spectrum(d.x, element=spectrum.spec, quantity="E"))
        else:
            d = spectrum.data_for_normal_chi2(f)
            prediction = model.compute_spectrum(d.x, element=spectrum.spec, quantity="E")

        res += chi_squared_loglikelihood(
            prediction=prediction,
            y=d.y,
            err_stat=d.err_stat,
            err_syst=d.err_syst,
            inv_err_cov=d.standard_inv_err_cov,
            method=chi2_method,
        )

    for lnA_data in fit_data.lnA:
        f = model.energy_shifts.f(lnA_data.experiment)
        # for lnA, the energy scale shift does not affect values as it includes dE in both numerator and denominator
        lnA_data = lnA_data.with_shifted_grid(f)
        res += chi_squared_loglikelihood(
            prediction=model.compute_lnA(lnA_data.x),
            y=lnA_data.y,
            err_stat=lnA_data.err_stat,
            err_syst=lnA_data.err_syst,
            inv_err_cov=lnA_data.standard_inv_err_cov,
            method=chi2_method,
        )

    for flux_ratio in fit_data.flux_ratios:
        # flux ratios are used at low energies, where energy scale is very well constrained, hence no shift is applied
        prediction = model.compute_flux_ratio(
            flux_ratio.Q, fr=flux_ratio.ratio, quantity=flux_ratio.quantity
        )
        d = flux_ratio.d
        if use_lognormal_chi2:
            prediction = np.log10(prediction)
            d = d.log10_ed
        res += chi_squared_loglikelihood(
            prediction=prediction,
            y=d.y,
            err_stat=d.err_stat,
            err_syst=d.err_syst,
            inv_err_cov=d.standard_inv_err_cov,
            method=chi2_method,
        )

    return res


def saturated_loglikelihood(
    fit_data: Data,
    chi2_method: Chi2Method | None = None,
    lognormal: bool | None = None,
) -> float:
    res = 0.0
    f = 1.0  # energy scale shifts do not impact the saturated likelihood computation
    use_lognormal_chi2 = lognormal or DEFAULT_LOGNORMAL_CHI2

    for spectrum in fit_data.spectra:
        if use_lognormal_chi2:
            d = spectrum.data_for_lognormal_chi2(f)
        else:
            d = spectrum.data_for_normal_chi2(f)
        res += chi_squared_loglikelihood(
            prediction=d.y,
            y=d.y,
            err_stat=d.err_stat,
            err_syst=d.err_syst,
            inv_err_cov=d.standard_inv_err_cov,
            method=chi2_method,
        )

    for lnA_data in fit_data.lnA:
        res += chi_squared_loglikelihood(
            prediction=lnA_data.y,
            y=lnA_data.y,
            err_stat=lnA_data.err_stat,
            err_syst=lnA_data.err_syst,
            inv_err_cov=lnA_data.standard_inv_err_cov,
            method=chi2_method,
        )

    for flux_ratio in fit_data.flux_ratios:
        d = flux_ratio.d
        if use_lognormal_chi2:
            d = d.log10_ed
        res += chi_squared_loglikelihood(
            prediction=d.y,
            y=d.y,
            err_stat=d.err_stat,
            err_syst=d.err_syst,
            inv_err_cov=d.standard_inv_err_cov,
            method=chi2_method,
        )

    return res


# to optimize logposterior evaluation in a multiprocessing setup
# see https://emcee.readthedocs.io/en/stable/tutorials/parallel/#pickling-data-transfer-arguments
fit_data_global: Data | None


def set_global_fit_data(fit_data: Data):
    global fit_data_global
    fit_data_global = fit_data


def logposterior(
    model_or_theta: Model | np.ndarray,
    fit_data: Data | None,
    config: ModelConfig,
) -> float:

    model = ensure_model(model_or_theta, config)
    logpi = logprior(
        model,
        model_packed=model_or_theta if not isinstance(model_or_theta, Model) else None,
    )
    if not np.isfinite(logpi):
        return logpi

    fit_data_ = fit_data or fit_data_global
    if fit_data_ is None:
        raise ValueError("fit data must be either passed directly or through a global variable")
    res = logpi + loglikelihood(model, fit_data_, config)
    if not np.isfinite(res):
        return -np.inf
    return res


def saturated_logposterior(
    model_or_theta: Model | np.ndarray,
    fit_data: Data | None,
    config: ModelConfig,
) -> float:
    model = ensure_model(model_or_theta, config)
    logpi = saturated_logprior(model)
    fit_data_ = fit_data or fit_data_global
    if fit_data_ is None:
        raise ValueError("fit data must be either passed directly or through a global variable")
    return logpi + saturated_loglikelihood(fit_data_)


@dataclasses.dataclass
class GoodnessOfFit:
    logpost: float
    deviance: float
    n_data: int
    n_param: int
    aic: float

    def __str__(self) -> str:
        return (
            f"logpost = {self.logpost:.6g}; "
            + f"D / ndof = {self.deviance:.4g} / {self.ndof} = {self.deviance / self.ndof:.4g}; "
            + f"AIC = {self.aic:.6g}"
        )

    @property
    def ndof(self) -> int:
        return self.n_data - self.n_param

    @staticmethod
    def compute(model: Model, fit_data: Data) -> "GoodnessOfFit":
        logpost = logposterior(model, fit_data, model.layout_info())
        return GoodnessOfFit(
            logpost=logpost,
            deviance=float(
                2 * (saturated_logposterior(model, fit_data, model.layout_info()) - logpost)
            ),
            n_data=fit_data.size(),
            n_param=model.size,
            aic=2 * model.size - 2 * logpost,
        )

    def save(self, file: Path) -> None:
        file.write_text(json.dumps(dataclasses.asdict(self), indent=2))

    @staticmethod
    def load(file: Path) -> "GoodnessOfFit":
        return GoodnessOfFit(**json.loads(file.read_text()))
