import itertools
import os
from typing import Literal

import numpy as np
from scipy import stats  # type: ignore

from cr_knee_fit import experiments
from cr_knee_fit.cr_model import ExpCutoff, LognormalSourceMaxAcceleration
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


def logprior(model: Model) -> float:
    res = 0.0

    for population in model.populations:
        # breaks must be ordered to avoid ambiguity
        breaks_lgR = [m.lg_break for m in population.breaks]
        if breaks_lgR != sorted(breaks_lgR):
            return -np.inf

        for brk in population.breaks:
            if not (
                brk.config.lg_break_prior_limits[0]
                < brk.lg_break
                < brk.config.lg_break_prior_limits[1]
            ):
                return -np.inf
            d_alpha_min, d_alpha_max = brk.config.delta_alpha_prior_limits
            if not (d_alpha_min < brk.d_alpha < d_alpha_max):
                return -np.inf
            if brk.config.fixed_lg_sharpness is None:
                s = 10**brk.lg_sharpness
                if not (0.1 < s < 20):
                    return -np.inf

        for cutoff in (population.cutoff, population.cutoff_lower):
            if cutoff is None:
                continue
            if not (
                cutoff.config.lg_cut_prior_limits[0]
                < cutoff.lg_cut
                < cutoff.config.lg_cut_prior_limits[1]
            ):
                return -np.inf

            match cutoff:
                case ExpCutoff() as c:
                    if c.config.fixed_lg_sharpness is None:
                        b = 10**cutoff.lg_sharpness
                        if not (0.1 < b < 20):
                            return -np.inf
                case LognormalSourceMaxAcceleration() as ln:
                    if not 0 < ln.sigma < 10:
                        return -np.inf

        for component in population.base_spectra:
            # ad hoc bound for all spectral normalizations to [10^-20; 10^6];
            # this is roughly +/- 5 orders of magnitude w.r.t. values we find in the fit, so it shouldn't affect
            # the "normal" flux estimation, but it limits the parameter space in cases where a particular spectrum
            # is poorly or not at all constrained by data
            if not all(-15 < lgI < 1 for lgI in component.lgI_per_element.values()):
                return -np.inf

            if (
                component.lg_scale_contrib_to_all is not None
                and component.lg_scale_contrib_to_all < 0
            ):
                return -np.inf

        # other model params
        lgK = population.all_particle_lg_shift
        if lgK is not None and not (1 <= 10**lgK <= 2):
            return -np.inf
        if population.free_Z is not None and not (1 <= population.free_Z <= 26.5):
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


def ensure_model(model_or_theta: Model | np.ndarray, config: ModelConfig) -> Model:
    if isinstance(model_or_theta, Model):
        return model_or_theta
    else:
        return Model.unpack(model_or_theta, layout_info=config)


Chi2Method = Literal["correlated", "dimidated"]
DEFAULT_CHI2_METHOD = os.environ.get("CRKNEE_CHI2_METHOD", "correlated")


def chi_squared_loglikelihood(
    prediction: np.ndarray,
    y: np.ndarray,
    err_stat: np.ndarray,
    err_syst: np.ndarray,
    err_cov_inv: np.ndarray,
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
            result = -0.5 * (residual_vec.T @ err_cov_inv @ residual_vec)
            return float(np.squeeze(result))
        case unexpected:
            raise RuntimeError(
                f"Unexpected chi2 method specified in {'argument' if method is None else 'env var'}: {unexpected}"
            )


DEFAULT_LOGNORMAL_CHI2 = os.environ.get("CRKNEE_LOGNORMAL_CHI2", "1") == "1"


def loglikelihood(
    model_or_theta: Model | np.ndarray,
    fit_data: Data,
    config: ModelConfig,
    chi2_method: Chi2Method | None = None,
    lognormal: bool = DEFAULT_LOGNORMAL_CHI2,
) -> float:
    model = ensure_model(model_or_theta, config)
    res = 0.0

    for spectrum in fit_data.spectra:
        f = model.energy_shifts.f(spectrum.experiment)
        if lognormal:
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
            err_cov_inv=d.standard_inv_err_cov,
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
            err_cov_inv=lnA_data.standard_inv_err_cov,
            method=chi2_method,
        )

    for flux_ratio in fit_data.flux_ratios:
        # flux ratios are used at low energies, where energy scale is very well constrained, hence no shift is applied
        prediction = model.compute_flux_ratio(
            flux_ratio.Q, fr=flux_ratio.ratio, quantity=flux_ratio.quantity
        )
        d = flux_ratio.d
        if lognormal:
            prediction = np.log10(prediction)
            d = d.log10_ed
        res += chi_squared_loglikelihood(
            prediction=prediction,
            y=d.y,
            err_stat=d.err_stat,
            err_syst=d.err_syst,
            err_cov_inv=d.standard_inv_err_cov,
            method=chi2_method,
        )

    return res


# to optimize logposterior evaluation in a multiprocessing setup
# see https://emcee.readthedocs.io/en/stable/tutorials/parallel/#pickling-data-transfer-arguments
fit_data_global: Data | None = None


def set_global_fit_data(fit_data: Data):
    global fit_data_global
    fit_data_global = fit_data


def logposterior(
    model_or_theta: Model | np.ndarray, fit_data: Data | None, config: ModelConfig
) -> float:
    model = ensure_model(model_or_theta, config)
    logpi = logprior(model)
    if not np.isfinite(logpi):
        return logpi
    fit_data_ = fit_data or fit_data_global
    if fit_data_ is None:
        raise ValueError("fit data must be either passed directly or through a global variable")
    res = logpi + loglikelihood(model, fit_data_, config)
    if not np.isfinite(res):
        return -np.inf
    return res
