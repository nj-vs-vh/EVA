import numpy as np
from scipy import stats  # type: ignore

from cr_knee_fit import experiments
from cr_knee_fit.fit_data import Data
from cr_knee_fit.model_ import Model, ModelConfig

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
            if (brk.d_alpha > 0) != brk.config.is_softening:
                return -np.inf
            if brk.config.fixed_lg_sharpness is None:
                s = 10**brk.lg_sharpness
                if not (0.1 < s < 20):
                    return -np.inf

        if population.cutoff is not None:
            cutoff = population.cutoff
            if not (
                cutoff.config.lg_cut_prior_limits[0]
                < cutoff.lg_cut
                < cutoff.config.lg_cut_prior_limits[1]
            ):
                return -np.inf
            if cutoff.config.fixed_lg_sharpness is None:
                b = 10**cutoff.lg_sharpness
                if not (0.1 < b < 20):
                    return -np.inf

        for component in population.base_spectra:
            # ad hoc bound for all spectral normalizations to [10^-20; 10^6];
            # this is roughly +/- 10 orders of magnitude w.r.t. values we find in the fit, so it shouldn't affect
            # the "normal" flux estimation, but it limits the parameter space in cases where a particular spectrum
            # is poorly or not at all constrained by data
            if not all(-20 < lgI < 6 for lgI in component.lgI_per_element.values()):
                return -np.inf

            if (
                component.lg_scale_contrib_to_all is not None
                and component.lg_scale_contrib_to_all < 0
            ):
                return -np.inf

        # other model params
        lgK = population.all_particle_lg_shift
        if lgK is not None:
            if not (1 <= 10**lgK <= 2):
                return -np.inf
        if population.free_Z is not None:
            if not (1 <= population.free_Z <= 26.5):
                return -np.inf

    # experimental energy shifts' prior
    for exp, lg_shift in model.energy_shifts.lg_shifts.items():
        lg_sigma = get_energy_scale_lg_uncertainty(exp)
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


def chi_squared_loglikelihood(
    prediction: np.ndarray, y: np.ndarray, errlo: np.ndarray, errhi: np.ndarray
) -> float:
    residual = prediction - y
    loglike_per_bin = -0.5 * (
        np.where(residual > 0, (residual / errhi) ** 2, (residual / errlo) ** 2)
    )
    return float(np.sum(loglike_per_bin))


def loglikelihood(
    model_or_theta: Model | np.ndarray,
    fit_data: Data,
    config: ModelConfig,
) -> float:
    model = to_model(model_or_theta, config)
    res = 0.0
    for exp, data_by_element in fit_data.element_spectra.items():
        for element, el_data in data_by_element.items():
            el_data = el_data.with_shifted_energy_scale(f=model.energy_shifts.f(exp))
            res += chi_squared_loglikelihood(
                prediction=model.compute_spectrum(el_data.E, element=element),
                y=el_data.F,
                errhi=el_data.F_errhi,
                errlo=el_data.F_errlo,
            )
    for exp, all_data in fit_data.all_particle_spectra.items():
        all_data = all_data.with_shifted_energy_scale(f=model.energy_shifts.f(exp))
        res += chi_squared_loglikelihood(
            prediction=model.compute_spectrum(all_data.E, element=None),
            y=all_data.F,
            errhi=all_data.F_errhi,
            errlo=all_data.F_errlo,
        )
    for exp, lnA_data in fit_data.lnA.items():
        f = model.energy_shifts.f(exp)
        E_exp = lnA_data.x
        # for lnA, energy scale shift does not affect values as they include dE in num. and denom.
        E_true = E_exp * f
        res += chi_squared_loglikelihood(
            prediction=model.compute_lnA(E_true),
            y=lnA_data.y,
            errhi=lnA_data.y_errhi,
            errlo=lnA_data.y_errlo,
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
    model = to_model(model_or_theta, config)
    logpi = logprior(model)
    if not np.isfinite(logpi):
        return logpi
    fit_data_ = fit_data or fit_data_global
    if fit_data_ is None:
        raise ValueError("fit data must be either passed directly or through a global variable")
    return logpi + loglikelihood(model, fit_data_, config)
