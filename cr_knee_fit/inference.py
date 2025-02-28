import numpy as np
from scipy import stats  # type: ignore

from cr_knee_fit import experiments
from cr_knee_fit.cr_model import SpectralBreak
from cr_knee_fit.fit_data import FitData
from cr_knee_fit.model_ import Model, ModelConfig


def break_prior(
    b: SpectralBreak,
    lg_break_min: float,
    lg_break_max: float,
    is_softening: bool,
) -> float:
    res = 0
    if not (lg_break_min < b.lg_break < lg_break_max):
        return -np.inf
    if (b.d_alpha > 0) != is_softening:
        return -np.inf
    if not b.fix_sharpness:
        s = 10**b.lg_sharpness
        if not (0.1 < s < 20):
            return -np.inf
    return res


def lg_from_percent(percent: float) -> float:
    return np.log10(1 + percent / 100)


energy_scale_lg_uncertainties = {
    # 10.1103/PhysRevLett.113.121101
    # > This comparison limits the uncertainty of the absolute energy scale to 2% in the range covered by
    #   the beam test results, 10–290 GeV. It increases to 5% at 0.5 GeV and to 3% at 500 GeV.
    experiments.ams02: lg_from_percent(3),
    # 10.1103/PhysRevLett.119.181101
    # > It is found that the average ratio of the expected to measured cutoff position in the electron flux
    #   is 1.035 +/- 0.009 (stat). As a result, a correction of the energy scale by 3.5% was implemented in
    #   the analysis.
    experiments.calet: lg_from_percent(0.9),
    # 10.22323/1.301.0197
    # > we provide an estimation on absolute energy scale of DAMPE 1.25% higher than expected at about 13GeV
    #   energy with uncertainty about ±1.75%(stat)±1.34%(sys)
    # combining 1.75 and 1.34 in lg quadratures, we get 2.2
    experiments.dampe: lg_from_percent(2.2),
    # no direct statement found on energy scale uncertainty found, but arXiV:2004.10371 lists energy resolution
    # at a level of 6.4% at 150 GeV, so we take the same value for energy scale uncertainty
    experiments.cream: lg_from_percent(6.4),
    experiments.iss_cream: lg_from_percent(6.4),
    # arXiv:1710.00890
    # no direct statement on energy scale uncertainty, but 10% is used as a plausible shift
    experiments.hawc: lg_from_percent(10),
    # 10.1016/j.nima.2004.11.025
    # no energy scale data found, but energy resolution is said to be ~10%
    experiments.grapes: lg_from_percent(10),
    # 10.1103/PhysRevD.104.062007 and 10.1051/epjconf/202328302002
    # > The uncertainty of 30% is statistics dominant in the measurement of the shift.
    experiments.lhaaso_epos: lg_from_percent(30),
    experiments.lhaaso_qgsjet: lg_from_percent(30),
    experiments.lhaaso_sibyll: lg_from_percent(30),
}


def logprior(model: Model) -> float:
    res = 0.0

    # main population prior

    main = model.populations[0]

    # breaks must be ordered in R to avoid ambiguity
    breaks_lgR = [m.lg_break for m in main.breaks]
    if breaks_lgR != sorted(breaks_lgR):
        return -np.inf

    # dampe break
    res += break_prior(main.breaks[0], lg_break_min=3, lg_break_max=5, is_softening=True)
    if len(main.breaks) > 1:
        # grapes hardening
        res += break_prior(main.breaks[1], lg_break_min=4.5, lg_break_max=6, is_softening=False)
    if len(main.breaks) > 2:
        # all-particle knee
        res += break_prior(main.breaks[2], lg_break_min=5.5, lg_break_max=7, is_softening=True)

    # components prior
    for component in main.base_spectra:
        if component.lg_scale_contrib_to_all is not None and component.lg_scale_contrib_to_all < 0:
            return -np.inf

    # other model params
    lgK = main.all_particle_lg_shift
    if lgK is not None:
        if not (1 <= 10**lgK <= 2):
            return -np.inf
    if main.free_Z is not None:
        if not (1 <= main.free_Z <= 26.5):
            return -np.inf

    # experimental energy shifts' prior
    for exp, lg_shift in model.energy_shifts.lg_shifts.items():
        lg_sigma = energy_scale_lg_uncertainties.get(exp, lg_from_percent(10))
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
    fit_data: FitData,
    config: ModelConfig,
) -> float:
    model = to_model(model_or_theta, config)
    res = 0.0
    for exp, data_by_element in fit_data.spectra.items():
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
