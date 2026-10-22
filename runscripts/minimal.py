import argparse
from collections.abc import Callable
from typing import Literal

import numpy as np
from crams import InjectionLognormalDist, PropagationParams

from cr_knee_fit import experiments
from cr_knee_fit.analysis import (
    FitConfig,
    McmcConfig,
    PlotsConfig,
)
from cr_knee_fit.cr_models.crams import SHARED_PROPAGATION, CramsModel, init_injection
from cr_knee_fit.cr_models.pheno import (
    PopulationMetadata,
)
from cr_knee_fit.cr_models.utils import Param
from cr_knee_fit.elements import Element
from cr_knee_fit.fit_data import (
    DataConfig,
    FluxRatioDataConfig,
    GenericExperimentData,
    SpectrumDataConfig,
)
from cr_knee_fit.guesses import (
    initial_guess_energy_shifts,
)
from cr_knee_fit.inference import (
    Chi2Method,
    set_default_chi2_method,
    set_default_lognormal_chi2,
)
from cr_knee_fit.local import LocalRunOptions, get_analysis_name, run_local
from cr_knee_fit.model import CRModel, Model
from cr_knee_fit.plotting import PosteriorPlotConfig


def generate_minimal_populations() -> list[CRModel]:
    crams_LE = CramsModel.make(
        to_high_E=True,
        randomize_init=True,
        init_propagation=PropagationParams(
            H_kpc=Param.frozen(7.0),
            D_0_cm2_sec=Param.fitted(3.21e28, 1e26),
            v_A_km_sec=Param.fitted(1.67, 0.1),
            R_b_GV=Param.fitted(10**2.61, 1),
            delta=Param.fitted(0.47, 0.01),
            ddelta=Param.fitted(0.332, 0.01),
            smoothness=Param.frozen(0.1),
            phi=Param.fitted(0.51, 0.05),
            X_src=Param.frozen(-1.0),
        ),
        init_injection=init_injection(
            fitted_abundances={
                Element.H: Param.fitted_relative(0.042, 0.01),
                Element.He: Param.fitted_relative(0.02, 0.01),
                Element.C: Param.fitted_relative(0.004, 0.01),
                Element.N: Param.fitted_relative(0.00053, 0.01),
                Element.O: Param.fitted_relative(0.008, 0.01),
                Element.Ne: Param.fitted_relative(0.0017, 0.01),
                Element.Mg: Param.fitted_relative(0.00312, 0.01),
                Element.Si: Param.fitted_relative(0.0037, 0.01),
                Element.S: Param.fitted_relative(0.00069, 0.01),
                Element.Fe: Param.fitted_relative(0.00914, 0.01),
            },
            slopes=[
                Param.fitted(4.45, 0.01),
                Param.fitted(4.38, 0.01),
                Param.fitted(4.41, 0.01),
            ],
            features=[
                InjectionLognormalDist(
                    R_mean_GV=Param.fitted(10**4.48, 100),
                    sigma=Param.fitted(0.514, 0.01),
                    beta=Param.frozen(1.0),
                )
            ],
        ),
        population_meta=PopulationMetadata(name="LE", linestyle="--"),
    )

    crams_HE = CramsModel.make(
        to_high_E=True,
        randomize_init=True,
        init_propagation=SHARED_PROPAGATION,
        init_injection=init_injection(
            fitted_abundances={
                Element.H: Param.fitted_relative(0.000692, 0.01),
                Element.He: Param.fitted_relative(0.00044, 0.01),
                Element.C: Param.fitted_relative(0.0000377, 0.01),
                Element.O: Param.fitted_relative(0.000036, 0.01),
                Element.Fe: Param.fitted_relative(0.000018, 0.01),
            },
            frozen_abundances={el: 0.0 for el in Element.regular()},
            slopes=[
                Param.fitted(4.1, 0.001),
            ],
            features=[
                InjectionLognormalDist(
                    R_mean_GV=Param.fitted(10**6.4, 1000),
                    sigma=Param.fitted(0.4, 0.02),
                    beta=Param.frozen(1.0),
                )
            ],
        ),
        population_meta=PopulationMetadata(name="HE", linestyle=":"),
    )
    return [crams_LE, crams_HE]


def run(
    analysis_name: str,
    opts: LocalRunOptions,
    lhaaso_him: Literal["sibyll", "qgsjet", "epos"] = "sibyll",
    chi2_method: Chi2Method = "correlated",
    is_lognormal_chi2: bool = True,
    lambda_syst: float = 1.0,
    omit_detailed_plots: bool = False,
    generate_populations: Callable[[], list[CRModel]] = generate_minimal_populations,
):
    print(f"Analysis name: {analysis_name}")

    print(f"Chi2 method: {chi2_method}")
    set_default_chi2_method(chi2_method)

    print(f"Use lognormal Chi2: {is_lognormal_chi2}")
    set_default_lognormal_chi2(is_lognormal_chi2)

    print(f"Correlation length of systematics: {lambda_syst}")
    GenericExperimentData.default_systematics_correlation_length = lambda_syst

    ams02_ratios = [
        Element.B / Element.C,
        Element.B / Element.O,
        Element.C / Element.O,
        Element.H / Element.He,
        Element.He / Element.O,
    ]
    elements = [
        Element.H,
        Element.He,
        Element.B,
        Element.C,
        Element.N,
        Element.O,
        Element.Ne,
        Element.Mg,
        Element.Si,
        Element.S,
        Element.Fe,
    ]

    print(f"Using LHAASO data with HIM: {lhaaso_him}")
    match lhaaso_him:
        case "sibyll":
            lhaaso = experiments.lhaaso_sibyll
        case "epos":
            lhaaso = experiments.lhaaso_epos
        case "qgsjet":
            lhaaso = experiments.lhaaso_qgsjet

    fit_data_config = DataConfig(
        spectra=[
            SpectrumDataConfig(
                experiments.ams02,
                spec=element,
                bounds=(
                    (
                        20.0
                        if element
                        in {Element.Fe, Element.N, Element.Mg, Element.Ne, Element.S, Element.Si}
                        else 10.0
                    ),
                    np.inf,
                ),
            )
            for element in elements
        ]
        + [SpectrumDataConfig(experiments.dampe, element) for element in elements]
        + [SpectrumDataConfig(experiments.calet, element) for element in elements]
        + [
            SpectrumDataConfig(lhaaso, Element.H),
            SpectrumDataConfig(lhaaso, Element.He),
        ],
        flux_ratios=[
            FluxRatioDataConfig(experiments.ams02, ratio, Q_bounds=(10.0, np.inf))
            for ratio in ams02_ratios
        ]
        + [
            FluxRatioDataConfig(experiments.dampe, ratio)
            for ratio in [
                Element.B / Element.C,
                Element.B / Element.O,
            ]
        ],
        lnA=[lhaaso],
    )

    validation_data_config = DataConfig(
        spectra=[
            SpectrumDataConfig.allparticle(experiments.hawc),
            SpectrumDataConfig.allparticle(lhaaso),
            SpectrumDataConfig.allparticle(experiments.kascade_re_qgsjet),
        ]
        + [SpectrumDataConfig(experiments.grapes, Element.H)]
        + [SpectrumDataConfig(experiments.ams02, element) for element in elements]
        + [SpectrumDataConfig(experiments.dampe, element) for element in elements]
        + [SpectrumDataConfig(experiments.calet, element) for element in elements],
        flux_ratios=[
            FluxRatioDataConfig(experiments.calet, ratio)
            for ratio in [Element.B / Element.C, Element.C / Element.O, Element.H / Element.He]
        ],
        lnA=[lhaaso, experiments.tale],
    ).excluding(fit_data_config)

    def generate_guess() -> Model:
        return Model(
            populations=generate_populations(),
            energy_shifts=initial_guess_energy_shifts(
                fit_data_config.experiments_spectra,
                fixed=experiments.ams02,
            ),
        )

    config = FitConfig.from_guessing_func(
        name=analysis_name,
        fit_data=fit_data_config,
        mcmc=McmcConfig(
            n_steps=30000,
            n_walkers=128,
            processes=11,
            reuse_saved=True,
            runtime_thinning=100,
        ),
        generate_guess=generate_guess,
        plots=PlotsConfig(
            validation_data_config=validation_data_config,
            datasets=not omit_detailed_plots,
            observables=not omit_detailed_plots,
            corner=not omit_detailed_plots,
            elements=PosteriorPlotConfig(
                ylim_override=(3e3, 5e4),
                xlim_override=(1e1, 5e7),
                population_contribs_best_fit=True,
            ),
            observables_posterior=PosteriorPlotConfig(
                max_margin_around_data=0.5,
                margin_around_fitted_data_only=True,
            ),
        ),
    )
    config.optimizer = "minuit"

    run_local(config=config, opts=opts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-fast-plots", action="store_true")
    opts = LocalRunOptions.parse(parser)
    analysis_name = get_analysis_name(__file__, opts)

    run(
        analysis_name=analysis_name,
        opts=opts,
        omit_detailed_plots=opts.args_raw.only_fast_plots,
    )
