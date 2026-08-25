import argparse
from typing import Literal

import numpy as np
from crams import LognormalRmaxDistribution, PropagationParams
from scipy import stats  # type: ignore

from cr_knee_fit import experiments
from cr_knee_fit.analysis import (
    FitConfig,
    McmcConfig,
    PlotsConfig,
)
from cr_knee_fit.cr_model import (
    CosmicRaysModel,
    ExpCutoff,
    ExpCutoffConfig,
    LognormalSourceMaxAcceleration,
    LognormalSourceMaxAccelerationConfig,
    PopulationMetadata,
    SharedPowerLawSpectrum,
    omega2lg_sharpness,
)
from cr_knee_fit.crams_model import FREE, CramsModel
from cr_knee_fit.elements import Element
from cr_knee_fit.fit_data import (
    DataConfig,
    FluxRatioDataConfig,
    GenericExperimentData,
    SpectrumDataConfig,
)
from cr_knee_fit.guesses import (
    initial_guess_cutoff,
    initial_guess_energy_shifts,
    initial_guess_pl_index,
)
from cr_knee_fit.inference import Chi2Method, set_default_chi2_method, set_default_lognormal_chi2
from cr_knee_fit.local import LocalRunOptions, get_analysis_name, run_local
from cr_knee_fit.model import Model
from cr_knee_fit.plotting import PosteriorPlotConfig


def run_2_or_3_pop(
    analysis_name: str,
    opts: LocalRunOptions,
    add_population_3: bool,
    lhaaso_him: Literal["sibyll", "qgsjet", "epos"] = "sibyll",
    chi2_method: Chi2Method = "correlated",
    is_lognormal_chi2: bool = True,
    lambda_syst: float = 1.0,
    omit_detailed_plots: bool = False,
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
            SpectrumDataConfig(lhaaso, Element.H),  # , bounds=(3e5, np.inf)),
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
        lnA=[lhaaso],
    ).excluding(fit_data_config)

    def generate_guess() -> Model:
        crams = CramsModel.make(
            up2PeV=True,
            randomize_init=True,
            freeze_propagation=PropagationParams(
                H_kpc=7.0,
                v_A_km_sec=0.0001,
                R_b_GV=FREE,
                delta=0.469,
                ddelta=FREE,
                D_0_cm2_sec=FREE,
                X_src=-1.0,
                phi=0.631,
            ),
            source_feature=LognormalRmaxDistribution(
                R_mean_GV=FREE,
                sigma=FREE,
                beta=1.0,
            ),
        )
        crams.config.population_meta = PopulationMetadata(name="LE", linestyle="--")

        # manually setting things very close to MLE to cut down on convergence time
        crams.propagation.R_b_GV = 10 ** stats.norm.rvs(2.5, 0.01)
        crams.propagation.ddelta = stats.norm.rvs(0.284, 0.05)
        crams.propagation.D_0_cm2_sec = 1e28 * stats.norm.rvs(3.51, 0.01)
        for element, abundance in [
            (Element.H, stats.norm.rvs(0.0463, 0.000463)),
            (Element.He, stats.norm.rvs(0.0227, 0.000227)),
            (Element.C, stats.norm.rvs(0.00459, 0.0000459)),
            (Element.N, stats.norm.rvs(0.000526, 0.00000526)),
            (Element.O, stats.norm.rvs(0.00846, 0.0000846)),
            (Element.Ne, stats.norm.rvs(0.0017, 0.000017)),
            (Element.Mg, stats.norm.rvs(0.00307, 0.0030007)),
            (Element.Si, stats.norm.rvs(0.00366, 0.0000366)),
            (Element.S, stats.norm.rvs(0.000669, 0.00000669)),
            (Element.Fe, stats.norm.rvs(0.00949, 0.0000949)),
        ]:
            crams.injection.set_abundance(round(element.Z), abundance)
        crams.injection.slopes = [
            stats.norm.rvs(4.41, 0.01),
            stats.norm.rvs(4.35, 0.01),
            stats.norm.rvs(4.4, 0.01),
        ]
        match crams.injection.feature:
            case LognormalRmaxDistribution() as d:
                d.R_mean_GV = 10 ** stats.norm.rvs(4.48, 0.01)
                d.sigma = stats.norm.rvs(0.514, 0.01)

        populations = [
            CosmicRaysModel(
                base_spectra=[
                    SharedPowerLawSpectrum(
                        lgI_per_element={
                            Element.H: stats.norm.rvs(loc=-12.1, scale=0.05),
                            Element.He: stats.norm.rvs(loc=-12.6, scale=0.05),
                            Element.C: stats.norm.rvs(loc=-13.8, scale=0.05),
                            Element.O: stats.norm.rvs(loc=-14.1, scale=0.05),
                            Element.Fe: stats.norm.rvs(loc=-15.3, scale=0.05),
                        },
                        alpha=initial_guess_pl_index(center=2.46, width=0.01),
                        R0=1e6,
                    ),
                    # SharedPowerLawSpectrum(
                    #     lgI_per_element={
                    #         Element.He: stats.norm.rvs(loc=-4.8 - 7.8, scale=0.05),
                    #     },
                    #     alpha=initial_guess_pl_index(center=2.3),
                    #     R0=1e6,
                    # ),
                ],
                cutoff=LognormalSourceMaxAcceleration(
                    lg_cut=stats.norm.rvs(6.59, 0.01),
                    sigma=stats.norm.rvs(0.379, 0.01),
                    beta=1.0,
                    config=LognormalSourceMaxAccelerationConfig(
                        lg_cut_hint=6.59,
                        lg_cut_prior_limits=(5.5, 8),
                        fixed_beta=1.0,
                    ),
                ),
                cutoff_lower=(
                    ExpCutoff(
                        lg_cut=stats.norm.rvs(3.75, 0.05),
                        lg_sharpness=stats.uniform.rvs(
                            omega2lg_sharpness(1.1), omega2lg_sharpness(1.0)
                        ),
                        config=ExpCutoffConfig(
                            lg_cut_prior_limits=(3, 6),
                            lg_cut_hint=5,
                            lg_sharpness_prior_limits=(
                                omega2lg_sharpness(3),
                                omega2lg_sharpness(1.0),
                            ),
                        ),
                    )
                ),
                population_meta=PopulationMetadata(name="HE", linestyle=":"),
            )
        ]

        if add_population_3:
            populations.append(
                CosmicRaysModel(
                    base_spectra=[
                        SharedPowerLawSpectrum(
                            lgI_per_element={
                                Element.H: stats.norm.rvs(loc=-11.5, scale=0.05),
                                Element.He: stats.norm.rvs(loc=-11.8, scale=0.05),
                                # Element.C: stats.norm.rvs(loc=-12.5, scale=0.05),
                                # Element.O: stats.norm.rvs(loc=-12.5, scale=0.05),
                                # Element.Fe: stats.norm.rvs(loc=-14.0, scale=0.05),
                            },
                            alpha=initial_guess_pl_index(center=2.0),
                            R0=1e6,
                        ),
                    ],
                    cutoff=initial_guess_cutoff(
                        ExpCutoffConfig(
                            lg_cut_prior_limits=(3, 5.5),
                            lg_cut_hint=4.5,
                        )
                    ),
                    population_meta=PopulationMetadata(name="LS", linestyle="-."),
                )
            )

        return Model(
            populations=populations,
            crams=crams,
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
            n_walkers=256,
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
        ),
    )
    config.optimizer = "minuit"

    run_local(config=config, opts=opts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--three-pop", action="store_true")
    parser.add_argument("--only-fast-plots", action="store_true")
    opts = LocalRunOptions.parse(parser)
    analysis_name = get_analysis_name(__file__, opts)
    is_3pop = opts.args_raw.three_pop
    if is_3pop:
        analysis_name = analysis_name.replace("two", "three")

    run_2_or_3_pop(
        analysis_name=analysis_name,
        opts=opts,
        add_population_3=is_3pop,
        omit_detailed_plots=opts.args_raw.only_fast_plots,
    )
