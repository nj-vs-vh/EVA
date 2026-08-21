from argparse import ArgumentParser

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
    ExpCutoffConfig,
    LognormalSourceMaxAccelerationConfig,
    PopulationMetadata,
    SharedPowerLawSpectrum,
    omega2lg_sharpness,
)
from cr_knee_fit.crams_model import FREE, CramsModel
from cr_knee_fit.elements import Element
from cr_knee_fit.fit_data import DataConfig, FluxRatioDataConfig, SpectrumDataConfig
from cr_knee_fit.guesses import (
    initial_guess_cutoff,
    initial_guess_energy_shifts,
    initial_guess_pl_index,
)
from cr_knee_fit.local import LocalRunOptions, guess_analysis_name, run_local
from cr_knee_fit.model import Model
from cr_knee_fit.plotting import PosteriorPlotConfig

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--low-cut", action="store_true")
    opts = LocalRunOptions.parse(p)
    low_cut: bool = opts.args_raw.low_cut
    analysis_name = guess_analysis_name(__file__, opts)

    if low_cut:
        analysis_name += "_lowcut"

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

    lhaaso = experiments.lhaaso_sibyll

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
        # + [SpectrumDataConfig.allparticle(lhaaso)]
        + [SpectrumDataConfig(lhaaso, element) for element in elements],
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
        # mask_out_elements={Element.He: (1e5, 1e6)},
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

        crams.propagation.D_0_cm2_sec += 1e28  # init value compensation for HE population tail

        return Model(
            populations=[
                CosmicRaysModel(
                    base_spectra=[
                        SharedPowerLawSpectrum(
                            lgI_per_element={
                                Element.H: stats.norm.rvs(loc=-4.25 - 7.8, scale=0.05),
                                Element.He: stats.norm.rvs(loc=-4.8 - 7.8, scale=0.05),
                                Element.C: stats.norm.rvs(loc=-6.8 - 7.8, scale=0.05),
                                Element.O: stats.norm.rvs(loc=-6.8 - 7.8, scale=0.05),
                                Element.Fe: stats.norm.rvs(loc=-8.0 - 7.8, scale=0.05),
                            },
                            alpha=initial_guess_pl_index(center=2.3),
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
                    cutoff=initial_guess_cutoff(
                        LognormalSourceMaxAccelerationConfig(
                            lg_cut_hint=6.65,
                            lg_cut_prior_limits=(5, 8),
                            fixed_beta=1.0,
                        )
                    ),
                    cutoff_lower=(
                        initial_guess_cutoff(
                            ExpCutoffConfig(
                                lg_cut_prior_limits=(3, 6),
                                lg_cut_hint=5,
                                lg_sharpness_prior_limits=(
                                    omega2lg_sharpness(3),
                                    omega2lg_sharpness(1.0),
                                ),
                            )
                        )
                        if low_cut
                        else None
                    ),
                    population_meta=PopulationMetadata(name="HE", linestyle=":"),
                )
            ],
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
            n_steps=20000,
            n_walkers=256,
            processes=11,
            reuse_saved=True,
            runtime_thinning=100,
        ),
        generate_guess=generate_guess,
        plots=PlotsConfig(
            validation_data_config=validation_data_config,
            elements=PosteriorPlotConfig(
                ylim_override=(3e3, 5e4),
                population_contribs_best_fit=True,
            ),
            corner=False,
        ),
    )
    config.optimizer = "minuit"

    run_local(
        config=config,
        opts=opts,
    )
