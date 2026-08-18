import numpy as np
from scipy import stats  # type: ignore

from cr_knee_fit import experiments
from cr_knee_fit.analysis import (
    FitConfig,
    PlotsConfig,
)
from cr_knee_fit.cr_model import (
    CosmicRaysModel,
    ExpCutoffConfig,
    LognormalSourceMaxAccelerationConfig,
    PopulationMetadata,
    SharedPowerLawSpectrum,
)
from cr_knee_fit.crams_model import FREE, CramsModel
from cr_knee_fit.elements import Element
from cr_knee_fit.fit_data import DataConfig, FluxRatioDataConfig, SpectrumDataConfig
from cr_knee_fit.guesses import (
    initial_guess_cutoff,
    initial_guess_energy_shifts,
    initial_guess_pl_index,
)
from cr_knee_fit.local import LocalRunOptions, get_analysis_name, run_local
from cr_knee_fit.model import Model

if __name__ == "__main__":
    opts = LocalRunOptions.parse()
    analysis_name = get_analysis_name(__file__, opts)

    elements = [
        Element.Fe,
        Element.H,
        Element.Mg,
        Element.N,
        Element.Ne,
        Element.O,
        Element.S,
        Element.Si,
        Element.B,
        Element.C,
        Element.He,
    ]

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
        + [SpectrumDataConfig(experiments.lhaaso_qgsjet, element) for element in elements]
        + [
            SpectrumDataConfig(experiments.kascade_re_qgsjet, element)
            for element in [Element.C, Element.Si, Element.Fe]
        ],
        flux_ratios=[
            FluxRatioDataConfig(experiments.dampe, Element.B / Element.C),
            FluxRatioDataConfig(experiments.dampe, Element.B / Element.O),
            FluxRatioDataConfig(experiments.ams02, Element.B / Element.C),
            FluxRatioDataConfig(experiments.ams02, Element.B / Element.O),
            FluxRatioDataConfig(experiments.ams02, Element.C / Element.O),
            FluxRatioDataConfig(experiments.ams02, Element.H / Element.He),
            FluxRatioDataConfig(experiments.ams02, Element.He / Element.O),
        ],
    )
    fit_data_config.remove_subdominant_spectra_constrained_by_flux_ratios()

    validation_data_config = DataConfig(
        spectra=[
            SpectrumDataConfig.allparticle(experiments.hawc),
            SpectrumDataConfig.allparticle(experiments.lhaaso_qgsjet),
        ]
        + [SpectrumDataConfig(experiments.calet, element) for element in elements]
        + [
            SpectrumDataConfig(experiments.kascade_re_qgsjet, element)
            for element in Element.regular()
        ],
        lnA=[experiments.lhaaso_qgsjet],
    ).excluding(fit_data_config)

    def generate_guess() -> Model:
        crams = CramsModel.make(
            up2PeV=True,
            source_feature="erfc-cutoff",
            freeze_propagation=True,
        )
        crams.config.population_meta = PopulationMetadata(name="LE", linestyle="--")
        crams.config.frozen_propagation.D_0_cm2_sec = FREE
        crams.config.frozen_propagation.ddelta = FREE

        return Model(
            populations=[
                CosmicRaysModel(
                    base_spectra=[
                        (
                            SharedPowerLawSpectrum(
                                lgI_per_element={
                                    Element.H: stats.norm.rvs(loc=-4.8, scale=0.05),
                                    Element.He: stats.norm.rvs(loc=-5.4, scale=0.05),
                                    Element.C: stats.norm.rvs(loc=-6.6, scale=0.05),
                                    Element.Si: stats.norm.rvs(loc=-7.2, scale=0.05),
                                    Element.Fe: stats.norm.rvs(loc=-8.0, scale=0.05),
                                },
                                alpha=initial_guess_pl_index(center=2.4, width=0.00001),
                            )
                        )
                    ],
                    cutoff=initial_guess_cutoff(
                        LognormalSourceMaxAccelerationConfig(
                            lg_cut_hint=6.6,
                            lg_cut_prior_limits=(5, 8),
                            fixed_beta=1.0,
                        )
                    ),
                    cutoff_lower=(
                        initial_guess_cutoff(
                            ExpCutoffConfig(
                                lg_cut_prior_limits=(2, 6),
                                lg_cut_hint=4.5,
                            )
                        )
                    ),
                    population_meta=PopulationMetadata(name="HE", linestyle=":"),
                )
            ],
            crams=crams,
            energy_shifts=initial_guess_energy_shifts(
                # fit_data_config.experiments_spectra,
                [],
                fixed=experiments.ams02,
            ),
        )

    config = FitConfig.from_guessing_func(
        name=analysis_name,
        fit_data=fit_data_config,
        mcmc=None,
        generate_guess=generate_guess,
        plots=PlotsConfig(
            validation_data_config=validation_data_config,
        ),
    )
    config.optimizer = "minuit"

    run_local(
        config=config,
        opts=opts,
    )
