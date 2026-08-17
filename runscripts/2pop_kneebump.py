import numpy as np
from scipy import stats  # type: ignore

from cr_knee_fit import experiments
from cr_knee_fit.analysis import (
    FitConfig,
    PlotsConfig,
)
from cr_knee_fit.cr_model import (
    CosmicRaysModel,
    LognormalSourceMaxAccelerationConfig,
    PopulationMetadata,
    SharedPowerLawSpectrum,
)
from cr_knee_fit.crams_model import CramsModel
from cr_knee_fit.elements import Element
from cr_knee_fit.fit_data import DataConfig, SpectrumDataConfig
from cr_knee_fit.guesses import (
    initial_guess_cutoff,
    initial_guess_energy_shifts,
    initial_guess_pl_index,
)
from cr_knee_fit.local import LocalRunOptions, guess_analysis_name, run_local
from cr_knee_fit.model import Model

if __name__ == "__main__":
    opts = LocalRunOptions.parse()
    analysis_name = guess_analysis_name(__file__)

    ratios = [
        Element.B / Element.C,
        Element.B / Element.O,
        Element.C / Element.O,
        Element.H / Element.He,
        Element.He / Element.O,
    ]
    elements = [Element.H, Element.He]

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
    )

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
            up2PeV=True, source_feature="erfc-cutoff", freeze_low_energy_params=True
        )
        crams.config.population_meta = PopulationMetadata(name="LE", linestyle="--")
        return Model(
            populations=[
                CosmicRaysModel(
                    base_spectra=[
                        (
                            SharedPowerLawSpectrum(
                                lgI_per_element={
                                    Element.H: stats.norm.rvs(loc=-4.25, scale=0.05),
                                    Element.He: stats.norm.rvs(loc=-4.8, scale=0.05),
                                },
                                alpha=initial_guess_pl_index(center=2.6),
                            )
                        )
                    ],
                    cutoff=initial_guess_cutoff(
                        LognormalSourceMaxAccelerationConfig(
                            lg_cut_hint=6.5,
                            lg_cut_prior_limits=(5, 8),
                            fixed_beta=1.0,
                        )
                    ),
                    # cutoff_lower=(  # type: ignore
                    #     initial_guess_cutoff(
                    #         ExpCutoffConfig(
                    #             lg_cut_prior_limits=(3, 6),
                    #             lg_cut_hint=5,
                    #         )
                    #     )
                    # ),
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
