import itertools

import numpy as np

from cr_knee_fit import experiments
from cr_knee_fit.analysis import (
    FitConfig,
    PlotsConfig,
)
from cr_knee_fit.crams_model import CramsModel
from cr_knee_fit.elements import Element
from cr_knee_fit.fit_data import DataConfig, FluxRatio, FluxRatioDataConfig, SpectrumDataConfig
from cr_knee_fit.guesses import (
    initial_guess_energy_shifts,
)
from cr_knee_fit.local import LocalRunOptions, guess_analysis_name, run_local
from cr_knee_fit.model import Model

if __name__ == "__main__":
    opts = LocalRunOptions.parse()
    analysis_name = guess_analysis_name(__file__)

    # this is the exact setup from CRAMS fit, with the same elements, ratios, and R bounds

    ratios_of_interest = [
        FluxRatio(Element.B, Element.C),
        FluxRatio(Element.B, Element.O),
        FluxRatio(Element.C, Element.O),
        FluxRatio(Element.H, Element.He),
        FluxRatio(Element.He, Element.O),
    ]
    elements_fitted = [
        Element.Fe,
        Element.H,
        Element.Mg,
        Element.N,
        Element.Ne,
        Element.O,
        Element.S,
        Element.Si,
    ]
    elements_constrained_through_ratios = [
        Element.B,  # constrained through B/O
        Element.C,  # constrained through C/O
        Element.He,  # constrained through H/He
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
            for element in elements_fitted
        ],
        flux_ratios=(
            [
                FluxRatioDataConfig(experiments.ams02, ratio, Q_bounds=(10.0, np.inf))
                for ratio in ratios_of_interest
            ]
            + [
                FluxRatioDataConfig(experiments.dampe, ratio, Q_bounds=(0.0, np.inf))
                for ratio in ratios_of_interest
            ]
        ),
    )

    validation_data_config = DataConfig(
        spectra=[
            SpectrumDataConfig.allparticle(experiments.hawc),
        ]
        + [
            SpectrumDataConfig(experiments.dampe, element)
            for element in elements_constrained_through_ratios + elements_fitted
        ]
        + [
            SpectrumDataConfig(experiments.ams02, element)
            for element in elements_constrained_through_ratios
        ],
        flux_ratios=[
            FluxRatioDataConfig(exp, ratio)
            for ratio, exp in itertools.product(ratios_of_interest, [experiments.calet])
        ],
    ).excluding(fit_data_config)

    def generate_guess() -> Model:
        return Model(
            populations=[],
            energy_shifts=initial_guess_energy_shifts(
                fit_data_config.experiments_spectra,
                fixed=experiments.ams02,
            ),
            crams=CramsModel.default(include_10TeV_break=False),
        )

    # m = generate_guess()
    # d = Data.load(fit_data_config, verbose=True)
    # print()
    # vd = Data.load(validation_data_config, verbose=True)
    # m.plot_spectra(d, scale=2.7, validation_data=vd).savefig("temp_spectra.png")
    # m.plot_flux_ratios(d, validation_data=vd).savefig("temp_ratios.png")

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
