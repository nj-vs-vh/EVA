from argparse import ArgumentParser

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
    p = ArgumentParser()
    p.add_argument("--feature", default="break")

    opts = LocalRunOptions.parse(p)
    analysis_name = guess_analysis_name(__file__)

    source_feature = opts.args_raw.feature
    analysis_name += f"_feature_{source_feature}"
    print(f"Running with feature={source_feature}, full analysis name: {analysis_name}")

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
        ]
        + [
            SpectrumDataConfig(experiments.dampe, element)
            for element in elements_constrained_through_ratios + elements_fitted
        ],
        flux_ratios=(
            [
                FluxRatioDataConfig(experiments.ams02, ratio, Q_bounds=(10.0, np.inf))
                for ratio in ratios_of_interest
            ]
            + [FluxRatioDataConfig(experiments.dampe, ratio) for ratio in ratios_of_interest]
            + [FluxRatioDataConfig(experiments.calet, ratio) for ratio in ratios_of_interest]
        ),
    )

    validation_data_config = DataConfig(
        spectra=[
            SpectrumDataConfig(experiments.ams02, element, (10, np.inf))
            for element in elements_constrained_through_ratios
        ]
        + [
            SpectrumDataConfig.allparticle(experiments.hawc),
            SpectrumDataConfig.allparticle(experiments.lhaaso_qgsjet),
        ]
        + [SpectrumDataConfig(experiments.lhaaso_qgsjet, element) for element in Element.regular()]
        + [
            SpectrumDataConfig(experiments.kascade_re_qgsjet, element)
            for element in Element.regular()
        ],
    ).excluding(fit_data_config)

    def generate_guess() -> Model:
        return Model(
            populations=[],
            energy_shifts=initial_guess_energy_shifts(
                fit_data_config.experiments_spectra,
                fixed=experiments.ams02,
            ),
            crams=CramsModel.default(up2PeV=True, source_feature=source_feature),
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
