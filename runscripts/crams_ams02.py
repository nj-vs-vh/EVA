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
                        else 5.0
                    ),
                    np.inf,
                ),
            )
            for element in [
                Element.B,
                Element.C,
                Element.Fe,
                Element.H,
                Element.He,
                Element.Mg,
                Element.N,
                Element.Ne,
                Element.O,
                Element.S,
                Element.Si,
            ]
        ],
        flux_ratios=[
            FluxRatioDataConfig(
                experiments.ams02, FluxRatio(Element.B, Element.C), R_bounds=(5.0, np.inf)
            ),
            FluxRatioDataConfig(
                experiments.ams02, FluxRatio(Element.B, Element.O), R_bounds=(5.0, np.inf)
            ),
            FluxRatioDataConfig(
                experiments.ams02, FluxRatio(Element.C, Element.O), R_bounds=(5.0, np.inf)
            ),
            FluxRatioDataConfig(
                experiments.ams02, FluxRatio(Element.H, Element.He), R_bounds=(5.0, np.inf)
            ),
            FluxRatioDataConfig(
                experiments.ams02, FluxRatio(Element.He, Element.O), R_bounds=(5.0, np.inf)
            ),
            FluxRatioDataConfig(
                experiments.ams02, FluxRatio(Element.He, Element.O), R_bounds=(5.0, np.inf)
            ),
        ],
    )

    validation_data_config = DataConfig(
        spectra=[
            SpectrumDataConfig.allparticle(experiments.hawc),
            SpectrumDataConfig(experiments.dampe, Element.H),
            SpectrumDataConfig(experiments.dampe, Element.He),
            SpectrumDataConfig(experiments.dampe, Element.Fe),
        ],
        flux_ratios=[
            FluxRatioDataConfig(experiments.calet, FluxRatio(Element.H, Element.He)),
        ],
    )

    def generate_guess() -> Model:
        return Model(
            populations=[],
            energy_shifts=initial_guess_energy_shifts(
                fit_data_config.experiments_spectra,
                fixed=experiments.ams02,
            ),
            crams=CramsModel.default(),
        )

    # m = generate_guess()
    # d = Data.load(fit_data_config, verbose=True)
    # print()
    # vd = Data.load(validation_data_config, verbose=True)
    # m.plot_spectra(d, scale=2.7, validation_data=vd).savefig("temp_spectra.png")
    # m.plot_flux_ratios(d, validation_data=vd).savefig("temp_ratios.png")

    run_local(
        config=FitConfig.from_guessing_func(
            name=analysis_name,
            fit_data=fit_data_config,
            mcmc=None,
            generate_guess=generate_guess,
            plots=PlotsConfig(
                validation_data_config=validation_data_config,
            ),
        ),
        opts=opts,
    )
