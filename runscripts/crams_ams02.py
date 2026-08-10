import numpy as np

from cr_knee_fit import experiments
from cr_knee_fit.analysis import (
    FitConfig,
    PlotsConfig,
)
from cr_knee_fit.crams_model import CramsModel
from cr_knee_fit.elements import Element
from cr_knee_fit.fit_data import DataConfig, FluxRatio, SpectrumDataConfig
from cr_knee_fit.guesses import (
    initial_guess_energy_shifts,
)
from cr_knee_fit.local import LocalRunOptions, guess_analysis_name, run_local
from cr_knee_fit.model import Model

if __name__ == "__main__":
    opts = LocalRunOptions.parse()
    analysis_name = guess_analysis_name(__file__)

    fit_data_config = DataConfig(
        spectra=[
            SpectrumDataConfig(experiments.ams02, spec=element, bounds=(5, np.inf))
            for element in Element.regular()
        ],
        flux_ratios=[
            (experiments.ams02, FluxRatio(Element.H, Element.He)),
            (experiments.ams02, FluxRatio(Element.Li, Element.B)),
            (experiments.ams02, FluxRatio(Element.B, Element.C)),
            (experiments.ams02, FluxRatio(Element.B, Element.O)),
            (experiments.ams02, FluxRatio(Element.Be, Element.B)),
            (experiments.ams02, FluxRatio(Element.C, Element.O)),
            (experiments.ams02, FluxRatio(Element.Fe, Element.O)),
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
            (experiments.calet, FluxRatio(Element.H, Element.He)),
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
