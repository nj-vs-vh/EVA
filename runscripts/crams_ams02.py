import numpy as np

from cr_knee_fit import experiments
from cr_knee_fit.analysis import (
    FitConfig,
    PlotsConfig,
)
from cr_knee_fit.crams_model import CramsModel
from cr_knee_fit.elements import Element
from cr_knee_fit.fit_data import DataConfig, FluxRatio
from cr_knee_fit.guesses import (
    initial_guess_energy_shifts,
)
from cr_knee_fit.local import LocalRunOptions, guess_analysis_name, run_local
from cr_knee_fit.model import Model

if __name__ == "__main__":
    opts = LocalRunOptions.parse()
    analysis_name = guess_analysis_name(__file__)

    fit_data_config = DataConfig(
        experiments_elements=[experiments.ams02],
        experiments_all_particle=[],
        experiments_lnA=[],
        default_elements=Element.regular(),
        elements_R_bounds=(5, np.inf),
    )

    validation_data_config = DataConfig(
        experiments_elements=[experiments.dampe, experiments.calet],
        experiments_all_particle=[experiments.hawc],
        experiments_lnA=[],
        aux_data=[
            (experiments.ams02, FluxRatio(Element.H, Element.He)),
            (experiments.calet, FluxRatio(Element.H, Element.He)),
            (experiments.ams02, FluxRatio(Element.Li, Element.B)),
            (experiments.ams02, FluxRatio(Element.B, Element.C)),
            (experiments.ams02, FluxRatio(Element.B, Element.O)),
            (experiments.ams02, FluxRatio(Element.Be, Element.B)),
            (experiments.ams02, FluxRatio(Element.C, Element.O)),
            (experiments.ams02, FluxRatio(Element.Fe, Element.O)),
        ],
        default_elements=Element.regular(),
        elements_R_bounds=(0, np.inf),
    ).excluding(fit_data_config)

    def generate_guess() -> Model:
        return Model(
            populations=[],
            energy_shifts=initial_guess_energy_shifts(
                fit_data_config.experiments_spectrum,
                fixed=experiments.ams02,
            ),
            crams=CramsModel.default(),
        )

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
