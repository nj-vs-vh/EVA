import time

import numpy as np

from cr_knee_fit import experiments
from cr_knee_fit.crams_model import CramsModel
from cr_knee_fit.elements import Element
from cr_knee_fit.fit_data import (
    Data,
    DataConfig,
    FluxRatio,
    FluxRatioDataConfig,
    SpectrumDataConfig,
)
from cr_knee_fit.inference import loglikelihood
from cr_knee_fit.model import Model
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts

m = Model(
    populations=[],
    energy_shifts=ExperimentEnergyScaleShifts({}),
    crams=CramsModel.default(),
)

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
            experiments.ams02, FluxRatio(Element.B, Element.C), Q_bounds=(5.0, np.inf)
        ),
        FluxRatioDataConfig(
            experiments.ams02, FluxRatio(Element.B, Element.O), Q_bounds=(5.0, np.inf)
        ),
        FluxRatioDataConfig(
            experiments.ams02, FluxRatio(Element.C, Element.O), Q_bounds=(5.0, np.inf)
        ),
        FluxRatioDataConfig(
            experiments.ams02, FluxRatio(Element.H, Element.He), Q_bounds=(5.0, np.inf)
        ),
        FluxRatioDataConfig(
            experiments.ams02, FluxRatio(Element.He, Element.O), Q_bounds=(5.0, np.inf)
        ),
        FluxRatioDataConfig(
            experiments.ams02, FluxRatio(Element.He, Element.O), Q_bounds=(5.0, np.inf)
        ),
    ],
)

d = Data.load(fit_data_config)

for up_to_stage in (0, 1, 2):
    start = time.time()
    n_evals = 100
    for _ in range(n_evals):
        m = Model.unpack(m.pack(), layout_info=m.layout_info())
        if up_to_stage == 1:
            assert m.crams is not None
            crams_data = m.crams._crams_lg_output
        elif up_to_stage == 2:
            ll = loglikelihood(m, fit_data=d, config=m.layout_info())
    total = time.time() - start
    print(["Parsing", "CRAMS only", "Full loglike"][up_to_stage])
    print(f"Total: {total:.2f} sec, per eval: {1000 * total / n_evals:.2f} msec")
