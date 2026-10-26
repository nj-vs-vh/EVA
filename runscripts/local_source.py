import argparse

from crams import InjectionLognormalDist

from cr_knee_fit.cr_models.crams import SHARED_PROPAGATION, CramsModel, init_injection
from cr_knee_fit.cr_models.utils import Param, PopulationMetadata
from cr_knee_fit.elements import Element
from cr_knee_fit.local import LocalRunOptions, get_analysis_name
from cr_knee_fit.model import CRModel
from runscripts.minimal import generate_minimal_populations, run


def generate_populations_with_bump() -> list[CRModel]:
    minimal = generate_minimal_populations()

    local_source_injection = init_injection(
        fitted_abundances={
            Element.H: Param.fitted_relative(1e-4, 0.01),
            Element.He: Param.fitted_relative(2e-3, 0.01),
        },
        frozen_abundances={el: 0.0 for el in Element.regular()},
        slopes=[
            Param.fitted(3.5, 0.001),
        ],
        features=[
            InjectionLognormalDist(
                R_mean_GV=Param.fitted(10**5.0, 1000),
                sigma=Param.fitted(0.2, 0.02),
                beta=Param.frozen(1.0),
            )
        ],
    )
    local_source_injection.abundances_are_precomputed_normalizations = True

    local_source = CramsModel.make(
        to_high_E=True,
        randomize_init=True,
        init_propagation=SHARED_PROPAGATION,
        init_injection=local_source_injection,
        population_meta=PopulationMetadata(name="Loc", linestyle="-."),
    )

    return [*minimal, local_source]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-fast-plots", action="store_true")
    opts = LocalRunOptions.parse(parser)
    analysis_name = get_analysis_name(__file__, opts)

    run(
        analysis_name=analysis_name,
        opts=opts,
        omit_detailed_plots=opts.args_raw.only_fast_plots,
        generate_populations=generate_populations_with_bump,
    )
