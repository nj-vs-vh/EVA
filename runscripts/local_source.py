import argparse

from crams import InjectionLognormalDist

from cr_knee_fit.cr_models.crams import CramsModel, init_injection
from cr_knee_fit.cr_models.utils import Param, PopulationMetadata
from cr_knee_fit.elements import Element
from cr_knee_fit.local import LocalRunOptions, get_analysis_name
from cr_knee_fit.model import CRModel
from runscripts.minimal import generate_minimal_populations, run


def generate_populations() -> list[CRModel]:
    minimal = generate_minimal_populations()

    local_source = CramsModel.make(
        to_high_E=True,
        randomize_init=True,
        # init_propagation=SHARED_PROPAGATION,
        init_injection=init_injection(
            fitted_abundances={
                Element.H: Param.fitted_relative(0.000692, 0.01),
                Element.He: Param.fitted_relative(0.00044, 0.01),
                # Element.C: Param.fitted_relative(0.0000377, 0.01),
                # Element.O: Param.fitted_relative(0.000036, 0.01),
                # Element.Fe: Param.fitted_relative(0.000018, 0.01),
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
        ),
        population_meta=PopulationMetadata(name="Loc", linestyle="-."),
    )

    print(local_source._crams_lg_output)

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
        generate_populations=generate_populations,
    )
