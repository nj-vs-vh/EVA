import argparse

from crams import InjectionBreak, InjectionLognormalDist, PropagationParams

from cr_knee_fit.cr_models.crams import CramsModel, init_injection
from cr_knee_fit.cr_models.pheno import (
    PopulationMetadata,
)
from cr_knee_fit.cr_models.utils import Param
from cr_knee_fit.elements import Element
from cr_knee_fit.local import LocalRunOptions, get_analysis_name
from cr_knee_fit.model import CRModel
from runscripts.minimal import run


def generate_single_population() -> list[CRModel]:
    crams_LE = CramsModel.make(
        to_high_E=True,
        randomize_init=True,
        init_propagation=PropagationParams(
            H_kpc=Param.frozen(7.0),
            D_0_cm2_sec=Param.fitted(3.21e28, 1e26),
            v_A_km_sec=Param.fitted(1.67, 0.1),
            R_b_GV=Param.fitted(10**2.61, 1),
            delta=Param.fitted(0.47, 0.01),
            ddelta=Param.fitted(0.332, 0.01),
            smoothness=Param.frozen(0.1),
            phi=Param.fitted(0.51, 0.05),
            X_src=Param.frozen(-1.0),
        ),
        init_injection=init_injection(
            fitted_abundances={
                Element.H: Param.fitted_relative(0.042, 0.01),
                Element.He: Param.fitted_relative(0.02, 0.01),
                Element.C: Param.fitted_relative(0.004, 0.01),
                Element.N: Param.fitted_relative(0.00053, 0.01),
                Element.O: Param.fitted_relative(0.008, 0.01),
                Element.Ne: Param.fitted_relative(0.0017, 0.01),
                Element.Mg: Param.fitted_relative(0.00312, 0.01),
                Element.Si: Param.fitted_relative(0.0037, 0.01),
                Element.S: Param.fitted_relative(0.00069, 0.01),
                Element.Fe: Param.fitted_relative(0.00914, 0.01),
            },
            slopes=[
                Param.fitted(4.45, 0.01),
                Param.fitted(4.38, 0.01),
                Param.fitted(4.41, 0.01),
            ],
            features=[
                InjectionBreak(
                    R_GV=Param.fitted_relative(10**4.11, 0.01),
                    delta_slope=Param.fitted(0.2, 0.01),
                    omega=Param.frozen(0.1),
                ),
                InjectionBreak(
                    R_GV=Param.fitted_relative(10**5.1, 0.01),
                    delta_slope=Param.fitted(-0.3, 0.01),
                    omega=Param.frozen(0.1),
                ),
                InjectionLognormalDist(
                    R_mean_GV=Param.fitted(10**6.4, 1000),
                    sigma=Param.fitted(0.4, 0.02),
                    beta=Param.frozen(1.0),
                ),
            ],
        ),
        population_meta=PopulationMetadata(name="", linestyle="-"),
    )

    return [crams_LE]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-fast-plots", action="store_true")
    opts = LocalRunOptions.parse(parser)
    analysis_name = get_analysis_name(__file__, opts)

    run(
        analysis_name=analysis_name,
        opts=opts,
        omit_detailed_plots=opts.args_raw.only_fast_plots,
        generate_populations=generate_single_population,
    )
