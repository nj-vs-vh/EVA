import argparse

from crams import InjectionLognormalDist, InjectionParams, PropagationParams

from cr_knee_fit.cr_models.crams import (
    SHARED_NUCLEI_INJECTION,
    SHARED_PROPAGATION,
    CramsModel,
    init_injection,
)
from cr_knee_fit.cr_models.pheno import (
    PopulationMetadata,
)
from cr_knee_fit.cr_models.utils import Param
from cr_knee_fit.elements import Element
from cr_knee_fit.local import LocalRunOptions, get_analysis_name
from cr_knee_fit.model import CRModel
from runscripts.minimal import generate_minimal_populations, run


def generate_2LE_pop() -> list[CRModel]:

    _, crams_HE = generate_minimal_populations()

    element_init_abundances = {
        Element.C: Param.fitted_relative(0.004, 0.01),
        Element.N: Param.fitted_relative(0.00053, 0.01),
        Element.O: Param.fitted_relative(0.008, 0.01),
        Element.Ne: Param.fitted_relative(0.0017, 0.01),
        Element.Mg: Param.fitted_relative(0.00312, 0.01),
        Element.Si: Param.fitted_relative(0.0037, 0.01),
        Element.S: Param.fitted_relative(0.00069, 0.01),
        Element.Fe: Param.fitted_relative(0.00914, 0.01),
    }

    crams_LE1 = CramsModel.make(
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
                Element.He: Param.fitted_relative(0.002, 0.01),
                **element_init_abundances,
            },
            slopes=[Param.fitted(4.45, 0.01)],
            features=[
                InjectionLognormalDist(
                    R_mean_GV=Param.fitted(10**4.48, 100),
                    sigma=Param.fitted(0.514, 0.01),
                    beta=Param.frozen(1.0),
                )
            ],
        ),
        population_meta=PopulationMetadata(name="LE1", linestyle="--"),
    )

    crams_LE2 = CramsModel.make(
        to_high_E=True,
        randomize_init=True,
        init_propagation=SHARED_PROPAGATION,
        init_injection=InjectionParams(
            abundances=[
                Param.fitted_relative(0.0042, 0.01),
                Param.fitted_relative(0.02, 0.01),
                *SHARED_NUCLEI_INJECTION,
            ],
            slopes=[Param.fitted(4.3, 0.01)],
            features=[
                InjectionLognormalDist(
                    R_mean_GV=Param.fitted(10**4.58, 100),
                    sigma=Param.fitted(0.5, 0.01),
                    beta=Param.frozen(1.0),
                )
            ],
        ),
        population_meta=PopulationMetadata(name="LE2", linestyle="-."),
    )
    crams_LE2.config.scale_nuclei_abundances = True
    crams_LE2.nuclei_abundance_scale = Param.fitted_relative(0.05, 0.01).sample()

    # pessimistic case: 100 sources with exp cutoff -> prior constraints on the plausible cosmic variance of the observed cutoff shape
    crams_LE1.config.sigma_lgR_max_gumbel_prior_params = (0.51, 0.057)
    crams_LE2.config.sigma_lgR_max_gumbel_prior_params = (0.51, 0.057)

    return [crams_LE1, crams_LE2, crams_HE]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-fast-plots", action="store_true")
    opts = LocalRunOptions.parse(parser)
    analysis_name = get_analysis_name(__file__, opts)

    run(
        analysis_name=analysis_name,
        opts=opts,
        omit_detailed_plots=opts.args_raw.only_fast_plots,
        generate_populations=generate_2LE_pop,
    )
