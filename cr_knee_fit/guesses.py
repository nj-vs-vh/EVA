import numpy as np
from scipy import stats  # type: ignore

from cr_knee_fit.cr_model import (
    CosmicRaysModel,
    CosmicRaysModelConfig,
    SharedPowerLawSpectrum,
    SpectralBreak,
    SpectralBreakConfig,
    UnresolvedElementsSpectrum,
)
from cr_knee_fit.elements import Element
from cr_knee_fit.model_ import Model, ModelConfig
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts


def initial_guess_break(bc: SpectralBreakConfig, break_idx: int) -> SpectralBreak:
    if bc.fixed_lg_sharpness:
        lg_s = bc.fixed_lg_sharpness
    else:
        lg_s = stats.norm.rvs(loc=np.log10(5), scale=0.01)

    break_pos_guesses = [4.2, 5.3, 6.5]
    d_alpha_guesses = [0.3, -0.3, 0.5]

    return SpectralBreak(
        lg_break=stats.norm.rvs(loc=break_pos_guesses[break_idx], scale=0.1),
        d_alpha=stats.norm.rvs(loc=d_alpha_guesses[break_idx], scale=0.05),
        lg_sharpness=lg_s,
        fix_sharpness=bc.fixed_lg_sharpness is not None,
        quantity=bc.quantity,
    )


def initial_guess_main_population(
    pop_config: CosmicRaysModelConfig,
    initial_guess_lgI_override: dict[Element, float] | None = None,
) -> CosmicRaysModel:
    initial_guess_lgI = {
        Element.H: -4,
        Element.He: -4.65,
        Element.C: -6.15,
        Element.O: -6.1,
        Element.Mg: -6.85,
        Element.Si: -6.9,
        Element.Fe: -6.9,
        Element.FreeZ: -8,
    }
    if initial_guess_lgI_override:
        initial_guess_lgI.update(initial_guess_lgI_override)
    return CosmicRaysModel(
        base_spectra=[
            (
                SharedPowerLawSpectrum(
                    lgI_per_element={
                        element: stats.norm.rvs(loc=initial_guess_lgI[element], scale=0.05)
                        for element in comp_conf.elements
                    },
                    alpha=stats.norm.rvs(
                        loc=2.6 if comp_conf.elements == [Element.H] else 2.5,
                        scale=0.05,
                    ),
                    lg_scale_contrib_to_all=(
                        stats.uniform.rvs(loc=0.01, scale=0.3)
                        if comp_conf.scale_contrib_to_allpart
                        else None
                    ),
                )
            )
            for comp_conf in pop_config.component_configs
        ],
        breaks=[initial_guess_break(bc, break_idx=i) for i, bc in enumerate(pop_config.breaks)],
        all_particle_lg_shift=(
            np.log10(stats.uniform.rvs(loc=1.1, scale=0.9))
            if pop_config.rescale_all_particle
            else None
        ),
        free_Z=(
            stats.uniform.rvs(loc=14, scale=26 - 14) if pop_config.has_free_Z_component else None
        ),
        unresolved_elements_spectrum=(
            UnresolvedElementsSpectrum(lgI=stats.norm.rvs(loc=-6.45, scale=0.3))
            if pop_config.add_unresolved_elements
            else None
        ),
    )


def initial_guess_one_population_model(config: ModelConfig) -> Model:
    return Model(
        populations=[
            initial_guess_main_population(config.population_configs[0]),
        ],
        energy_shifts=ExperimentEnergyScaleShifts(
            lg_shifts={exp: stats.norm.rvs(loc=0, scale=0.01) for exp in config.shifted_experiments}
        ),
    )
