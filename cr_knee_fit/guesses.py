import numpy as np
from scipy import stats  # type: ignore

from cr_knee_fit.cr_model import (
    CosmicRaysModel,
    CosmicRaysModelConfig,
    SharedPowerLawSpectrum,
    SpectralBreak,
    SpectralBreakConfig,
    SpectralCutoff,
    SpectralCutoffConfig,
    UnresolvedElementsSpectrum,
)
from cr_knee_fit.elements import Element
from cr_knee_fit.model import Model, ModelConfig
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts

# break_pos_guesses = [4.2, 5.3, 6.5]


def initial_guess_break(bc: SpectralBreakConfig) -> SpectralBreak:
    if bc.fixed_lg_sharpness:
        lg_s = bc.fixed_lg_sharpness
    else:
        lg_s = stats.norm.rvs(loc=np.log10(5), scale=0.01)

    lg_break_min, lg_break_max = bc.lg_break_prior_limits
    lg_break_mean = bc.lg_break_initial_guess()
    lg_break_width = min(0.1, lg_break_max - lg_break_mean, lg_break_mean - lg_break_min)
    if not np.isfinite(lg_break_width):
        lg_break_width = 3.0

    return SpectralBreak(
        lg_break=stats.uniform.rvs(loc=lg_break_mean - lg_break_width / 2, scale=lg_break_width),
        d_alpha=stats.norm.rvs(loc=0.5 * (1 if bc.is_softening else -1), scale=0.1),
        lg_sharpness=lg_s,
        config=bc,
    )


def initial_guess_cutoff(c: SpectralCutoffConfig) -> SpectralCutoff:
    lg_min, lg_max = c.lg_cut_prior_limits
    lg_mean = c.lg_cut_initial_guess()
    lg_width = min(0.1, lg_max - lg_mean, lg_mean - lg_min)
    if not np.isfinite(lg_width):
        lg_width = 3.0
    return SpectralCutoff(
        lg_cut=stats.uniform.rvs(loc=lg_mean - lg_width / 2, scale=lg_width),
        lg_sharpness=c.fixed_lg_sharpness or stats.norm.rvs(loc=np.log10(2), scale=0.05),
        config=c,
    )


def initial_guess_pl_index(center: float = 2.6) -> float:
    return stats.norm.rvs(loc=center, scale=0.05)


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
    }
    if initial_guess_lgI_override:
        initial_guess_lgI.update(initial_guess_lgI_override)
    return CosmicRaysModel(
        base_spectra=[
            (
                SharedPowerLawSpectrum(
                    lgI_per_element={
                        element: stats.norm.rvs(loc=initial_guess_lgI.get(element, -8), scale=0.05)
                        for element in comp_conf.elements
                    },
                    alpha=initial_guess_pl_index(center=2.6),
                    lg_scale_contrib_to_all=(
                        stats.uniform.rvs(loc=0.01, scale=0.3)
                        if comp_conf.scale_contrib_to_allpart
                        else None
                    ),
                )
            )
            for comp_conf in pop_config.component_configs
        ],
        breaks=[initial_guess_break(bc) for bc in pop_config.breaks],
        cutoff=initial_guess_cutoff(pop_config.cutoff) if pop_config.cutoff is not None else None,
        cutoff_lower=initial_guess_cutoff(pop_config.cutoff_lower)
        if pop_config.cutoff_lower is not None
        else None,
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
        population_meta=pop_config.population_meta,
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
