from scipy import stats  # type: ignore

from cr_knee_fit import experiments
from cr_knee_fit.analysis import (
    FitConfig,
    McmcConfig,
    PlotsConfig,
    PosteriorPlotConfig,
)
from cr_knee_fit.cr_model import (
    CosmicRaysModel,
    CosmicRaysModelConfig,
    PopulationMetadata,
    SharedPowerLawSpectrum,
    SpectralBreakConfig,
    SpectralComponentConfig,
)
from cr_knee_fit.elements import Element
from cr_knee_fit.fit_data import DataConfig
from cr_knee_fit.guesses import (
    initial_guess_break,
    initial_guess_main_population,
)
from cr_knee_fit.local import LocalRunOptions, guess_analysis_name, run_local
from cr_knee_fit.model import Model
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts

if __name__ == "__main__":
    opts = LocalRunOptions.parse()
    analysis_name = guess_analysis_name(__file__)

    fit_data_config = DataConfig(
        experiments_elements=list(
            experiments.DIRECT
            + [
                # experiments.grapes,
                experiments.lhaaso_qgsjet,
            ]
        ),
        experiments_all_particle=[
            # experiments.lhaaso_qgsjet,
            # experiments.hawc,
            # experiments.kascade_re_qgsjet,
        ],
        experiments_lnA=[
            # experiments.lhaaso_qgsjet,
        ],
        # elements=Element.regular(),
        default_elements=[Element.H, Element.He],
    )

    validation_data_config = DataConfig(
        experiments_elements=[
            experiments.kascade_re_qgsjet,
        ],
        experiments_all_particle=[
            # experiments.tale,
        ],
        experiments_lnA=[
            # experiments.lhaaso_qgsjet,
            # experiments.kascade_re_qgsjet,
        ],
        # elements=Element.regular(),
        default_elements=[Element.H, Element.He],
    ).excluding(fit_data_config)

    def generate_guess() -> Model:
        pop1_model = initial_guess_main_population(
            pop_config=CosmicRaysModelConfig(
                components=[
                    SpectralComponentConfig([Element.H, Element.He]),
                    # SpectralComponentConfig([Element.He]),
                    # SpectralComponentConfig(Element.nuclei()),
                    # SpectralComponentConfig(Element.regular()),
                ],
                breaks=[
                    SpectralBreakConfig(
                        fixed_lg_sharpness=None,
                        lg_break_prior_limits=(5.5, 7.0),
                        is_softening=True,
                        lg_break_hint=6.3,
                    ),
                ],
                rescale_all_particle=False,
                population_meta=PopulationMetadata(name="Base", linestyle="--"),
            )
        )

        pop2_model = CosmicRaysModel(
            base_spectra=[
                SharedPowerLawSpectrum(
                    lgI_per_element={
                        Element.H: stats.norm.rvs(loc=-4, scale=0.05),
                        Element.He: stats.norm.rvs(loc=-6, scale=0.05),
                        # Element.Fe: stats.norm.rvs(loc=-7, scale=0.05),
                    },
                    alpha=stats.norm.rvs(loc=1.8, scale=0.05),
                ),
            ],
            breaks=[
                initial_guess_break(
                    SpectralBreakConfig(
                        fixed_lg_sharpness=None,
                        lg_break_prior_limits=(3, 5),
                        is_softening=True,
                        lg_break_hint=4.5,
                    ),
                )
            ],
            # cutoff=initial_guess_cutoff(
            #     SpectralCutoffConfig(
            #         fixed_lg_sharpness=None,
            #         lg_cut_prior_limits=(3, 5),
            #     )
            # ),
            all_particle_lg_shift=None,
            free_Z=None,
            unresolved_elements_spectrum=None,
            population_meta=PopulationMetadata(name="Bump", linestyle=":"),
        )

        return Model(
            populations=[
                pop1_model,
                pop2_model,
            ],
            energy_shifts=ExperimentEnergyScaleShifts(
                lg_shifts={
                    exp: stats.norm.rvs(loc=0, scale=0.01)
                    for exp in fit_data_config.experiments_spectrum
                    if exp != experiments.dampe
                }
            ),
        )

    config = FitConfig.from_guessing_func(
        name=analysis_name,
        fit_data=fit_data_config,
        mcmc=(
            McmcConfig(
                n_steps=500_000,
                n_walkers=64,
                processes=12,
                reuse_saved=True,
            )
        ),
        generate_guess=generate_guess,
        plots=PlotsConfig(
            validation_data_config=validation_data_config,
            elements=PosteriorPlotConfig(
                max_margin_around_data=0.2,
                population_contribs_best_fit=True,
            ),
            all_particle=PosteriorPlotConfig(
                max_margin_around_data=1.0,
                population_contribs_best_fit=True,
            ),
        ),
    )

    run_local(config, opts)
