from scipy import stats  # type: ignore

from cr_knee_fit import experiments
from cr_knee_fit.analysis import (
    FitConfig,
    McmcConfig,
    PlotsConfig,
    PosteriorPlotConfig,
)
from cr_knee_fit.cr_model import (
    CosmicRaysModelConfig,
    SpectralBreakConfig,
    SpectralComponentConfig,
)
from cr_knee_fit.elements import Element
from cr_knee_fit.fit_data import DataConfig
from cr_knee_fit.guesses import (
    initial_guess_main_population,
)
from cr_knee_fit.local import LocalRunOptions, guess_run_name, run_local
from cr_knee_fit.model import Model
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts

if __name__ == "__main__":
    opts = LocalRunOptions.parse()
    analysis_name = guess_run_name(__file__)

    fit_data_config = DataConfig(
        experiments_elements=list(
            [
                experiments.dampe,
            ]
        ),
        experiments_all_particle=[],
        experiments_lnA=[
            # experiments.lhaaso_qgsjet,
        ],
        elements=[Element.H, Element.C, Element.O, Element.He],
    )

    validation_data_config = DataConfig()

    def generate_guess() -> Model:
        pop1_model = initial_guess_main_population(
            pop_config=CosmicRaysModelConfig(
                components=[
                    SpectralComponentConfig([Element.H]),
                    SpectralComponentConfig([Element.He]),
                    SpectralComponentConfig([Element.C, Element.O]),
                ],
                breaks=[
                    SpectralBreakConfig(
                        fixed_lg_sharpness=0.7,
                        quantity="R",
                        lg_break_prior_limits=(3.0, 5.0),
                        is_softening=True,
                        lg_break_hint=4.3,
                    ),
                ],
                rescale_all_particle=False,
                # population_meta=PopulationMetadata(name="Base", linestyle="--"),
            )
        )

        # pop2_model = CosmicRaysModel(
        #     base_spectra=[
        #         SharedPowerLawSpectrum(
        #             lgI_per_element={
        #                 Element.H: stats.norm.rvs(loc=-5, scale=0.05),
        #                 Element.He: stats.norm.rvs(loc=-6, scale=0.05),
        #                 Element.Fe: stats.norm.rvs(loc=-7, scale=0.05),
        #             },
        #             alpha=stats.norm.rvs(loc=2.4, scale=0.05),
        #         ),
        #     ],
        #     breaks=[
        #         initial_guess_break(
        #             SpectralBreakConfig(
        #                 fixed_lg_sharpness=0.7,
        #                 quantity="R",
        #                 lg_break_prior_limits=(6, 8),
        #                 is_softening=True,
        #                 lg_break_hint=6.5,
        #             ),
        #         )
        #     ],
        #     all_particle_lg_shift=None,
        #     free_Z=None,
        #     unresolved_elements_spectrum=None,
        #     population_meta=PopulationMetadata(name="Knee", linestyle=":"),
        # )

        return Model(
            populations=[
                pop1_model,
                # pop2_model,
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
