from scipy import stats  # type: ignore

from bayesian_analysis import (
    FitConfig,
    McmcConfig,
    PlotsConfig,
    PosteriorPlotConfig,
)
from cr_knee_fit import experiments
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
from cr_knee_fit.model_ import Model
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts
from local import LocalRunOptions, run_local

if __name__ == "__main__":
    opts = LocalRunOptions.parse()

    analysis_name = "lhaaso-dampe-energy-scale"

    print(f"Running pre-configured analysis: {analysis_name}")

    fit_data_config = DataConfig(
        experiments_elements=[experiments.dampe, experiments.grapes, experiments.lhaaso_qgsjet],
        experiments_all_particle=[],
        experiments_lnA=[],
        elements=[Element.H],
    )

    validation_data_config = DataConfig(
        experiments_elements=[],
        experiments_all_particle=[],
        experiments_lnA=[],
        elements=Element.regular(),
    ).excluding(fit_data_config)

    def generate_guess() -> Model:
        model = initial_guess_main_population(
            pop_config=CosmicRaysModelConfig(
                components=[
                    SpectralComponentConfig([Element.H]),
                ],
                breaks=[
                    SpectralBreakConfig(
                        fixed_lg_sharpness=0.7,
                        quantity="R",
                        lg_break_prior_limits=(3.0, 5.0),
                        is_softening=True,
                        lg_break_hint=4.3,
                    ),
                    SpectralBreakConfig(
                        fixed_lg_sharpness=0.7,
                        quantity="R",
                        lg_break_prior_limits=(4.0, 7.0),
                        is_softening=False,
                        lg_break_hint=5.0,
                    ),
                    SpectralBreakConfig(
                        fixed_lg_sharpness=0.7,
                        quantity="R",
                        lg_break_prior_limits=(5.0, 8.0),
                        is_softening=True,
                        lg_break_hint=6.2,
                    ),
                ],
                rescale_all_particle=False,
            )
        )

        return Model(
            populations=[model],
            energy_shifts=ExperimentEnergyScaleShifts(
                lg_shifts={
                    exp: stats.norm.rvs(loc=0, scale=0.01)
                    for exp in fit_data_config.experiments_spectrum
                    if exp != experiments.dampe
                }
            ),
            # energy_scale_lg_uncertainty_override={experiments.lhaaso_qgsjet: percent2lg(1.0)}
        )

    m = generate_guess()
    m.validate_packing()

    config = FitConfig.from_guessing_func(
        name=analysis_name,
        fit_data=fit_data_config,
        mcmc=(
            McmcConfig(
                n_steps=10_000,
                n_walkers=64,
                processes=12,
                reuse_saved=True,
            )
        ),
        generate_guess=generate_guess,
        plots=PlotsConfig(
            validation_data_config=validation_data_config,
            elements=PosteriorPlotConfig(
                max_margin_around_data=0.2, population_contribs_best_fit=True
            ),
            all_particle=PosteriorPlotConfig(
                max_margin_around_data=1.0, population_contribs_best_fit=True
            ),
        ),
    )

    run_local(config, opts)
