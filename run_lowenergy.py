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
from cr_knee_fit.guesses import initial_guess_main_population
from cr_knee_fit.model_ import Model
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts
from run_local import LocalRunOptions, run_local

if __name__ == "__main__":
    opts = LocalRunOptions.parse()

    analysis_name = "lowenergy"

    print(f"Running pre-configured analysis: {analysis_name}")

    fit_data_config = DataConfig(
        experiments_elements=list(experiments.DIRECT),
        experiments_all_particle=[
            # experiments.hawc,
            # experiments.tale,
            # experiments.kascade_re_qgsjet,
        ],
        experiments_lnA=[],
        elements=Element.regular(),
    )

    validation_data_config = DataConfig(
        experiments_elements=[],
        # experiments_all_particle=experiments.ALL,
        experiments_all_particle=[
            # experiments.hawc,
            # experiments.lhaaso_qgsjet,
        ],
        experiments_lnA=[
            # experiments.lhaaso_qgsjet,
        ],
        elements=[],
    ).excluding(fit_data_config)

    def generate_guess() -> Model:
        pop1_model = initial_guess_main_population(
            pop_config=CosmicRaysModelConfig(
                components=[
                    SpectralComponentConfig([Element.H]),
                    SpectralComponentConfig([Element.He]),
                    SpectralComponentConfig(Element.nuclei()),
                    # SpectralComponentConfig([Element.C, Element.O, Element.Mg, Element.Si]),
                ],
                breaks=[
                    SpectralBreakConfig(
                        fixed_lg_sharpness=0.7,
                        quantity="R",
                        lg_break_prior_limits=(3.8, 5.0),
                        is_softening=True,
                        lg_break_hint=4.0,
                    ),
                    # SpectralBreakConfig(
                    #     fixed_lg_sharpness=0.7,
                    #     quantity="R",
                    #     lg_break_prior_limits=(4.0, 6.8),
                    #     is_softening=False,
                    #     lg_break_hint=5.0,
                    # ),
                    # SpectralBreakConfig(
                    #     fixed_lg_sharpness=0.7,
                    #     quantity="R",
                    #     lg_break_prior_limits=(6.0, 7),
                    #     is_softening=True,
                    #     lg_break_hint=6.2,
                    # ),
                ],
                rescale_all_particle=False,
                # population_meta=PopulationMetadata(name="Base", linestyle="--"),
            )
        )

        return Model(
            populations=[pop1_model],
            energy_shifts=ExperimentEnergyScaleShifts(
                lg_shifts={
                    exp: stats.norm.rvs(loc=0, scale=0.01)
                    for exp in fit_data_config.experiments_spectrum
                    if exp != experiments.dampe
                }
            ),
        )

    m = generate_guess()
    m.validate_packing()

    config = FitConfig.from_guessing_func(
        name=analysis_name,
        fit_data=fit_data_config,
        mcmc=(
            McmcConfig(
                n_steps=50_000,
                n_walkers=64,
                processes=8,
                reuse_saved=True,
            )
        ),
        generate_guess=generate_guess,
        plots=PlotsConfig(
            validation_data_config=validation_data_config,
            elements=PosteriorPlotConfig(max_margin_around_data=0.2),
            all_particle=PosteriorPlotConfig(max_margin_around_data=1.0),
        ),
    )

    run_local(config, opts=opts)
