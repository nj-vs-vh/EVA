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
from cr_knee_fit.local import LocalRunOptions, run_local
from cr_knee_fit.model import Model
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts

if __name__ == "__main__":
    opts = LocalRunOptions.parse()

    for n_indices in (1, 2, 3, 4):
        analysis_name = f"le-spectral-indices-{n_indices}"

        print(f"Running pre-configured analysis: {analysis_name}")

        fit_data_config = DataConfig(
            experiments_elements=list(experiments.DIRECT + [experiments.grapes]),
            experiments_all_particle=[],
            experiments_lnA=[],
            elements=Element.regular(),
        )

        validation_data_config = DataConfig(
            experiments_elements=[
                experiments.lhaaso_epos,
            ],
            experiments_all_particle=[
                experiments.hawc,
                experiments.lhaaso_epos,
            ],
            experiments_lnA=[experiments.lhaaso_epos],
            elements=Element.regular(),
        ).excluding(fit_data_config)

        separate_index_elements = [Element.H, Element.He, Element.Fe][: n_indices - 1]
        print(f"Separate indices for: {separate_index_elements}")

        def generate_guess() -> Model:
            model = initial_guess_main_population(
                pop_config=CosmicRaysModelConfig(
                    components=[
                        *[SpectralComponentConfig([el]) for el in separate_index_elements],
                        SpectralComponentConfig(
                            [el for el in Element.regular() if el not in separate_index_elements]
                        ),
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
                            lg_break_prior_limits=(4.0, 8.0),
                            is_softening=False,
                            lg_break_hint=5.0,
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
