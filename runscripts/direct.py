from cr_knee_fit import experiments
from cr_knee_fit.analysis import (
    FitConfig,
    McmcConfig,
    PlotExportOpts,
    PlotsConfig,
    PosteriorPlotConfig,
)
from cr_knee_fit.cr_model import (
    CosmicRaysModelConfig,
    PopulationMetadata,
    SpectralBreakConfig,
    SpectralComponentConfig,
)
from cr_knee_fit.elements import Element
from cr_knee_fit.fit_data import DataConfig
from cr_knee_fit.guesses import (
    initial_guess_energy_shifts,
    initial_guess_main_population,
)
from cr_knee_fit.local import LocalRunOptions, guess_analysis_name, run_local
from cr_knee_fit.model import Model

if __name__ == "__main__":
    opts = LocalRunOptions.parse()
    analysis_name = guess_analysis_name(__file__)

    fit_data_config = DataConfig(
        experiments_elements=list(experiments.DIRECT),
        experiments_all_particle=[],
        experiments_lnA=[],
        default_elements=Element.regular(),
    )

    validation_data_config = DataConfig(
        experiments_elements=[
            # experiments.kascade_re_qgsjet,
            experiments.lhaaso_qgsjet,
        ],
        experiments_all_particle=[
            experiments.hawc,
            experiments.lhaaso_qgsjet,
        ],
        experiments_lnA=[experiments.lhaaso_qgsjet],
        default_elements=Element.regular(),
    ).excluding(fit_data_config)

    def generate_guess() -> Model:
        pop1_model = initial_guess_main_population(
            pop_config=CosmicRaysModelConfig(
                components=[
                    SpectralComponentConfig([Element.H]),
                    SpectralComponentConfig([Element.He]),
                    SpectralComponentConfig(Element.nuclei()),
                    # SpectralComponentConfig([Element.Fe]),
                ],
                breaks=[
                    SpectralBreakConfig(
                        fixed_lg_sharpness=0.7,
                        quantity="R",
                        lg_break_prior_limits=(3.8, 5.0),
                        is_softening=True,
                        lg_break_hint=4.0,
                    ),
                ],
                rescale_all_particle=False,
                population_meta=PopulationMetadata(name="Base", linestyle="--"),
            ),
        )

        return Model(
            populations=[pop1_model],
            energy_shifts=initial_guess_energy_shifts(
                fit_data_config.experiments_spectrum, fixed=experiments.dampe
            ),
        )

    run_local(
        config=FitConfig.from_guessing_func(
            name=analysis_name,
            fit_data=fit_data_config,
            mcmc=(
                McmcConfig(
                    n_steps=500_000,
                    n_walkers=64,
                    processes=8,
                    reuse_saved=True,
                )
            ),
            generate_guess=generate_guess,
            plots=PlotsConfig(
                validation_data_config=validation_data_config,
                export_opts=PlotExportOpts(main="composition-only-model.pdf"),
                elements=PosteriorPlotConfig(max_margin_around_data=0.2),
                all_particle=PosteriorPlotConfig(max_margin_around_data=1.0),
            ),
        ),
        opts=opts,
    )
