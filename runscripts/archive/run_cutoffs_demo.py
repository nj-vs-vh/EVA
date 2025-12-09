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
    PopulationMetadata,
    SpectralComponentConfig,
    SpectralCutoffConfig,
)
from cr_knee_fit.elements import Element
from cr_knee_fit.fit_data import Data, DataConfig
from cr_knee_fit.guesses import (
    initial_guess_main_population,
)
from cr_knee_fit.local import LocalRunOptions, run_local
from cr_knee_fit.model import Model
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts

if __name__ == "__main__":
    opts = LocalRunOptions.parse()

    analysis_name = "cutoffs-demo"

    print(f"Running pre-configured analysis: {analysis_name}")

    fit_data_config = DataConfig(
        experiments_elements=list(
            experiments.DIRECT + [experiments.grapes, experiments.lhaaso_qgsjet]
        ),
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
        return Model(
            populations=[
                initial_guess_main_population(
                    CosmicRaysModelConfig(
                        components=[
                            SpectralComponentConfig([Element.H]),
                        ],
                        breaks=[],
                        rescale_all_particle=False,
                        cutoff=SpectralCutoffConfig(
                            lg_cut_prior_limits=(3, 5),
                            lg_cut_hint=4.0,
                        ),
                        population_meta=PopulationMetadata(name="Pop. 1", linestyle="--"),
                    )
                ),
                initial_guess_main_population(
                    CosmicRaysModelConfig(
                        components=[
                            SpectralComponentConfig([Element.H]),
                        ],
                        breaks=[],
                        rescale_all_particle=False,
                        cutoff=SpectralCutoffConfig(
                            lg_cut_prior_limits=(5, 7),
                            lg_cut_hint=6.0,
                        ),
                        cutoff_lower=SpectralCutoffConfig(
                            lg_cut_prior_limits=(3, 6),
                            lg_cut_hint=4.0,
                        ),
                        population_meta=PopulationMetadata(name="Pop. 2", linestyle="-."),
                    ),
                    initial_guess_lgI_override={Element.H: -4.5},
                ),
            ],
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

    m.plot_spectra(Data.load(fit_data_config), scale=2.6).savefig("test.png")

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
