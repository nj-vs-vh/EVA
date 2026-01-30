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
    PopulationMetadata,
    SharedPowerLawSpectrum,
    SpectralBreakConfig,
    SpectralCutoffConfig,
)
from cr_knee_fit.elements import Element
from cr_knee_fit.fit_data import DataConfig
from cr_knee_fit.guesses import (
    initial_guess_break,
    initial_guess_cutoff,
    initial_guess_energy_shifts,
)
from cr_knee_fit.local import LocalRunOptions, guess_analysis_name, run_local
from cr_knee_fit.model import Model

if __name__ == "__main__":
    opts = LocalRunOptions.parse()
    analysis_name = guess_analysis_name(__file__)

    elements = [Element.H]

    fit_data_config = DataConfig(
        experiments_elements=list(
            experiments.DIRECT
            + [
                experiments.lhaaso_qgsjet,
                experiments.ice_top_sibyll,
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
        default_elements=elements,
    )

    validation_data_config = DataConfig(
        experiments_elements=[],
        experiments_all_particle=[],
        experiments_lnA=[],
        default_elements=elements,
        # aux_data=[
        #     (experiments.dampe, (Element.H, Element.He)),
        #     (experiments.cream, (Element.H, Element.He)),
        #     (experiments.lhaaso_sibyll, (Element.H, Element.He)),
        # ],
    ).excluding(fit_data_config)

    def generate_guess() -> Model:
        le_model = CosmicRaysModel(
            base_spectra=[
                SharedPowerLawSpectrum(
                    lgI_per_element={
                        Element.H: stats.norm.rvs(loc=-4.2, scale=0.05),
                        # Element.He: stats.norm.rvs(loc=-4.2, scale=0.05),
                    },
                    alpha=stats.norm.rvs(loc=2.7, scale=0.05),
                ),
            ],
            breaks=[
                initial_guess_break(
                    SpectralBreakConfig(
                        is_softening=True,
                        fixed_lg_sharpness=0.7,
                        lg_break_hint=4.0,
                        lg_break_prior_limits=(3.0, 5.5),
                    ),
                    abs_d_alpha_guess=1.7,
                )
            ],
            population_meta=PopulationMetadata(name="LE", linestyle=":"),
        )

        he_model = CosmicRaysModel(
            base_spectra=[
                SharedPowerLawSpectrum(
                    lgI_per_element={
                        Element.H: stats.norm.rvs(loc=-4.7, scale=0.05),
                        # Element.He: stats.norm.rvs(loc=-4.7, scale=0.05),
                    },
                    alpha=stats.norm.rvs(loc=2.49, scale=0.05),
                ),
            ],
            breaks=[
                initial_guess_break(
                    SpectralBreakConfig(
                        is_softening=True,
                        fixed_lg_sharpness=0.7,
                        lg_break_hint=6.5,
                        lg_break_prior_limits=(5.5, 7.5),
                    ),
                    abs_d_alpha_guess=1.22,
                )
            ],
            cutoff_lower=initial_guess_cutoff(SpectralCutoffConfig(lg_cut_hint=5.0)),
            population_meta=PopulationMetadata(name="HE", linestyle="--"),
        )

        return Model(
            populations=[
                le_model,
                he_model,
            ],
            energy_shifts=initial_guess_energy_shifts(
                experiments=fit_data_config.experiments_spectrum, fixed=experiments.dampe
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
