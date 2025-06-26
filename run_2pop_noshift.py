from scipy import stats  # type: ignore

from bayesian_analysis import (
    FitConfig,
    McmcConfig,
    PlotExportOpts,
    PlotsConfig,
    PosteriorPlotConfig,
)
from cr_knee_fit import experiments
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
from cr_knee_fit.model_ import Model
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts
from run_local import LocalRunOptions, run_local

if __name__ == "__main__":
    opts = LocalRunOptions.parse()

    analysis_name = "2pop-noshift"

    print(f"Running pre-configured analysis: {analysis_name}")

    fit_data_config = DataConfig(
        experiments_elements=list(
            experiments.DIRECT
            + [
                experiments.grapes,
                experiments.lhaaso_epos,
                # (experiments.kascade_re_qgsjet, [Element.H, Element.He]),
                # (experiments.kascade_re_qgsjet, Element.regular()),
            ]
        ),
        experiments_all_particle=[
            experiments.lhaaso_epos,
            experiments.hawc,
            experiments.kascade_re_qgsjet,
            # experiments.tale,
        ],
        experiments_lnA=[],
        elements=Element.regular(),
    )

    validation_data_config = DataConfig(
        experiments_elements=[
            experiments.kascade_re_qgsjet,
        ],
        experiments_all_particle=[
            # experiments.hawc,
            # experiments.lhaaso_epos,
        ],
        experiments_lnA=[experiments.lhaaso_epos],
        elements=Element.regular(),
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
                        lg_break_prior_limits=(3.0, 5.0),
                        is_softening=True,
                        lg_break_hint=4.3,
                    ),
                    # SpectralBreakConfig(
                    #     fixed_lg_sharpness=0.7,
                    #     quantity="R",
                    #     lg_break_prior_limits=(4.3, 8.8),
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
                # cutoff=SpectralCutoffConfig(
                #     fixed_lg_sharpness=None,
                #     lg_cut_prior_limits=(3.5, 4.5),
                #     lg_cut_hint=4.0,
                # ),
                rescale_all_particle=False,
                population_meta=PopulationMetadata(name="LE", linestyle="--"),
            )
        )

        pop2_model = CosmicRaysModel(
            base_spectra=[
                SharedPowerLawSpectrum(
                    lgI_per_element={
                        Element.H: stats.norm.rvs(loc=-5, scale=0.05),
                        Element.He: stats.norm.rvs(loc=-6, scale=0.05),
                        # Element.C: stats.norm.rvs(loc=-7, scale=0.05),
                        # Element.Si: stats.norm.rvs(loc=-8, scale=0.05),
                        # Element.Fe: stats.norm.rvs(loc=-8, scale=0.05),
                    },
                    alpha=stats.norm.rvs(loc=2.4, scale=0.05),
                ),
                # SharedPowerLawSpectrum(
                #     lgI_per_element={
                #         Element.He: stats.norm.rvs(loc=-6, scale=0.05),
                #     },
                #     alpha=stats.norm.rvs(loc=2.4, scale=0.05),
                # ),
                # SharedPowerLawSpectrum(
                #     lgI_per_element={
                #         Element.C: stats.norm.rvs(loc=-7, scale=0.05),
                #         Element.Si: stats.norm.rvs(loc=-8, scale=0.05),
                #         Element.Fe: stats.norm.rvs(loc=-8, scale=0.05),
                #     },
                #     alpha=stats.norm.rvs(loc=2.4, scale=0.05),
                # ),
            ],
            breaks=[
                initial_guess_break(
                    SpectralBreakConfig(
                        fixed_lg_sharpness=0.7,
                        quantity="R",
                        lg_break_prior_limits=(6, 7),
                        is_softening=True,
                        lg_break_hint=6.5,
                    ),
                )
            ],
            # cutoff=initial_guess_cutoff(
            #     SpectralCutoffConfig(
            #         fixed_lg_sharpness=None,
            #         lg_cut_prior_limits=(6.0, 7.0),
            #         lg_cut_hint=6.5,
            #     )
            # ),
            all_particle_lg_shift=None,
            free_Z=None,
            unresolved_elements_spectrum=None,
            population_meta=PopulationMetadata(
                name="Knee",
                linestyle=":",
            ),
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
            energy_scale_lg_uncertainty_override={
                experiments.hawc: 3.0,
                experiments.lhaaso_epos: 3.0,
                experiments.lhaaso_qgsjet: 3.0,
                experiments.lhaaso_sibyll: 3.0,
            },
        )

    m = generate_guess()
    m.validate_packing()

    config = FitConfig.from_guessing_func(
        name=analysis_name,
        fit_data=fit_data_config,
        mcmc=(
            McmcConfig(
                n_steps=200_000,
                n_walkers=64,
                processes=12,
                reuse_saved=True,
            )
        ),
        generate_guess=generate_guess,
        plots=PlotsConfig(
            validation_data_config=validation_data_config,
            export_opts=PlotExportOpts(main="2pop-model.pdf"),
            elements=PosteriorPlotConfig(
                max_margin_around_data=0.2, population_contribs_best_fit=True
            ),
            all_particle=PosteriorPlotConfig(
                max_margin_around_data=1.0, population_contribs_best_fit=True
            ),
        ),
    )

    run_local(config, opts)
