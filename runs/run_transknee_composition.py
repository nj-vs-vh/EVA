import itertools

from scipy import stats  # type: ignore

from bayesian_analysis import FitConfig, McmcConfig, PlotsConfig
from cr_knee_fit import experiments
from cr_knee_fit.cr_model import (
    CosmicRaysModel,
    CosmicRaysModelConfig,
    PopulationMetadata,
    SharedPowerLawSpectrum,
    SpectralBreakConfig,
    SpectralComponentConfig,
    SpectralCutoffConfig,
)
from cr_knee_fit.elements import Element
from cr_knee_fit.fit_data import DataConfig
from cr_knee_fit.guesses import (
    initial_guess_break,
    initial_guess_cutoff,
    initial_guess_main_population,
    initial_guess_pl_index,
)
from cr_knee_fit.model_ import Model
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts
from local import LocalRunOptions, run_local

if __name__ == "__main__":
    opts = LocalRunOptions.parse()
    analysis_name = "trans-knee-composition-2pop"

    print(f"Running pre-configured analysis: {analysis_name}")

    components: list[tuple[Element, ...]] = [
        (Element.H,),
        (Element.He,),
        (Element.C, Element.O),
        (Element.Si,),
        (Element.Fe,),
    ]

    fit_data_config = DataConfig(
        experiments_elements=list(
            experiments.DIRECT
            + [
                experiments.grapes,
                # experiments.lhaaso_epos,
                experiments.kascade_re_qgsjet,
            ]
        ),
        experiments_all_particle=[
            # experiments.lhaaso_epos,
            # experiments.hawc,
            # experiments.tale,
            experiments.kascade_re_qgsjet,
        ],
        experiments_lnA=[],
        elements=list(itertools.chain.from_iterable(components)),
    )

    validation_data_config = DataConfig(
        experiments_elements=experiments.ALL,  # type: ignore
        experiments_all_particle=[
            experiments.hawc,
            experiments.lhaaso_epos,
            experiments.tale,
            experiments.kascade_re_qgsjet,
        ],
        experiments_lnA=[
            experiments.lhaaso_epos,
            experiments.kascade_re_qgsjet,
        ],
        elements=Element.regular(),
    ).excluding(fit_data_config)

    def generate_guess() -> Model:
        shifted_experiments = (
            fit_data_config.experiments_spectrum + fit_data_config.experiments_spectrum
        )
        pop1_model = initial_guess_main_population(
            pop_config=CosmicRaysModelConfig(
                # components=[SpectralComponentConfig([el]) for el in Element.regular()],
                components=[SpectralComponentConfig(list(component)) for component in components],
                breaks=[
                    # SpectralBreakConfig(
                    #     fixed_lg_sharpness=0.7,
                    #     quantity="R",
                    #     lg_break_prior_limits=(3.8, 4.3),
                    #     is_softening=True,
                    #     lg_break_hint=4.0,
                    # ),
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
                cutoff=SpectralCutoffConfig(
                    fixed_lg_sharpness=None,
                    lg_cut_prior_limits=(3, 5),
                ),
                rescale_all_particle=False,
                population_meta=PopulationMetadata(name="LE", linestyle="--"),
                # population_meta=None,
            )
        )

        pop2_norm_guess = {
            Element.H: stats.norm.rvs(loc=-5, scale=0.05),
            Element.He: stats.norm.rvs(loc=-6, scale=0.05),
            Element.C: stats.norm.rvs(loc=-7, scale=0.05),
            Element.Si: stats.norm.rvs(loc=-8, scale=0.05),
            Element.Fe: stats.norm.rvs(loc=-8, scale=0.05),
        }
        pop2_model = CosmicRaysModel(
            base_spectra=[
                SharedPowerLawSpectrum(
                    lgI_per_element={el: pop2_norm_guess[component[0]] for el in component},
                    alpha=initial_guess_pl_index(2.5),
                )
                for component in components
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
            all_particle_lg_shift=None,
            free_Z=None,
            unresolved_elements_spectrum=None,
            population_meta=PopulationMetadata(
                name="Knee",
                linestyle=":",
            ),
        )

        hole_filling_model = CosmicRaysModel(  # noqa: F841
            base_spectra=[
                SharedPowerLawSpectrum(
                    lgI_per_element={Element.He: -8},
                    alpha=initial_guess_pl_index(2.1),
                )
            ],
            breaks=[],
            cutoff=initial_guess_cutoff(
                SpectralCutoffConfig(
                    fixed_lg_sharpness=None,
                    lg_cut_prior_limits=(5.0, 7.0),
                    lg_cut_hint=7.0,
                )
            ),
            all_particle_lg_shift=None,
            free_Z=None,
            unresolved_elements_spectrum=None,
            population_meta=PopulationMetadata(
                name="Ad hoc",
                linestyle="-.",
            ),
        )

        return Model(
            populations=[
                pop1_model,
                pop2_model,
                # hole_filling_model,
            ],
            energy_shifts=ExperimentEnergyScaleShifts(
                lg_shifts={
                    exp: float(stats.norm.rvs(loc=0, scale=0.01)) for exp in shifted_experiments
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
                n_steps=30_000,
                n_walkers=64,
                processes=8,
                reuse_saved=True,
            )
        ),
        generate_guess=generate_guess,
        plots=PlotsConfig(validation_data_config=validation_data_config),
    )

    run_local(config, opts=opts)
