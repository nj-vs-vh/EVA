import argparse
import contextlib
import sys
from pathlib import Path

from scipy import stats  # type: ignore

from bayesian_analysis import FitConfig, PlotsConfig, run_bayesian_analysis
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
from cr_knee_fit.guesses import initial_guess_break, initial_guess_main_population
from cr_knee_fit.model_ import Model
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts

OUT_DIR = Path(__file__).parent / "out"


def run_local(config: FitConfig) -> None:
    print("Running:")
    print(config)

    outdir = OUT_DIR / config.name
    if outdir.exists():
        print(f"Output directory exists: {outdir}")
        answer = input("Continue? This will overwrite some files! [Yn] ")
        if answer.lower() == "n":
            sys.exit(0)
    outdir.mkdir(exist_ok=True, parents=True)

    logfile = outdir / "log.txt"
    with logfile.open("w") as log, contextlib.redirect_stdout(log):
        run_bayesian_analysis(config, outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    if args.run_dir is not None:
        run_dir = Path(args.run_dir).absolute()
        if not run_dir.exists():
            print(f"Run directory must exist: {run_dir}")
        print(f"Rerunning analysis saved in {run_dir}")
        config_dump_file = run_dir / "config-dump.json"
        config = FitConfig.model_validate_json(config_dump_file.read_text())
        config.name = str(run_dir.relative_to(OUT_DIR))
        print(config)
        input("Press Enter to confirm")

    else:
        analysis_name = "H-He-compat-test"

        print(f"Running pre-configured analysis: {analysis_name}")

        fit_data_config = DataConfig(
            experiments_elements=experiments.DIRECT
            + [
                experiments.grapes,
                experiments.lhaaso_epos,
                experiments.kascade_re_qgsjet,
            ],
            experiments_all_particle=[
                # experiments.lhaaso_epos,
                # experiments.hawc,
                # experiments.tale,
                # experiments.kascade_re_qgsjet,
            ],
            experiments_lnA=[],
            elements=[
                Element.H,
                Element.He,
            ],
        )

        validation_data_config = DataConfig(
            experiments_elements=[],
            # experiments_all_particle=experiments.ALL,
            experiments_all_particle=[
                experiments.hawc,
                experiments.lhaaso_epos,
            ],
            experiments_lnA=[],
            elements=[
                Element.H,
                Element.He,
            ],
        ).excluding(fit_data_config)
        print(validation_data_config)

        def generate_guess() -> Model:
            shifted_experiments = (
                fit_data_config.experiments_elements + fit_data_config.experiments_all_particle
            )
            pop1_model = initial_guess_main_population(
                pop_config=CosmicRaysModelConfig(
                    components=[
                        SpectralComponentConfig([Element.H]),
                        SpectralComponentConfig([Element.He]),
                        # SpectralComponentConfig([Element.Fe]),
                        # SpectralComponentConfig(Element.nuclei()),
                    ],
                    breaks=[
                        SpectralBreakConfig(
                            fixed_lg_sharpness=0.7,
                            quantity="R",
                            lg_break_prior_limits=(3.8, 4.3),
                            is_softening=True,
                            lg_break_hint=4.0,
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
                    rescale_all_particle=False,
                    population_meta=PopulationMetadata(name="Base", linestyle="--"),
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
                    )
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

            return Model(
                populations=[
                    pop1_model,
                    pop2_model,
                ],
                energy_shifts=ExperimentEnergyScaleShifts(
                    lg_shifts={
                        exp: stats.norm.rvs(loc=0, scale=0.01) for exp in shifted_experiments
                    }
                ),
            )

        m = generate_guess()
        m.validate_packing()

        config = FitConfig.from_guessing_func(
            name=analysis_name,
            fit_data=fit_data_config,
            # mcmc=McmcConfig(
            #     n_steps=30_000,
            #     n_walkers=64,
            #     processes=8,
            #     reuse_saved=True,
            # ),
            mcmc=None,
            generate_guess=generate_guess,
            plots=PlotsConfig(validation_data_config=validation_data_config),
        )

    run_local(config)
