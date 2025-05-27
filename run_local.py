import argparse
import contextlib
import sys
from pathlib import Path

from bayesian_analysis import FitConfig, PlotsConfig, run_bayesian_analysis
from cr_knee_fit import experiments
from cr_knee_fit.cr_model import (
    CosmicRaysModelConfig,
    SpectralBreakConfig,
    SpectralComponentConfig,
)
from cr_knee_fit.elements import Element
from cr_knee_fit.fit_data import DataConfig
from cr_knee_fit.guesses import initial_guess_one_population_model
from cr_knee_fit.model_ import Model, ModelConfig

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
        analysis_name = "lhaaso-protons"

        print(f"Running pre-configured analysis: {analysis_name}")

        fit_data_config = DataConfig(
            experiments_elements=experiments.direct_experiments
            + [
                # experiments.grapes,
                experiments.lhaaso_epos,
            ],
            experiments_all_particle=[
                # experiments.kascade_sibyll,
                # experiments.kascade_grande_sibyll,
                # experiments.ice_top_sibyll,
                # experiments.lhaaso_epos,
                # experiments.gamma,
            ],
            experiments_lnA=[],
            elements=Element.regular(),
        )

        validation_data_config = DataConfig(
            experiments_elements=[
                # experiments.lhaaso_qgsjet,
            ],
            experiments_all_particle=[
                experiments.kascade_sibyll,
                experiments.kascade_grande_sibyll,
                experiments.ice_top_sibyll,
                # experiments.lhaaso_sibyll,
                experiments.gamma,
            ],
            experiments_lnA=[experiments.lhaaso_epos],
            elements=[Element.H],
        )

        def generate_guess() -> Model:
            return initial_guess_one_population_model(
                config=ModelConfig(
                    population_configs=[
                        CosmicRaysModelConfig(
                            components=[
                                SpectralComponentConfig([Element.H]),
                                SpectralComponentConfig([Element.He]),
                                SpectralComponentConfig(
                                    Element.nuclei(),
                                    scale_contrib_to_allpart=False,
                                ),
                            ],
                            breaks=[
                                SpectralBreakConfig(fixed_lg_sharpness=0.7, quantity="R"),
                                SpectralBreakConfig(fixed_lg_sharpness=0.7, quantity="R"),
                                SpectralBreakConfig(fixed_lg_sharpness=0.7, quantity="R"),
                            ],
                            rescale_all_particle=False,
                        )
                    ],
                    shifted_experiments=(
                        fit_data_config.experiments_elements
                        + fit_data_config.experiments_all_particle
                    ),
                )
            )

        m = generate_guess()
        m.validate_packing()

        config = FitConfig.from_guessing_func(
            name=analysis_name,
            fit_data=fit_data_config,
            # mcmc=McmcConfig(
            #     n_steps=100_000,
            #     n_walkers=256,
            #     processes=8,
            #     reuse_saved=True,
            # ),
            mcmc=None,
            generate_guess=generate_guess,
            plots=PlotsConfig(validation_data_config=validation_data_config),
        )

    run_local(config)
