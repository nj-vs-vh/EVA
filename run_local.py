import contextlib
import sys
from pathlib import Path

from bayesian_analysis import FitConfig, McmcConfig, run_bayesian_analysis
from cr_knee_fit import experiments
from cr_knee_fit.cr_model import (
    CosmicRaysModelConfig,
    SpectralBreakConfig,
    SpectralComponentConfig,
)
from cr_knee_fit.elements import Element
from cr_knee_fit.guesses import initial_guess_one_population_model
from cr_knee_fit.model_ import Model, ModelConfig


def run_local(config: FitConfig) -> None:
    print("Running:")
    print(config)

    outdir = Path(__file__).parent / "out" / config.name
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
    analysis_name = "vanilla-plus-lhaaso-rescale-all"

    experiments_detailed: list[experiments.Experiment] = experiments.direct_experiments + [
        experiments.grapes
    ]
    experiments_all_particle: list[experiments.Experiment] = [experiments.lhaaso_sibyll]
    experiments_lnA: list[experiments.Experiment] = []

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
                            SpectralBreakConfig(fixed_lg_sharpness=None, quantity="R"),
                        ],
                        rescale_all_particle=True,
                        add_unresolved_elements=False,
                    )
                ],
                shifted_experiments=experiments_detailed + experiments_all_particle,
            )
        )

    generate_guess().validate_packing()

    config = FitConfig.from_guessing_func(
        name=analysis_name,
        experiments_detailed=experiments_detailed,
        experiments_all_particle=experiments_all_particle,
        experiments_lnA=experiments_lnA,
        mcmc=McmcConfig(
            n_steps=200_000,
            n_walkers=128,
            processes=8,
            reuse_saved=True,
        ),
        generate_guess=generate_guess,
    )
    run_local(config)
