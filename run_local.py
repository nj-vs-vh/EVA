import contextlib
import itertools
import sys
from pathlib import Path

from scipy import stats  # type: ignore

from bayesian_analysis import FitConfig, McmcConfig, run_bayesian_analysis
from cr_knee_fit import experiments
from cr_knee_fit.cr_model import (
    CosmicRaysModel,
    CosmicRaysModelConfig,
    PopulationMetadata,
    SharedPowerLawSpectrum,
    SpectralBreakConfig,
    SpectralComponentConfig,
)
from cr_knee_fit.guesses import initial_guess_break, initial_guess_one_population_model
from cr_knee_fit.model_ import Model, ModelConfig
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts
from cr_knee_fit.types_ import Element


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
    analysis_name = "scale-only-nuclei"

    experiments_detailed = experiments.direct_experiments + [experiments.grapes]
    experiments_all_particle = [experiments.hawc, experiments.lhaaso_epos]
    experiments_lnA = []

    def generate_guess() -> Model:
        return initial_guess_one_population_model(
            config=ModelConfig(
                population_configs=[
                    CosmicRaysModelConfig(
                        components=[
                            [Element.H],
                            [Element.He],
                            SpectralComponentConfig(
                                Element.nuclei(),
                                scale_contrib_to_allpart=True,
                            ),
                        ],
                        breaks=[
                            SpectralBreakConfig(fixed_lg_sharpness=None, quantity="R"),
                            SpectralBreakConfig(fixed_lg_sharpness=None, quantity="R"),
                            SpectralBreakConfig(fixed_lg_sharpness=None, quantity="R"),
                        ],
                        rescale_all_particle=False,
                        add_unresolved_elements=False,
                    )
                ],
                shifted_experiments=[
                    exp
                    for exp in set(experiments_detailed + experiments_all_particle)
                    if exp != experiments.ams02
                ],
            )
        )

    generate_guess().validate_packing()

    config = FitConfig.from_guessing_func(
        name=analysis_name,
        experiments_detailed=experiments_detailed,
        experiments_all_particle=experiments_all_particle,
        experiments_lnA=experiments_lnA,
        mcmc=McmcConfig(
            n_steps=100_000,
            n_walkers=96,
            processes=8,
            reuse_saved=True,
        ),
        generate_guess=generate_guess,
    )
    run_local(config)
