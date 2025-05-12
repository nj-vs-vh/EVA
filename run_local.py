import contextlib
import sys
from pathlib import Path

from bayesian_analysis import FitConfig, McmcConfig, PlotsConfig, run_bayesian_analysis
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
    analysis_name = "vanilla+lhaaso"

    fit_data_config = DataConfig(
        experiments_elements=experiments.direct_experiments + [experiments.grapes],
        experiments_all_particle=[experiments.lhaaso_sibyll],
        experiments_lnA=[],
        elements=Element.regular(),
    )

    validation_data_config = DataConfig(
        experiments_elements=[],
        experiments_all_particle=[experiments.hawc],
        experiments_lnA=[experiments.lhaaso_sibyll],
        elements=[],
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
                        rescale_all_particle=True,
                    )
                ],
                shifted_experiments=(
                    fit_data_config.experiments_elements + fit_data_config.experiments_all_particle
                ),
            )
        )

    m = generate_guess()
    m.validate_packing()

    config = FitConfig.from_guessing_func(
        name=analysis_name,
        fit_data=fit_data_config,
        mcmc=McmcConfig(
            n_steps=100_000,
            n_walkers=256,
            processes=8,
            reuse_saved=True,
        ),
        generate_guess=generate_guess,
        plots=PlotsConfig(validation_data_config=validation_data_config),
    )
    run_local(config)
