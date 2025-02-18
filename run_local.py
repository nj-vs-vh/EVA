import contextlib
import itertools
import sys
from pathlib import Path

from bayesian_analysis import FitConfig, McmcConfig, run_bayesian_analysis
from cr_knee_fit import experiments
from cr_knee_fit.cr_model import (
    CosmicRaysModelConfig,
    SpectralBreakConfig,
    SpectralComponentConfig,
)
from cr_knee_fit.model_ import ModelConfig
from cr_knee_fit.types_ import Primary


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
    for with_lhaaso in (True, False):
        analysis_name = f"scale-only-nuclei-try-{with_lhaaso=}"

        experiments_detailed = experiments.direct_experiments + [experiments.grapes]
        experiments_all_particle = [experiments.dampe, experiments.hawc]
        experiments_lnA = []
        if with_lhaaso:
            experiments_all_particle.append(experiments.lhaaso_epos)
            experiments_lnA.append(experiments.lhaaso_epos)

        config = FitConfig(
            name=analysis_name,
            experiments_detailed=experiments_detailed,
            experiments_all_particle=experiments_all_particle,
            experiments_lnA=experiments_lnA,
            model=ModelConfig(
                cr_model_config=CosmicRaysModelConfig(
                    components=[
                        [Primary.H],
                        [Primary.He],
                        SpectralComponentConfig(
                            primaries=[
                                Primary.C,
                                Primary.O,
                                Primary.Mg,
                                Primary.Si,
                                Primary.Fe,
                            ],
                            scale_contrib_to_allpart=True,
                        ),
                    ],
                    breaks=[
                        SpectralBreakConfig(fixed_lg_sharpness=None, quantity="R"),
                        SpectralBreakConfig(fixed_lg_sharpness=None, quantity="R"),
                    ]
                    + (
                        [
                            SpectralBreakConfig(fixed_lg_sharpness=None, quantity="R"),
                        ]
                        if with_lhaaso
                        else []
                    ),
                    rescale_all_particle=False,
                ),
                shifted_experiments=[
                    e
                    for e in itertools.chain(experiments_detailed, experiments_all_particle)
                    if e != experiments.ams02
                ],
            ),
            mcmc=McmcConfig(
                n_steps=200_000,
                n_walkers=128,
                processes=8,
                reuse_saved=True,
            ),
        )
        run_local(config)
