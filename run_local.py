import contextlib
from pathlib import Path

import numpy as np

from bayesian_analysis import FitConfig, McmcConfig, run_bayesian_analysis
from cr_knee_fit import experiments
from cr_knee_fit.cr_model import CosmicRaysModelConfig, SpectralBreakConfig
from cr_knee_fit.model import ModelConfig
from cr_knee_fit.types_ import Primary

if __name__ == "__main__":
    analysis_name = "test-knee-in-e-n"

    experiments_detailed = experiments.direct_experiments + [experiments.grapes]
    lhaaso = experiments.lhaaso_epos
    experiments_all_particle = [lhaaso]
    experiments_lnA = [lhaaso]
    # experiments_all_particle = []
    # experiments_lnA = []

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
                    [
                        Primary.C,
                        Primary.O,
                        Primary.Mg,
                        Primary.Si,
                        Primary.Fe,
                    ],
                ],
                breaks=[
                    SpectralBreakConfig(fixed_lg_sharpness=np.log10(5), quantity="R"),
                    SpectralBreakConfig(fixed_lg_sharpness=np.log10(10), quantity="R"),
                    SpectralBreakConfig(fixed_lg_sharpness=np.log10(5), quantity="E_n"),
                ],
                rescale_all_particle=True,
            ),
            shifted_experiments=[
                # e for e in experiments_detailed + experiments_all_particle if e != experiments.ams02
            ],
        ),
        mcmc=None,
        # mcmc=McmcConfig(
        #     n_steps=200_000,
        #     n_walkers=64,
        #     processes=8,
        #     reuse_saved=True,
        # ),
    )

    outdir = Path(__file__).parent / "out" / config.name
    outdir.mkdir(exist_ok=True, parents=True)

    logfile = outdir / "log.txt"
    with logfile.open("w") as log, contextlib.redirect_stdout(log):
        run_bayesian_analysis(config, outdir)
