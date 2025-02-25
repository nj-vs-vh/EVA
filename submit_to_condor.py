import itertools
import os
import sys
from pathlib import Path

import htcondor  # type: ignore

from bayesian_analysis import FitConfig, McmcConfig
from cr_knee_fit import experiments
from cr_knee_fit.cr_model import (
    CosmicRaysModelConfig,
    SpectralBreakConfig,
    SpectralComponentConfig,
)
from cr_knee_fit.guesses import initial_guess_one_population_model
from cr_knee_fit.model_ import ModelConfig
from cr_knee_fit.types_ import Element


def submit_job(config: FitConfig) -> None:
    print("Submitting config:")
    print(config)
    out_dir = Path(__file__).parent / "condor-out" / config.name
    out_dir.mkdir(parents=True)

    (out_dir / "config.json").write_text(config.model_dump_json(indent=2))

    python_path = sys.executable
    print(f"Python path (check if it's the correct env and on a shared disk):\n{python_path}")

    bayesian_analysis_script = Path(__file__).parent.resolve() / "bayesian_analysis.py"

    job = htcondor.Submit(  # type: ignore
        {
            "executable": "/storage/gpfs_data/auger/vaiman/miniconda3/bin/conda",
            "arguments": f"run --name crknees --no-capture-output python {bayesian_analysis_script}",
            "environment": '"CRKNEES_CLUSTER=1 OMP_NUM_THREADS=1"',
            "transfer_input_files": "config.json",
            "batch_name": config.name,
            "initialdir": str(out_dir),
            "output": "condor.out",
            "error": "condor.err",
            "log": "condor.log",
            "request_cpus": str(config.mcmc.processes if config.mcmc else 1),
            "request_memory": "8GB",
            "should_transfer_files": "IF_NEEDED",
            "max_transfer_output_mb": "100",
        }
    )
    print()
    print(job)

    print("Submitting...")
    location = htcondor.Collector().locate(htcondor.DaemonTypes.Schedd, os.environ["SCHEDD_NAME"])  # type: ignore
    schedd = htcondor.Schedd(location)  # type: ignore
    schedd.submit(job)
    print("Done!")


if __name__ == "__main__":
    for with_lhaaso in (True, False):
        analysis_name = f"scale-only-nuclei-try-{with_lhaaso=}"

        experiments_detailed = experiments.direct_experiments + [experiments.grapes]
        experiments_all_particle = [experiments.dampe, experiments.hawc]
        experiments_lnA = []
        if with_lhaaso:
            experiments_all_particle.append(experiments.lhaaso_epos)

            experiments_lnA.append(experiments.lhaaso_epos)

        model_config = ModelConfig(
            cr_model_config=CosmicRaysModelConfig(
                components=[
                    [Element.H],
                    [Element.He],
                    SpectralComponentConfig(
                        elements=[
                            Element.C,
                            Element.O,
                            Element.Mg,
                            Element.Si,
                            Element.Fe,
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
        )

        config = FitConfig.from_guessing_func(
            name=analysis_name,
            experiments_detailed=experiments_detailed,
            experiments_all_particle=experiments_all_particle,
            experiments_lnA=experiments_lnA,
            mcmc=McmcConfig(
                n_steps=50_000,
                n_walkers=64,
                processes=8,
                reuse_saved=True,
            ),
            generate_guess=lambda: initial_guess_one_population_model(model_config),
        )
        submit_job(config)
