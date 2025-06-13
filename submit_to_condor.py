import os
import sys
from pathlib import Path

import htcondor  # type: ignore

from bayesian_analysis import FitConfig, McmcConfig, PlotsConfig
from cr_knee_fit import experiments
from cr_knee_fit.cr_model import (
    CosmicRaysModelConfig,
    SpectralBreakConfig,
    SpectralComponentConfig,
)
from cr_knee_fit.elements import Element
from cr_knee_fit.fit_data import DataConfig
from cr_knee_fit.guesses import initial_guess_one_population_model
from cr_knee_fit.model_ import ModelConfig


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
    analysis_name = "vanilla+icetop"

    fit_data_config = DataConfig(
        experiments_elements=experiments.DIRECT + [experiments.grapes],
        experiments_all_particle=[experiments.ice_top_sibyll],
        experiments_lnA=[],
        elements=Element.regular(),
    )

    validation_data_config = DataConfig(
        experiments_elements=[],
        experiments_all_particle=[
            experiments.lhaaso_sibyll,
            experiments.kascade_sibyll,
            experiments.kascade_grande_sibyll,
            experiments.gamma,
        ],
        experiments_lnA=[experiments.lhaaso_sibyll],
        elements=[],
    )

    model_config = ModelConfig(
        cr_model_config=CosmicRaysModelConfig(
            components=[
                [Element.H],
                [Element.He],
                SpectralComponentConfig(
                    elements=Element.nuclei(),
                    scale_contrib_to_allpart=True,
                ),
            ],
            breaks=[
                SpectralBreakConfig(fixed_lg_sharpness=0.7),
                SpectralBreakConfig(fixed_lg_sharpness=0.7),
                SpectralBreakConfig(fixed_lg_sharpness=0.7),
            ],
        ),
        shifted_experiments=fit_data_config.experiments_spectrum,
    )

    config = FitConfig.from_guessing_func(
        name=analysis_name,
        fit_data=fit_data_config,
        mcmc=McmcConfig(
            n_steps=500_000,
            n_walkers=256,
            processes=8,
            reuse_saved=True,
        ),
        plots=PlotsConfig(
            validation_data_config=validation_data_config,
        ),
        generate_guess=lambda: initial_guess_one_population_model(model_config),
        n_guesses=50,
    )
    submit_job(config)
