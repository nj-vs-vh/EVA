import os
import sys
from pathlib import Path

import htcondor  # type: ignore
import numpy as np

from bayesian_analysis import FitConfig, McmcConfig
from cr_knee_fit import experiments
from cr_knee_fit.cr_model import CosmicRaysModelConfig, SpectralBreakConfig
from cr_knee_fit.model import ModelConfig
from cr_knee_fit.types_ import Primary

if __name__ == "__main__":
    analysis_name = "composition + lhaaso (epos); knee dependent on energy per nucleon"

    experiments_detailed = experiments.direct_experiments + [experiments.grapes]
    lhaaso = experiments.lhaaso_epos
    experiments_all_particle = [lhaaso]
    experiments_lnA = [lhaaso]

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
                    SpectralBreakConfig(fixed_lg_sharpness=None, quantity="E_n"),
                ],
                rescale_all_particle=True,
            ),
            shifted_experiments=[
                e for e in experiments_detailed + experiments_all_particle if e != experiments.ams02
            ],
        ),
        mcmc=McmcConfig(
            n_steps=500_000,
            n_walkers=128,
            processes=16,
            reuse_saved=True,
        ),
    )

    out_dir = Path(__file__).parent / "condor-out" / config.name
    out_dir.mkdir(parents=True, exist_ok=True)

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
            "request_cpus": str(config.mcmc.processes),
            "should_transfer_files": "IF_NEEDED",
        }
    )
    print()
    print(job)

    print("Submitting...")
    location = htcondor.Collector().locate(htcondor.DaemonTypes.Schedd, os.environ["SCHEDD_NAME"])  # type: ignore
    schedd = htcondor.Schedd(location)  # type: ignore
    schedd.submit(job)
    print("Done!")
