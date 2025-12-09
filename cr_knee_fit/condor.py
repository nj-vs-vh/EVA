import os
import sys
from pathlib import Path

import htcondor  # type: ignore

from cr_knee_fit.analysis import FitConfig


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
