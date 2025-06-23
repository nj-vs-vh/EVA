import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from bayesian_analysis import FitConfig, run_bayesian_analysis

OUT_DIR = Path(__file__).parent / "out"


class FileStdoutTee:
    def __init__(self, file: TextIO, write_to_stdout: bool):
        self.file = file
        self.write_to_stdout = write_to_stdout
        self._old_stdout = sys.stdout

    def write(self, data: str) -> int:
        if self.write_to_stdout:
            self._old_stdout.write(data)
        return self.file.write(data)

    def flush(self):
        if self.write_to_stdout:
            self._old_stdout.flush()
        self.file.flush()

    def __enter__(self):
        self._old_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exctype, excinst, exctb):
        sys.stdout = self._old_stdout


def run_local(config: FitConfig, log_to_stdout: bool = False) -> None:
    print("Running:")
    print(config)

    outdir = OUT_DIR / config.name
    if outdir.exists():
        print(f"Output directory exists: {outdir}")
        answer = input("Continue? This will overwrite some files! [Yn] ")
        if answer.lower() == "n":
            sys.exit(0)
    outdir.mkdir(exist_ok=True, parents=True)

    if log_to_stdout:
        logfile = outdir / "log.txt"
        with logfile.open("w") as log, FileStdoutTee(log, write_to_stdout=log_to_stdout):
            run_bayesian_analysis(config, outdir)


@dataclass
class LocalRunOptions:
    mcmc: bool
    export: bool

    @staticmethod
    def parse() -> "LocalRunOptions":
        parser = argparse.ArgumentParser()
        parser.add_argument("--mcmc", action="store_true")
        parser.add_argument("--export", action="store_true")
        args = parser.parse_args()
        return LocalRunOptions(mcmc=args.mcmc, export=args.export)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).absolute()
    if not run_dir.exists():
        print(f"Run directory must exist: {run_dir}")
    print(f"Rerunning analysis saved in {run_dir}")
    config_dump_file = run_dir / "config-dump.json"
    config = FitConfig.model_validate_json(config_dump_file.read_text())
    config.name = str(run_dir.relative_to(OUT_DIR))
    print(config)
    input("Press Enter to confirm")

    run_local(config)
