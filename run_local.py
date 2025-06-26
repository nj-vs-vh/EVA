import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from bayesian_analysis import FitConfig, run_bayesian_analysis

OUT_DIR = Path(__file__).parent / "out"


class FileStdoutTee(TextIO):
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


@dataclass
class LocalRunOptions:
    mcmc: bool
    export: bool
    overwrite: bool
    plots_only: bool
    no_stdout: bool

    args_raw: Any

    @staticmethod
    def parse(p: argparse.ArgumentParser | None = None) -> "LocalRunOptions":
        parser = p or argparse.ArgumentParser()
        parser.add_argument("--mcmc", action="store_true")
        parser.add_argument("--export", action="store_true")
        parser.add_argument("--force", "-f", action="store_true")
        parser.add_argument("--plots-only", action="store_true")
        parser.add_argument("--no-stdout", action="store_true")
        args = parser.parse_args()
        return LocalRunOptions(
            mcmc=args.mcmc,
            export=args.export,
            overwrite=args.force,
            plots_only=args.plots_only,
            no_stdout=args.no_stdout,
            args_raw=args,
        )


def run_local(config: FitConfig, opts: LocalRunOptions) -> None:
    if not opts.mcmc:
        config.mcmc = None
    if not opts.export:
        config.plots.export_opts.main = None
    if opts.plots_only:
        config.reuse_saved_models = True

    print("Running:")
    print(config)

    outdir = OUT_DIR / config.name
    if outdir.exists() and not opts.overwrite:
        print(f"Output directory exists: {outdir}")
        answer = input("Continue? This will overwrite some files! [Yn] ")
        if answer.lower() == "n":
            sys.exit(0)
    outdir.mkdir(exist_ok=True, parents=True)

    logfile = outdir / "log.txt"
    with logfile.open("w") as log, FileStdoutTee(log, write_to_stdout=(not opts.no_stdout)):  # type: ignore
        run_bayesian_analysis(config, outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run-dir")
    opts = LocalRunOptions.parse(parser)

    run_dir = Path(opts.args_raw.run_dir).absolute()
    if not run_dir.exists():
        print(f"Run directory must exist: {run_dir}")
    print(f"Re-running analysis saved in {run_dir}")
    config_dump_file = run_dir / "config-dump.json"
    config = FitConfig.model_validate_json(config_dump_file.read_text())
    config.name = str(run_dir.relative_to(OUT_DIR))
    print(config)
    input("Press Enter to confirm")

    run_local(config, opts)
