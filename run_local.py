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
    analysis_name = "additional-population-and-dsa-1"

    experiments_detailed = experiments.direct_experiments + [experiments.grapes]
    experiments_all_particle = [experiments.dampe, experiments.hawc, experiments.lhaaso_epos]
    experiments_lnA = [experiments.lhaaso_epos]

    def generate_guess() -> Model:
        initial_guess_lgI = {
            Element.H: -4,
            Element.He: -4.65,
            Element.C: -6.15,
            Element.O: -6.1,
            Element.Mg: -6.85,
            Element.Si: -6.9,
            Element.Fe: -6.9,
            Element.FreeZ: -8,
        }
        main_population = CosmicRaysModel(
            base_spectra=[
                SharedPowerLawSpectrum(
                    lgI_per_element={
                        element: stats.norm.rvs(loc=initial_guess_lgI[element], scale=0.05)
                        for element in Element.regular()
                    },
                    alpha=stats.norm.rvs(loc=2.6, scale=0.05),
                    lg_scale_contrib_to_all=stats.uniform.rvs(loc=0.01, scale=0.3),
                )
            ],
            breaks=[
                initial_guess_break(
                    SpectralBreakConfig(fixed_lg_sharpness=None, quantity="R"),
                    break_idx=i,
                )
                for i in range(3)
            ],
            all_particle_lg_shift=None,
            free_Z=None,
        )

        low_energy_population = CosmicRaysModel(
            base_spectra=[
                SharedPowerLawSpectrum.single_element(
                    Element.H,
                    lgI=stats.norm.rvs(loc=-4.5, scale=0.05),
                    alpha=stats.norm.rvs(loc=3, scale=0.1),
                )
            ],
            breaks=[],
            all_particle_lg_shift=None,
            free_Z=None,
            population_meta=PopulationMetadata(
                name="Low energy",
                linestyle="--",
            ),
        )
        return Model(
            populations=[main_population, low_energy_population],
            energy_shifts=ExperimentEnergyScaleShifts(
                lg_shifts={
                    exp: stats.norm.rvs(loc=0, scale=0.01)
                    for exp in set(experiments_detailed + experiments_all_particle)
                }
            ),
        )

    generate_guess().validate_packing()

    config = FitConfig.from_guessing_func(
        name=analysis_name,
        experiments_detailed=experiments_detailed,
        experiments_all_particle=experiments_all_particle,
        experiments_lnA=experiments_lnA,
        mcmc=McmcConfig(
            n_steps=300_000,
            n_walkers=128,
            processes=8,
            reuse_saved=True,
        ),
        generate_guess=generate_guess,
    )
    run_local(config)
