import itertools
import warnings
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from cr_knee_fit.cr_model import (
    CosmicRaysModel,
    CosmicRaysModelConfig,
    SharedPowerLaw,
    SpectralBreak,
)
from cr_knee_fit.experiments import Experiment
from cr_knee_fit.fit_data import FitData
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts
from cr_knee_fit.types_ import Packable, Primary, most_abundant_stable_izotope_A
from cr_knee_fit.utils import legend_with_added_items


@dataclass
class ModelConfig:
    shifted_experiments: list[Experiment]
    population_configs: list[CosmicRaysModelConfig] = field(default_factory=list)

    cr_model_config: CosmicRaysModelConfig | None = None  # backwards compatibility

    def __post_init__(self) -> None:
        if self.cr_model_config is not None:
            warnings.warn("cr_model_config class is deprecated, use population_configs")
            if self.population_configs:
                raise ValueError(
                    "population_configs and cr_model_config parameters are mutually exclusive"
                )
            self.population_configs = [self.cr_model_config]

    def primaries(self, observed_only: bool = True) -> list[Primary]:
        return sorted(
            {
                p
                for p in itertools.chain.from_iterable(c.primaries for c in self.population_configs)
                if p is not Primary.Unobserved or not observed_only
            }
        )


@dataclass
class Model(Packable[ModelConfig]):
    populations: list[CosmicRaysModel]
    energy_shifts: ExperimentEnergyScaleShifts

    def __post_init__(self) -> None:
        assert self.populations, "populations list can't be empty"

    def pack(self) -> np.ndarray:
        chunks = [pop.pack() for pop in self.populations]
        chunks.append(self.energy_shifts.pack())
        return np.hstack(chunks)

    def labels(self, latex: bool) -> list[str]:
        return list(
            itertools.chain.from_iterable(m.labels(latex) for m in self.populations)
        ) + self.energy_shifts.labels(latex)

    def layout_info(self) -> ModelConfig:
        return ModelConfig(
            population_configs=[pop.layout_info() for pop in self.populations],
            shifted_experiments=self.energy_shifts.experiments,
        )

    @classmethod
    def unpack(cls, theta: np.ndarray, layout_info: ModelConfig) -> "Model":
        populations: list[CosmicRaysModel] = []
        offset = 0
        for pop_conf in layout_info.population_configs:
            population = CosmicRaysModel.unpack(theta[offset:], layout_info=pop_conf)
            offset += population.ndim()
            populations.append(population)

        energy_shifts = ExperimentEnergyScaleShifts.unpack(
            theta[offset:],
            layout_info=layout_info.shifted_experiments,
        )
        return Model(
            populations=populations,
            energy_shifts=energy_shifts,
        )

    def plot(self, fit_data: FitData, scale: float) -> Figure:
        fig, ax = plt.subplots(figsize=(10, 8))

        for exp, data_by_particle in fit_data.spectra.items():
            for _, data in data_by_particle.items():
                data.with_shifted_energy_scale(f=self.energy_shifts.f(exp)).plot(
                    scale=scale,
                    ax=ax,
                    add_label=False,
                )
        for _, data in fit_data.all_particle_spectra.items():
            data.plot(scale=scale, ax=ax, add_label=False)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ylim = ax.get_ylim()  # respecting ylim set by data

        for pop in self.populations:
            pop.plot(
                Emin=fit_data.E_min(),
                Emax=fit_data.E_max(),
                scale=scale,
                axes=ax,
                all_particle=len(fit_data.all_particle_spectra) > 0 and len(pop.primaries) > 1,
            )
        if len(self.populations) > 1:
            multipop_primaries = [
                primary
                for primary in Primary.all()
                if len([pop for pop in self.populations if primary in pop.primaries]) > 1
            ]
            E_grid = np.logspace(np.log10(fit_data.E_min()), np.log10(fit_data.E_max()), 100)
            E_factor = E_grid**scale
            for primary in multipop_primaries:
                ax.plot(
                    E_grid,
                    E_factor * self.compute_spectrum(E_grid, primary=primary),
                    label="Total " + primary.name,
                    color=primary.color,
                    linewidth=2,
                )
            if len(fit_data.all_particle_spectra) > 0:
                ax.plot(
                    E_grid,
                    E_factor * self.compute_spectrum(E_grid, primary=None),
                    label="Total all particle",
                    color="black",
                    linewidth=2,
                )

        legend_with_added_items(
            ax,
            [(e.legend_artist(), e.name) for e in fit_data.all_experiments()],
            fontsize="x-small",
        )
        ax.set_ylim(*ylim)
        return fig

    def compute_spectrum(self, E: np.ndarray, primary: Primary | None) -> np.ndarray:
        return sum(
            (
                (
                    pop.compute(E, primary=primary, contrib_to_all_particle=False)
                    if primary is not None
                    else pop.compute_all_particle(E)
                )
                for pop in self.populations
            ),
            start=np.zeros_like(E),
        )

    def compute_lnA(self, E: np.ndarray) -> np.ndarray:
        simple_primaries = self.layout_info().primaries()
        spectra = [self.compute_spectrum(E, primary=primary) for primary in simple_primaries]
        lnA = [np.log(p.A) for p in simple_primaries]

        # adding unobserved components per-population as they are not additive
        for pop in self.populations:
            if Primary.Unobserved not in pop.layout_info().primaries:
                continue
            spectra.append(pop.compute(E, primary=Primary.Unobserved))
            lnA.append(
                np.log(most_abundant_stable_izotope_A(round(pop.primary_Z(Primary.Unobserved))))
            )

        spectra_arr = np.vstack(spectra)
        lnA_arr = np.expand_dims(np.array(lnA), axis=1)
        return np.sum(spectra_arr * lnA_arr, axis=0) / np.sum(spectra_arr, axis=0)


if __name__ == "__main__":
    m = Model(
        populations=[
            CosmicRaysModel(
                base_spectra=[
                    SharedPowerLaw(
                        lgI_per_primary={p: np.random.random()},
                        alpha=np.random.random(),
                        lg_scale_contrib_to_all=0.1,
                    )
                    for p in Primary
                ],
                breaks=[
                    SpectralBreak(
                        lg_break=np.random.random(),
                        d_alpha=np.random.random(),
                        lg_sharpness=np.random.random(),
                        quantity="R",
                    )
                    for _ in range(5)
                ],
                all_particle_lg_shift=np.random.random(),
                unobserved_component_effective_Z=np.random.random(),
            )
            for _ in range(3)
        ],
        energy_shifts=ExperimentEnergyScaleShifts(
            lg_shifts={e: np.random.random() for e in [Experiment("a"), Experiment("b")]}
        ),
    )
    m.validate_packing()
