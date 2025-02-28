import itertools
import warnings
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from cr_knee_fit.cr_model import (
    CosmicRaysModel,
    CosmicRaysModelConfig,
    SharedPowerLawSpectrum,
    SpectralBreak,
)
from cr_knee_fit.elements import (
    Z_to_element_name,
    low_energy_CR_spectra,
    unresolved_element_names,
)
from cr_knee_fit.experiments import Experiment
from cr_knee_fit.fit_data import FitData
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts
from cr_knee_fit.types_ import Element, Packable, isotope_average_A
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
            self.cr_model_config = None
        deduplicated_shifted_experiments: list[Experiment] = []
        for e in self.shifted_experiments:
            if e in deduplicated_shifted_experiments:
                continue
            deduplicated_shifted_experiments.append(e)
        self.shifted_experiments = deduplicated_shifted_experiments

    def elements(self, only_fixed_Z: bool) -> list[Element]:
        return sorted(
            {
                p
                for p in itertools.chain.from_iterable(
                    c.resolved_elements for c in self.population_configs
                )
                if p is not Element.FreeZ or not only_fixed_Z
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
                all_particle=len(fit_data.all_particle_spectra) > 0 and len(pop.all_elements) > 1,
            )
        if len(self.populations) > 1:
            multipop_elements = [
                element
                for element in Element.regular()
                if len([pop for pop in self.populations if element in pop.all_elements]) > 1
            ]
            E_grid = np.logspace(np.log10(fit_data.E_min()), np.log10(fit_data.E_max()), 100)
            E_factor = E_grid**scale
            for element in multipop_elements:
                ax.plot(
                    E_grid,
                    E_factor * self.compute_spectrum(E_grid, element=element),
                    label="Total " + element.name,
                    color=element.color,
                    linewidth=2,
                )
            if len(fit_data.all_particle_spectra) > 0:
                ax.plot(
                    E_grid,
                    E_factor * self.compute_spectrum(E_grid, element=None),
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

    def compute_spectrum(self, E: np.ndarray, element: Element | None) -> np.ndarray:
        return sum(
            (
                (
                    pop.compute(E, element=element, contrib_to_all_particle=False)
                    if element is not None
                    else pop.compute_all_particle(E)
                )
                for pop in self.populations
            ),
            start=np.zeros_like(E),
        )

    def compute_lnA(self, E: np.ndarray) -> np.ndarray:
        simple_elements = self.layout_info().elements(only_fixed_Z=True)
        spectra = [self.compute_spectrum(E, element=element) for element in simple_elements]
        lnA = [np.log(p.A) for p in simple_elements]

        # adding FreeZ components per-population as they are not additive
        for pop in self.populations:
            if Element.FreeZ not in pop.layout_info().resolved_elements:
                continue
            spectra.append(pop.compute(E, element=Element.FreeZ))
            lnA.append(np.log(isotope_average_A(round(pop.element_Z(Element.FreeZ)))))

        spectra_arr = np.vstack(spectra)
        lnA_arr = np.expand_dims(np.array(lnA), axis=1)
        return np.sum(spectra_arr * lnA_arr, axis=0) / np.sum(spectra_arr, axis=0)

    def compute_abundances(self, R: float) -> dict[Element | str, float]:
        pop_abundances = [pop.compute_abundances(R) for pop in self.populations]
        all_elements = list(set(itertools.chain.from_iterable(ab.keys() for ab in pop_abundances)))
        return {el: sum(ab.get(el, 0.0) for ab in pop_abundances) for el in all_elements}

    def plot_abundances(self) -> Figure:
        fitted_abundances = {
            el.name if isinstance(el, Element) else el: ab
            for el, ab in self.compute_abundances(R=1e3).items()
        }

        Z_grid = np.arange(1, 29, step=1, dtype=int)
        pre = []
        post = []
        for Z in Z_grid:
            element_name = Z_to_element_name[Z]
            pre.append(low_energy_CR_spectra[element_name][0])
            post.append(fitted_abundances.get(element_name, np.nan))
        pre = np.array(pre)
        post = np.array(post)

        fig, ax = plt.subplots()
        ax.plot(
            Z_grid,
            pre,
            label="Extrapolated from GeV range",
            zorder=-10,
        )
        is_unresolved_mask = np.array(
            [Z_to_element_name[Z] in unresolved_element_names for Z in Z_grid]
        )
        ax.scatter(
            Z_grid[~is_unresolved_mask],
            post[~is_unresolved_mask],
            marker="o",
            label="From TeV - PeV element data",
            color="tab:orange",
        )
        ax.scatter(
            Z_grid[is_unresolved_mask],
            post[is_unresolved_mask],
            marker="x",
            label="From all-particle (relative abundances fixed)",
            color="tab:orange",
        )
        ax.set_xlabel("Z")
        ax.set_ylabel("Abundance")
        ax.set_yscale("log")
        ax.grid(True, "major", "y")
        ax.legend()
        return fig


if __name__ == "__main__":
    m = Model(
        populations=[
            CosmicRaysModel(
                base_spectra=[
                    SharedPowerLawSpectrum(
                        lgI_per_element={p: np.random.random()},
                        alpha=np.random.random(),
                        lg_scale_contrib_to_all=0.1,
                    )
                    for p in Element
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
                free_Z=np.random.random(),
                unresolved_elements_spectrum=None,
            )
            for _ in range(3)
        ],
        energy_shifts=ExperimentEnergyScaleShifts(
            lg_shifts={e: np.random.random() for e in [Experiment("a"), Experiment("b")]}
        ),
    )
    m.validate_packing()
