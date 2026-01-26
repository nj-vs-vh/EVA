import dataclasses
import itertools
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cr_knee_fit.cr_model import (
    CosmicRaysModel,
    CosmicRaysModelConfig,
    SharedPowerLawSpectrum,
    SpectralBreak,
    SpectralBreakConfig,
)
from cr_knee_fit.elements import (
    Element,
    Z_to_element_name,
    isotope_average_A,
    low_energy_CR_spectra,
    unresolved_element_names,
)
from cr_knee_fit.experiments import Experiment
from cr_knee_fit.fit_data import CRSpectrumData, Data
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts
from cr_knee_fit.types_ import Packable
from cr_knee_fit.utils import (
    E_GEV_LABEL,
    LN_A_LABEL,
    LegendItem,
    add_elements_lnA_secondary_axis,
    add_log_margin,
    energy_shift_suffix,
    legend_with_added_items,
)


@dataclasses.dataclass
class ModelConfig:
    shifted_experiments: list[Experiment]

    population_configs: list[CosmicRaysModelConfig] = dataclasses.field(default_factory=list)
    cr_model_config: CosmicRaysModelConfig | None = None  # backwards compatibility

    energy_scale_lg_uncertainty_override: dict[Experiment, float] = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.cr_model_config is not None:
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


@dataclasses.dataclass
class Model(Packable[ModelConfig]):
    populations: list[CosmicRaysModel]
    energy_shifts: ExperimentEnergyScaleShifts

    energy_scale_lg_uncertainty_override: dict[Experiment, float] = dataclasses.field(
        default_factory=dict
    )

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
            energy_scale_lg_uncertainty_override=self.energy_scale_lg_uncertainty_override,
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
            energy_scale_lg_uncertainty_override=layout_info.energy_scale_lg_uncertainty_override,
        )

    def plot_spectra(
        self,
        fit_data: Data,
        scale: float,
        validation_data: Data | None = None,
        axes: Axes | None = None,
    ) -> Figure:
        if axes is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            ax = axes
            fig = cast(Figure, ax.figure)

        legend_items_by_exp: dict[Experiment, LegendItem] = {}
        plot_allpart = False
        all_energies: list[float] = []
        for data_, is_fitted in ((fit_data, True), (validation_data, False)):
            if data_ is None:
                continue
            for exp, data_by_particle in data_.element_spectra.items():
                f_exp = self.energy_shifts.f(exp)
                for _, element_data in data_by_particle.items():
                    element_data = element_data.with_shifted_energy_scale(f=f_exp)
                    element_data.plot(
                        scale=scale,
                        ax=ax,
                        add_legend_label=False,
                        is_fitted=is_fitted,
                    )
                    all_energies.extend(element_data.E)
                    legend_items_by_exp.setdefault(
                        exp,
                        (
                            exp.legend_artist(is_fitted=is_fitted),
                            exp.name + energy_shift_suffix(f_exp),
                        ),
                    )
            for exp, allpart_data in data_.all_particle_spectra.items():
                f_exp = self.energy_shifts.f(exp)
                all_energies.extend(allpart_data.E)
                allpart_data = allpart_data.with_shifted_energy_scale(f_exp)
                allpart_data.plot(scale=scale, ax=ax, add_legend_label=False, is_fitted=is_fitted)
                legend_items_by_exp.setdefault(
                    exp,
                    (exp.legend_artist(is_fitted=is_fitted), exp.name + energy_shift_suffix(f_exp)),
                )
                plot_allpart = True

        E_min, E_max = add_log_margin(np.min(all_energies), np.max(all_energies))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ylim = ax.get_ylim()  # respecting ylim set by data

        for pop in self.populations:
            pop.plot(
                Emin=E_min,
                Emax=E_max,
                scale=scale,
                axes=ax,
                all_particle=plot_allpart and len(pop.all_elements) > 1,
            )
        if len(self.populations) > 1:
            multipop_elements = [
                element
                for element in Element.regular()
                if len([pop for pop in self.populations if element in pop.all_elements]) > 1
            ]
            E_grid = np.geomspace(E_min, E_max, 300)
            E_factor = E_grid**scale
            for element in multipop_elements:
                ax.plot(
                    E_grid,
                    E_factor * self.compute_spectrum(E_grid, element=element),
                    label="Total " + element.name,
                    color=element.color,
                    linewidth=2,
                )
            if plot_allpart:
                ax.plot(
                    E_grid,
                    E_factor * self.compute_spectrum(E_grid, element=None),
                    label="Total all particle",
                    color="black",
                    linewidth=2,
                )

        legend_with_added_items(
            ax,
            list(legend_items_by_exp.values()),
            fontsize="small",
            bbox_to_anchor=(0.00, 1.05, 1.0, 0.0),
            loc="lower left",
            fancybox=True,
            shadow=True,
            ncol=4,
        )
        ax.set_ylim(*ylim)
        ax.set_xlim(E_min, E_max)

        fig.tight_layout()
        # fig.canvas.draw()
        # legend_bbox = legend.get_window_extent()
        # legend_bbox_fig = legend_bbox.transformed(fig.transFigure.inverted())
        # legend_height = legend_bbox_fig.height
        # box = ax.get_position()
        # padding = 0.05
        # ax.set_position((box.x0, box.y0, box.width, box.height - legend_height - padding))
        # fig.canvas.draw()

        return fig

    def plot_lnA(
        self,
        fit_data: Data,
        validation_data: Data | None = None,
    ) -> Figure:
        fig, ax = plt.subplots(figsize=(10, 8))

        all_energies: list[float] = []
        legend_items = []
        for data, is_fitted in ((fit_data, True), (validation_data, False)):
            if data is None:
                continue
            for exp, lnA_data in data.lnA.items():
                f_exp = self.energy_shifts.f(exp)
                lnA_data = dataclasses.replace(lnA_data, x=lnA_data.x * f_exp)
                lnA_data.plot(
                    scale=0,
                    ax=ax,
                    add_legend_label=False,
                    color="black",
                    is_fitted=is_fitted,
                )
                legend_items.append(
                    (exp.legend_artist(is_fitted), exp.name + energy_shift_suffix(f_exp))
                )
                all_energies.extend(lnA_data.x)

        if not all_energies:
            return fig

        E_min = np.min(all_energies)
        E_max = np.max(all_energies)
        E_grid = np.geomspace(E_min, E_max, 100)
        ax.plot(
            E_grid,
            self.compute_lnA(E_grid),
            color="red",
        )

        ax.set_xscale("log")
        ax.set_xlabel(E_GEV_LABEL)
        ax.set_ylabel(LN_A_LABEL)
        legend_with_added_items(ax, legend_items, fontsize="x-small")
        add_elements_lnA_secondary_axis(ax)
        return fig

    def plot_aux_data(
        self,
        spectra_scale: float,
        validation_data: Data,
    ) -> Figure:
        aux_spectra = [d for d in validation_data.aux_data if isinstance(d, CRSpectrumData)]
        aux_spectra.sort(key=CRSpectrumData.element_label)
        grouped_spectra = list(
            (label, list(spectra))
            for label, spectra in itertools.groupby(aux_spectra, key=CRSpectrumData.element_label)
        )

        fig, ax_or_axes = plt.subplots(
            nrows=len(grouped_spectra), figsize=(10, 6 * len(grouped_spectra))
        )
        axes = [ax_or_axes] if isinstance(ax_or_axes, Axes) else ax_or_axes

        for ax, (label, spectra) in zip(axes, grouped_spectra):
            all_energies: list[float] = []
            legend_items = []
            for spectrum in spectra:
                exp = spectrum.d.experiment
                f_exp = self.energy_shifts.f(exp)
                spectrum.with_shifted_energy_scale(f=f_exp).plot(
                    ax=ax, scale=spectra_scale, add_legend_label=False
                )
                legend_items.append(
                    (exp.legend_artist(is_fitted=False), exp.name + energy_shift_suffix(f_exp))
                )
                all_energies.extend(spectrum.E)

            if not all_energies:
                continue
            ax.set_xscale("log")
            ax.set_yscale("log")
            ylim = ax.get_ylim()  # respecting ylim set by data

            E_min = np.min(all_energies)
            E_max = np.max(all_energies)
            E_grid = np.geomspace(E_min, E_max, 100)
            match spectra[0].element:
                case Element() | None as el:
                    y = self.compute_spectrum(E_grid, el)
                case tuple() as elements:
                    y = sum(
                        [self.compute_spectrum(E_grid, el) for el in elements],
                        start=np.zeros_like(E_grid),
                    )
            ax.plot(E_grid, E_grid**spectra_scale * y, color="tab:blue")
            ax.set_ylim(*ylim)

            ax.set_title(label)
            legend_with_added_items(ax, legend_items, fontsize="x-small")

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

        # adding FreeZ components per-population as they are potentially distinct
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
        pre_list: list[float] = []
        post_list: list[float] = []
        for Z in Z_grid:
            element_name = Z_to_element_name[Z]
            pre_list.append(low_energy_CR_spectra[element_name][0])
            post_list.append(fitted_abundances.get(element_name, np.nan))
        pre = np.array(pre_list)
        post = np.array(post_list)

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
                        config=SpectralBreakConfig(
                            quantity="R",
                            fixed_lg_sharpness=None,
                            lg_break_prior_limits=(4, 10),
                            is_softening=True,
                        ),
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
            lg_shifts={
                e: np.random.random()
                for e in [
                    Experiment("a", filename_stem="aaa"),
                    Experiment("b", filename_stem="bbb"),
                ]
            }
        ),
    )
    m.validate_packing()
