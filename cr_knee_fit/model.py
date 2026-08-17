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
    PopulationMetadata,
    SharedPowerLawSpectrum,
    SpectralBreak,
    SpectralBreakConfig,
)
from cr_knee_fit.crams_model import CramsModel, CramsModelConfig
from cr_knee_fit.elements import Element, isotope_average_A
from cr_knee_fit.experiments import Experiment
from cr_knee_fit.fit_data import (
    EMPTY_DATA,
    CRSpectrumData,
    Data,
    DataConfig,
    FluxRatio,
    FluxRatioData,
    GenericExperimentData,
    SpectrumDataSpec,
    spectrum_data_spec_label,
)
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts
from cr_knee_fit.types_ import Packable
from cr_knee_fit.utils import (
    E_GEV_LABEL,
    LN_A_LABEL,
    CharacteristicQuantity,
    LegendItem,
    add_elements_lnA_secondary_axis,
    add_log_margin,
    energy_shift_suffix,
    legend_with_added_items,
    quantity_label,
)


@dataclasses.dataclass
class ModelConfig:
    shifted_experiments: list[Experiment]

    crams_config: CramsModelConfig | None = None

    population_configs: list[CosmicRaysModelConfig] = dataclasses.field(default_factory=list)
    # single population, for backwards compatibility
    cr_model_config: CosmicRaysModelConfig | None = None

    # if specified, these uncertainties are used instead of the default one hard-coded in inference.py
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
        all = {
            element
            for element in itertools.chain.from_iterable(
                c.resolved_elements for c in self.population_configs
            )
            if element is not Element.FreeZ or not only_fixed_Z
        }
        if self.crams_config is not None:
            all.update(self.crams_config.elements)
        return sorted(all)


@dataclasses.dataclass
class Model(Packable[ModelConfig]):
    populations: list[CosmicRaysModel]
    energy_shifts: ExperimentEnergyScaleShifts

    crams: CramsModel | None = None

    energy_scale_lg_uncertainty_override: dict[Experiment, float] = dataclasses.field(
        default_factory=dict
    )

    def pack(self) -> np.ndarray:
        chunks: list[np.ndarray] = []
        if self.crams is not None:
            chunks.append(self.crams.pack())
        chunks.extend(pop.pack() for pop in self.populations)
        chunks.append(self.energy_shifts.pack())
        return np.hstack(chunks)

    def ml_bounds(self) -> list[tuple[float, float] | None] | None:
        if self.crams is None:
            return None
        crams_bounds = self.crams.ml_bounds()
        if crams_bounds is None:
            return None
        return crams_bounds + [None] * (self.ndim() - self.crams.ndim())

    def labels(self, latex: bool) -> list[str]:
        return (
            (self.crams.labels(latex) if self.crams is not None else [])
            + list(itertools.chain.from_iterable(m.labels(latex) for m in self.populations))
            + self.energy_shifts.labels(latex)
        )

    def layout_info(self) -> ModelConfig:
        return ModelConfig(
            crams_config=self.crams.layout_info() if self.crams is not None else None,
            population_configs=[pop.layout_info() for pop in self.populations],
            shifted_experiments=self.energy_shifts.experiments,
            energy_scale_lg_uncertainty_override=self.energy_scale_lg_uncertainty_override,
        )

    @classmethod
    def unpack(cls, theta: np.ndarray, layout_info: ModelConfig) -> "Model":
        offset = 0

        if layout_info.crams_config is not None:
            crams = CramsModel.unpack(theta, layout_info=layout_info.crams_config)
            offset += crams.ndim()
        else:
            crams = None

        populations: list[CosmicRaysModel] = []
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
            crams=crams,
            energy_shifts=energy_shifts,
            energy_scale_lg_uncertainty_override=layout_info.energy_scale_lg_uncertainty_override,
        )

    def plot_spectra(
        self,
        fit_data: Data,
        scale: float,
        validation_data: Data = EMPTY_DATA,
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
            for exp, data_by_particle in data_.element_spectra.items():
                f_exp = self.energy_shifts.f(exp)
                for element_data in data_by_particle.values():
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

        self._plot_predictions(
            ax,
            E_min,
            E_max,
            scale,
            elements=sorted(set(fit_data.elements() + validation_data.elements())),
            allparticle=plot_allpart,
            caption_elements=True,
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
        validation_data: Data = EMPTY_DATA,
    ) -> Figure | None:
        fig, ax = plt.subplots(figsize=(10, 8))

        all_energies: list[float] = []
        legend_items = []
        for data, is_fitted in ((fit_data, True), (validation_data, False)):
            for lnA_data in data.lnA:
                exp = lnA_data.experiment
                f_exp = self.energy_shifts.f(exp)
                lnA_data = lnA_data.with_shifted_grid(f_exp)
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
            return None

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

    def _plot_flux_ratios_per_quantity(
        self,
        ax: Axes,
        fit_data: Data,
        validation_data: Data,
        only_quantity: CharacteristicQuantity,
        only_ratio: FluxRatio | None = None,
    ) -> bool:
        all_Q: list[float] = []
        ratios_to_plot: set[FluxRatio] = set()
        for data, is_fitted in ((fit_data, True), (validation_data, False)):
            for fr in data.flux_ratios:
                if fr.quantity != only_quantity:
                    continue
                if only_ratio is not None and fr.ratio != only_ratio:
                    continue
                fr.plot(
                    ax=ax,
                    add_legend_label=True,
                    is_fitted=is_fitted,
                )
                all_Q.extend(fr.Q)
                ratios_to_plot.add(fr.ratio)

        if not all_Q:
            return False

        Q_min, Q_max = add_log_margin(np.min(all_Q), np.max(all_Q))
        Q = np.geomspace(Q_min, Q_max, 100)
        for ratio in ratios_to_plot:
            ax.plot(
                Q,
                self.compute_flux_ratio(Q, fr=ratio, quantity=only_quantity),
                color=ratio.color(),
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(quantity_label(only_quantity))
        ax.set_ylabel("Flux ratio")
        ax.set_xlim(Q_min, Q_max)
        ax.legend()
        return True

    def plot_flux_ratios(
        self,
        fit_data: Data,
        validation_data: Data = EMPTY_DATA,
    ) -> Figure | None:
        fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
        something_plotted = False
        for i, q in enumerate(("E_n", "R")):
            ax = axes[i]
            if self._plot_flux_ratios_per_quantity(
                ax=ax,
                fit_data=fit_data,
                validation_data=validation_data,
                only_quantity=q,  # type: ignore
            ):
                something_plotted = True
            else:
                ax.remove()
        fig.tight_layout()
        if something_plotted:
            return fig
        else:
            return None

    def _chi2_label(self, data: CRSpectrumData | GenericExperimentData | FluxRatioData):
        # private imports to avoid circular import error
        from cr_knee_fit.inference import DEFAULT_CHI2_METHOD, Chi2Method, loglikelihood

        match data:
            case CRSpectrumData():
                oneshot_data = Data(spectra=[data], config=DataConfig())
            case GenericExperimentData():
                oneshot_data = Data(lnA=[data], config=DataConfig())
            case FluxRatioData():
                oneshot_data = Data(flux_ratios=[data], config=DataConfig())

        chi2s: list[float] = []
        run_methods: list[Chi2Method] = (
            ["correlated", "dimidated"] if DEFAULT_CHI2_METHOD == "correlated" else ["dimidated"]
        )
        for method in run_methods:
            chi2s.append(
                -2
                * loglikelihood(
                    model_or_theta=self,
                    fit_data=oneshot_data,
                    config=self.layout_info(),
                    chi2_method=method,
                )
            )
        chi2_main = chi2s[0]
        if np.isnan(chi2_main):
            return "$ \\chi^2 $ not available"
        res = f"$\\chi^2 \\; / \\; n_\\text{{data}} = {chi2_main:.2g} \\; / \\; {data.size()}$"
        if len(chi2s) > 1:
            res += f" (uncorrelated $\\chi^2 = {chi2s[1]:.2g}$)"
        return res

    def plot_all_datasets(
        self,
        fit_data: Data,
        spectra_scale: float,
        validation_data: Data = EMPTY_DATA,
    ) -> dict[tuple[Experiment, str], Figure]:

        res: dict[tuple[Experiment, str], Figure] = {}
        for data_, is_fitted in ((fit_data, True), (validation_data, False)):
            for spectrum in data_.spectra:
                f_exp = self.energy_shifts.f(spectrum.experiment)
                spectrum = spectrum.with_shifted_energy_scale(f=f_exp)
                ax = spectrum.plot(scale=spectra_scale, is_fitted=is_fitted)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.legend()
                Emin, Emax = add_log_margin(spectrum.E[0], spectrum.E[-1])
                E = np.geomspace(Emin, Emax, 100)
                prediction = self.compute_spectrum(E, element=spectrum.spec, quantity="E")
                ax.plot(E, E**spectra_scale * prediction, color="k", linewidth=2)
                ax.set_title(self._chi2_label(spectrum))
                ax.set_xlim(Emin, Emax)
                ax.figure.tight_layout()  # type: ignore
                res[(spectrum.experiment, spectrum.element_label())] = ax.figure  # type: ignore

            for lnA in data_.lnA:
                lnA = lnA.with_shifted_grid(self.energy_shifts.f(lnA.experiment))
                ax = lnA.plot(is_fitted=is_fitted)
                ax.set_xscale("log")
                ax.legend()
                Emin, Emax = add_log_margin(lnA.x[0], lnA.x[-1])
                E = np.geomspace(Emin, Emax, 100)
                prediction = self.compute_lnA(E)
                ax.plot(E, prediction, color="k", linewidth=2)
                ax.set_title(self._chi2_label(lnA))
                ax.set_xlim(Emin, Emax)
                ax.figure.tight_layout()  # type: ignore
                res[(lnA.experiment, "lnA")] = ax.figure  # type: ignore

            for flux_ratio in data_.flux_ratios:
                # NOTE: we don't apply energy shifts to flux ratios as they are measured in R
                ax = flux_ratio.plot(is_fitted=is_fitted)
                ax.set_xscale("log")
                ax.legend()
                Qmin, Qmax = add_log_margin(flux_ratio.Q[0], flux_ratio.Q[-1])
                Q = np.geomspace(Qmin, Qmax, 100)
                prediction = self.compute_flux_ratio(
                    Q,
                    fr=flux_ratio.ratio,
                    quantity=flux_ratio.quantity,
                )
                ax.plot(Q, prediction, color="k", linewidth=2)
                ax.set_title(self._chi2_label(flux_ratio))
                ax.set_xlim(Qmin, Qmax)
                ax.figure.tight_layout()  # type: ignore
                res[
                    (
                        flux_ratio.d.experiment,
                        f"ratio_{flux_ratio.ratio.num.name}_over_{flux_ratio.ratio.denom.name}",
                    )
                ] = ax.figure  # type: ignore

        return res

    def _plot_predictions(
        self,
        ax: Axes,
        E_min: float,
        E_max: float,
        scale: float,
        elements: list[Element],
        allparticle: bool,
        caption_elements: bool,
    ):
        if self.crams is not None:
            self.crams.plot(
                Emin=E_min,
                Emax=E_max,
                scale=scale,
                axes=ax,
                all_particle=allparticle,
                elements=elements,
                caption_elements=caption_elements,
            )
        for pop in self.populations:
            pop.plot(
                Emin=E_min,
                Emax=E_max,
                scale=scale,
                axes=ax,
                all_particle=allparticle and len(pop.all_elements) > 1,
                elements=elements,
                caption_elements=caption_elements,
            )
        if int(self.crams is not None) + len(self.populations) > 1:
            per_pop_elements = [
                element
                for element in elements
                if (
                    len([pop for pop in self.populations if element in pop.all_elements])
                    > (1 if self.crams is None else 0)
                )
            ]
            E_grid = np.geomspace(E_min, E_max, 300)
            E_factor = E_grid**scale
            for plot_element in per_pop_elements:
                ax.plot(
                    E_grid,
                    E_factor * self.compute_spectrum(E_grid, element=plot_element, quantity="E"),
                    label="Total " + plot_element.name,
                    color=plot_element.color,
                    linewidth=2,
                )
            if allparticle:
                ax.plot(
                    E_grid,
                    E_factor * self.compute_spectrum(E_grid, element=None, quantity="E"),
                    label="Total all particle",
                    color="black",
                    linewidth=2,
                )

    def _plot_spectrum(
        self,
        spec: SpectrumDataSpec,
        fit_data: Data,
        scale: float,
        validation_data: Data = EMPTY_DATA,
        axes: Axes | None = None,
    ) -> Figure:
        if axes is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            ax = axes
            fig = cast(Figure, ax.figure)

        legend_items_by_exp: dict[Experiment, LegendItem] = {}
        all_energies: list[float] = []
        for data_, is_fitted in ((fit_data, True), (validation_data, False)):
            for sd in data_.spectra:
                if sd.spec != spec:
                    continue
                exp = sd.experiment
                f_exp = self.energy_shifts.f(exp)
                sd = sd.with_shifted_energy_scale(f=f_exp)
                sd.plot(scale=scale, ax=ax, add_legend_label=False, is_fitted=is_fitted)
                all_energies.extend(sd.E)
                legend_items_by_exp.setdefault(
                    exp,
                    (
                        exp.legend_artist(is_fitted=is_fitted),
                        exp.name + energy_shift_suffix(f_exp),
                    ),
                )

        E_min, E_max = add_log_margin(np.min(all_energies), np.max(all_energies))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ylim = ax.get_ylim()  # respecting ylim set by data

        match spec:
            case Element():
                plot_elements = [spec]
            case tuple():
                plot_elements = list(spec)
            case None:
                plot_elements = Element.regular()
        plot_allpart = spec is None

        self._plot_predictions(
            ax,
            E_min,
            E_max,
            scale,
            elements=plot_elements,
            allparticle=plot_allpart,
            caption_elements=spec is not None,
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

    def plot_all_observables(
        self,
        fit_data: Data,
        spectra_scale: float,
        validation_data: Data = EMPTY_DATA,
    ) -> dict[str, Figure]:
        res: dict[str, Figure] = {}

        spectrum_specs = {s.spec for s in fit_data.spectra + validation_data.spectra}
        for spec in spectrum_specs:
            fig, ax = plt.subplots()
            self._plot_spectrum(
                spec=spec,
                fit_data=fit_data,
                scale=spectra_scale,
                validation_data=validation_data,
                axes=ax,
            )
            res[spectrum_data_spec_label(spec)] = fig

        flux_ratios = {frd.ratio for frd in fit_data.flux_ratios + validation_data.flux_ratios}
        for fr in flux_ratios:
            for quantity in ("R", "E_n"):
                fig, ax = plt.subplots()
                if self._plot_flux_ratios_per_quantity(
                    ax=ax,
                    fit_data=fit_data,
                    validation_data=validation_data,
                    only_quantity=quantity,
                    only_ratio=fr,
                ):
                    res[f"ratio_{fr.num.name}_over_{fr.denom.name}_vs_{quantity}"] = fig

        return res

    def compute_spectrum(
        self,
        Q: np.ndarray,
        element: SpectrumDataSpec,
        quantity: CharacteristicQuantity,
    ) -> np.ndarray:
        if quantity != "E" and element is None:
            raise ValueError("All-particle spectrum can only be computed in total energy")

        if isinstance(element, tuple):
            element_group = element
            return sum(
                (self.compute_spectrum(Q, element, quantity=quantity) for element in element_group),
                np.zeros_like(Q),
            )

        components: list[np.ndarray] = []
        if self.crams is not None:
            components.append(
                self.crams.compute_spectrum(Q, element, quantity=quantity)
                if element is not None
                else self.crams.compute_all_particle_spectrum(Q)
            )

        for pop in self.populations:
            if element is not None:
                if pop.has_element(element):
                    components.append(
                        pop.compute_spectrum(
                            Q,
                            element,
                            quantity=quantity,
                            contrib_to_all_particle=False,
                        )
                    )
            else:
                components.append(pop.compute_all_particle_spectrum(Q))

        return sum(components, start=np.zeros_like(Q))

    def compute_flux_ratio(
        self,
        Q: np.ndarray,
        fr: FluxRatio,
        quantity: CharacteristicQuantity,
    ) -> np.ndarray:
        num = self.compute_spectrum(Q, element=fr.num, quantity=quantity)
        denom = self.compute_spectrum(Q, element=fr.denom, quantity=quantity)
        return num / denom

    def compute_lnA(self, E: np.ndarray) -> np.ndarray:
        elements = self.layout_info().elements(only_fixed_Z=True)
        spectra = [self.compute_spectrum(E, element=element, quantity="E") for element in elements]
        lnA = [np.log(p.A) for p in elements]

        # adding FreeZ components per-population as they are potentially distinct
        for pop in self.populations:
            if Element.FreeZ not in pop.layout_info().resolved_elements:
                continue
            spectra.append(pop.compute_spectrum(E, element=Element.FreeZ, quantity="E"))
            lnA.append(np.log(isotope_average_A(round(pop.element_Z(Element.FreeZ)))))

        spectra_arr = np.vstack(spectra)
        lnA_arr = np.expand_dims(np.array(lnA), axis=1)
        return np.sum(spectra_arr * lnA_arr, axis=0) / np.sum(spectra_arr, axis=0)

    def compute_abundances(self, R: float) -> dict[Element | str, float]:
        abundances = [pop.compute_abundances(R) for pop in self.populations]
        if self.crams is not None:
            abundances.append(self.crams.compute_abundances(R))
        all_elements = list(set(itertools.chain.from_iterable(ab.keys() for ab in abundances)))
        return {el: sum(ab.get(el, 0.0) for ab in abundances) for el in all_elements}

    def plot_abundances(self) -> Figure:
        fitted_abundances = {
            el.name if isinstance(el, Element) else el: ab
            for el, ab in self.compute_abundances(R=1e3).items()
        }

        Z_grid = np.arange(1, 29, step=1, dtype=int)
        post_list: list[float] = []
        for Z in Z_grid:
            element_name = Element(int(Z)).name
            post_list.append(fitted_abundances.get(element_name, np.nan))
        post = np.array(post_list)

        fig, ax = plt.subplots()
        ax.scatter(
            Z_grid,
            post,
            marker="o",
            label="From TeV - PeV element data",
            color="tab:orange",
        )
        ax.scatter(
            Z_grid,
            post,
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
    for m in [
        Model(
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
                    population_meta=PopulationMetadata(name=f"Pop #{pop_idx + 1}"),
                )
                for pop_idx in range(3)
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
        ),
        Model(
            populations=[],
            energy_shifts=ExperimentEnergyScaleShifts(
                lg_shifts={
                    e: np.random.random()
                    for e in [
                        Experiment("a", filename_stem="aaa"),
                        Experiment("b", filename_stem="bbb"),
                    ]
                }
            ),
        ),
        Model(
            populations=[],
            crams=CramsModel.make(up2PeV=False, source_feature="break"),
            energy_shifts=ExperimentEnergyScaleShifts(
                lg_shifts={Experiment("a", filename_stem="aaa"): np.random.random()}
            ),
        ),
    ]:
        print()
        m.print_params()
        m.validate_packing()
