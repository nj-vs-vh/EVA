import itertools
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cr_knee_fit.elements import Element
from cr_knee_fit.experiments import Experiment
from cr_knee_fit.utils import energy_shift_suffix, label_energy_flux, legend_with_added_items
from model.utils import load_data


@dataclass
class GenericExperimentData:
    x: np.ndarray  # typically GV or GeV

    y: np.ndarray
    y_errlo: np.ndarray
    y_errhi: np.ndarray

    experiment: Experiment

    label: str | None = None

    @classmethod
    def load(
        cls,
        exp: Experiment,
        suffix: str,
        x_bounds: tuple[float, float],
        label: str | None = None,
    ) -> "GenericExperimentData":
        data = load_data(
            filename=f"{exp.filename_prefix}_{suffix}.txt",
            slope=0,  # multiplying data by E^0 = leaving as-is
            norm=1,  # no renormalizing
            min_energy=x_bounds[0],
            max_energy=x_bounds[1],
        )
        return GenericExperimentData(
            x=data[0],
            y=data[1],
            y_errlo=data[2],
            y_errhi=data[3],
            experiment=exp,
            label=label,
        )

    def plot(
        self,
        ax: Axes | None = None,
        color: Any | None = None,
        add_label: bool = True,
        scale: float = 0,
        is_fitted: bool = True,
    ) -> Axes:
        if ax is None:
            _, ax = plt.subplots()
        x_factor = self.x**scale
        label = self.experiment.name
        if self.label is not None:
            label += " " + self.label
        ax.errorbar(
            self.x,
            x_factor * self.y,
            yerr=[x_factor * self.y_errlo, x_factor * self.y_errhi],
            color=color,
            markersize=4.0,
            elinewidth=0.75,
            capsize=2.0,
            label=label if add_label else None,
            fmt=self.experiment.marker,
            alpha=1.0 if is_fitted else 0.3,
        )
        if add_label:
            ax.legend()
        return ax


@dataclass
class CRSpectrumData:
    E: np.ndarray  # GeV

    F: np.ndarray  # (GeV m^2 s sr)^-1
    F_errlo: np.ndarray
    F_errhi: np.ndarray

    experiment: Experiment
    element: Element | None | tuple[Element, ...]

    energy_scale_shift: float = 1.0

    def with_shifted_energy_scale(self, f: float) -> "CRSpectrumData":
        return CRSpectrumData(
            E=self.E * f,
            F=self.F / f,
            F_errlo=self.F_errlo / f,
            F_errhi=self.F_errhi / f,
            experiment=self.experiment,
            element=self.element,
            energy_scale_shift=self.energy_scale_shift * f,
        )

    @classmethod
    def load(cls, exp: Experiment, p: Element, R_bounds: tuple[float, float]) -> "CRSpectrumData":
        data = load_data(
            filename=f"{exp.filename_prefix}_{p.name}_energy.txt",
            slope=0,  # multiplying data by E^0 = leaving as-is
            norm=1,  # no renormalizing
            min_energy=R_bounds[0] * p.Z,
            max_energy=R_bounds[1] * p.Z,
        )
        return CRSpectrumData(
            E=data[0],
            F=data[1],
            F_errlo=data[2],
            F_errhi=data[3],
            experiment=exp,
            element=p,
        )

    @classmethod
    def load_all_particle(
        cls, exp: Experiment, max_energy: float | None = None
    ) -> "CRSpectrumData":
        data = load_data(
            filename=f"{exp.filename_prefix}_all_energy.txt",
            slope=0,  # multiplying data by E^0 = leaving as-is
            norm=1,  # no renormalizing
            min_energy=1e3,  # 1 TeV
            max_energy=max_energy or 1e12,
        )
        return CRSpectrumData(
            E=data[0],
            F=data[1],
            F_errlo=data[2],
            F_errhi=data[3],
            experiment=exp,
            element=None,
        )

    def plot_label(self) -> str:
        if self.element is None:
            element_label = "all"
        elif isinstance(self.element, tuple):
            element_label = "+".join(p.name for p in self.element)
        else:
            element_label = self.element.name
        return f"{self.experiment.name} {element_label}" + energy_shift_suffix(
            self.energy_scale_shift
        )

    def plot(
        self,
        scale: float,
        ax: Axes | None = None,
        color: Any | None = None,
        add_label: bool = True,
        is_fitted: bool = True,
    ) -> Axes:
        if ax is None:
            _, ax = plt.subplots()
        E_factor = self.E**scale
        ax.errorbar(
            self.E,
            E_factor * self.F,
            yerr=[E_factor * self.F_errlo, E_factor * self.F_errhi],
            color=(
                color
                or (
                    self.element.color
                    if isinstance(self.element, Element)
                    else ("black" if self.element is None else None)
                )
            ),
            label=self.plot_label() if add_label else None,
            markersize=4.0,
            elinewidth=0.75,
            capsize=2.0,
            fmt=self.experiment.marker,
            alpha=1.0 if is_fitted else 0.3,
        )
        label_energy_flux(ax, scale)
        if add_label:
            ax.legend()
        return ax


@dataclass
class DataConfig:
    experiments_elements: list[Experiment]
    experiments_all_particle: list[Experiment]
    experiments_lnA: list[Experiment]

    # detailed spectra config
    elements: list[Element]
    elements_R_bounds: tuple[float, float] = (7e2, 1e8)

    @property
    def experiments_spectrum(self) -> list[Experiment]:
        return list(set(self.experiments_elements + self.experiments_all_particle))


@dataclass
class Data:
    """Top-level container for a set of experimental data"""

    element_spectra: dict[Experiment, dict[Element, CRSpectrumData]]
    all_particle_spectra: dict[Experiment, CRSpectrumData]
    lnA: dict[Experiment, GenericExperimentData]

    config: DataConfig

    def all_experiments(self) -> list[Experiment]:
        all = set[Experiment]()
        for e in itertools.chain(
            self.element_spectra.keys(),
            self.all_particle_spectra.keys(),
            self.lnA.keys(),
        ):
            all.add(e)
        return sorted(all)

    def all_spectra(self) -> Iterable[CRSpectrumData]:
        for element_spectra in self.element_spectra.values():
            yield from element_spectra.values()
        yield from self.all_particle_spectra.values()

    def E_min(self) -> float:
        return min([s.E.min() for s in self.all_spectra()])

    def E_max(self) -> float:
        return max([s.E.max() for s in self.all_spectra()])

    @classmethod
    def load(cls, config: DataConfig) -> "Data":
        return Data(
            element_spectra={
                exp: load_spectra(exp, config.elements, config.elements_R_bounds)
                for exp in config.experiments_elements
            },
            all_particle_spectra={
                exp: CRSpectrumData.load_all_particle(exp)
                for exp in config.experiments_all_particle
            },
            lnA={
                exp: GenericExperimentData.load(
                    exp,
                    suffix="lnA_energy",
                    x_bounds=(0, np.inf),
                    label="$ \\langle \\ln A \\rangle $",
                )
                for exp in config.experiments_lnA
            },
            config=config,
        )

    def plot(self, scale: float, describe: bool = False) -> Figure:
        print_ = print if describe else lambda _: None
        if self.lnA:
            fig, axes = plt.subplots(ncols=2, figsize=(20, 8))
            axes = cast(Sequence[Axes], axes)
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            axes = [ax]

        print_("Data by element:")
        for exp, ps in self.element_spectra.items():
            print_(exp.name)
            for p, s in ps.items():
                print_(f"  {p.name}: {s.E.size} points from {s.E.min():.1e} to {s.E.max():.1e} GeV")
                s.plot(scale=scale, ax=axes[0], add_label=False)
        print_("All particle data:")
        for exp, s in self.all_particle_spectra.items():
            print_(f"    {exp.name}: {s.E.size} points from {s.E.min():.1e} to {s.E.max():.1e} GeV")
            s.plot(scale=scale, ax=axes[0], add_label=False)
        print_("lnA data:")
        for exp, lnA_data in self.lnA.items():
            print_(
                f"    {exp.name}: {lnA_data.x.size} points from {lnA_data.x.min():.1e} to {lnA_data.x.max():.1e} GeV"
            )
            lnA_data.plot(ax=axes[1])

        [ax.set_xscale("log") for ax in axes]

        axes[0].set_yscale("log")
        legend_with_added_items(
            axes[0],
            [(exp.legend_artist(), exp.name) for exp in sorted(self.element_spectra.keys())],
            fontsize="x-small",
        )
        if len(axes) > 1:
            label_energy_flux(axes[1], scale=0)
            axes[1].set_ylabel("$ \\langle \\ln A \\rangle $")
            axes[1].legend(fontsize="xx-small")
        return fig


def load_spectra(
    experiment: Experiment,
    elements: list[Element],
    R_bounds: tuple[float, float],
) -> dict[Element, "CRSpectrumData"]:
    res = dict[Element, CRSpectrumData]()
    for p in elements:
        try:
            res[p] = CRSpectrumData.load(experiment, p, R_bounds)
        except Exception:
            pass
    return res
