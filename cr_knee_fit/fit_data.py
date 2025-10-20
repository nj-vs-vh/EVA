import dataclasses
import itertools
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cr_knee_fit import experiments
from cr_knee_fit.constants import NON_FITTED_ALPHA
from cr_knee_fit.elements import Element
from cr_knee_fit.experiments import Experiment
from cr_knee_fit.utils import (
    LN_A_LABEL,
    energy_shift_suffix,
    label_energy_flux,
    legend_with_added_items,
)
from model.utils import DATA_DIR

DEFAULT_MARKER_SIZE = 4.0


def load_data(
    filename: str,
    x_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = str(DATA_DIR / filename)
    cols = (0, 1, 2, 3, 4, 5)
    data = np.loadtxt(path, usecols=cols)
    x = data[:, 0]
    mask = (x > x_bounds[0]) & (x < x_bounds[1])
    return (
        x[mask],
        data[mask, 1],  # y
        data[mask, 2:4],  # err stat
        data[mask, 4:6],  # err syst
    )


@dataclass
class GenericExperimentData:
    x: np.ndarray  # 1D

    y: np.ndarray

    err_stat: np.ndarray
    err_syst: np.ndarray

    experiment: Experiment

    custom_label: str | None = None

    def __post_init__(self) -> None:
        assert self.x.ndim == 1, "X must be 1-dimensional"
        assert self.y.ndim == 1, "Y must be 1-dimensional"
        npoints = self.x.size
        assert self.y.size == npoints, f"Bad Y size: {self.y.size} =/= {npoints}"
        assert self.err_stat.shape == (npoints, 2), (
            f"Bad stat error size: {self.err_stat.shape} =/= {(npoints, 2)}"
        )
        assert self.err_syst.shape == (npoints, 2), (
            f"Bad syst error size: {self.err_syst.shape} =/= {(npoints, 2)}"
        )

    @classmethod
    def load(
        cls,
        exp: Experiment,
        suffix: str,
        x_bounds: tuple[float, float],
        custom_label: str | None = None,
    ) -> "GenericExperimentData":
        x, y, stat, syst = load_data(
            filename=f"{exp.filename_prefix}_{suffix}.txt", x_bounds=x_bounds
        )
        return GenericExperimentData(
            x=x, y=y, err_stat=stat, err_syst=syst, experiment=exp, custom_label=custom_label
        )

    def plot(
        self,
        ax: Axes | None = None,
        color: Any | None = None,
        scale: float = 0,
        is_fitted: bool = True,
        marker_size: float = DEFAULT_MARKER_SIZE,
        label_override: str | None = None,
        add_legend_label: bool = True,
    ) -> Axes:
        if ax is None:
            _, ax = plt.subplots()
        x_factor = self.x**scale

        if label_override is not None:
            label = label_override
        else:
            label = self.experiment.name
            if self.custom_label is not None:
                label += " " + self.custom_label
        x_factor_2D = np.expand_dims(x_factor, axis=-1)

        alpha = 1.0 if is_fitted else NON_FITTED_ALPHA
        lines = ax.errorbar(
            self.x,
            x_factor * self.y,
            yerr=(x_factor_2D * self.err_stat).T,
            color=color,
            markersize=marker_size,
            elinewidth=0.75,
            # capsize=1.5,
            label=label if add_legend_label else None,
            linestyle="none",
            marker=self.experiment.marker,
            alpha=alpha,
        )
        ax.errorbar(
            self.x,
            x_factor * self.y,
            yerr=(x_factor_2D * self.err_syst).T,
            color=lines[0].get_color(),
            alpha=alpha * 0.33,
            elinewidth=marker_size,  # making systematic error bar as wide as the marker
            linestyle="none",
            marker="none",
        )
        return ax


SpectrumDataElementSpec = Element | None | tuple[Element, ...]


@dataclass
class CRSpectrumData:
    d: GenericExperimentData
    element: SpectrumDataElementSpec
    energy_scale_shift: float = 1.0

    @property
    def E(self) -> np.ndarray:
        """Energy in GeV"""
        return self.d.x

    @property
    def F(self) -> np.ndarray:
        """Flux in GeV^-1 m^-2 s^-1 sr^-1"""
        return self.d.y

    @property
    def F_err_stat(self) -> np.ndarray:
        return self.d.err_stat

    @property
    def F_err_syst(self) -> np.ndarray:
        return self.d.err_syst

    def scaled_flux(self, scale: float) -> np.ndarray:
        return self.F * (self.E**scale)

    def __post_init__(self) -> None:
        assert self.E.size > 0, "Empty spectrum data"

    def with_shifted_energy_scale(self, f: float) -> "CRSpectrumData":
        return CRSpectrumData(
            d=GenericExperimentData(
                x=self.d.x * f,
                y=self.d.y / f,
                err_stat=self.d.err_stat / f,
                err_syst=self.d.err_syst / f,
                experiment=self.d.experiment,
            ),
            element=self.element,
            energy_scale_shift=self.energy_scale_shift * f,
        )

    @classmethod
    def load(cls, exp: Experiment, p: Element, R_bounds: tuple[float, float]) -> "CRSpectrumData":
        return CRSpectrumData(
            d=GenericExperimentData.load(
                exp=exp,
                suffix=f"{p.name}_energy",
                x_bounds=(R_bounds[0] * p.Z, R_bounds[1] * p.Z),
            ),
            element=p,
        )

    @classmethod
    def load_all_particle(
        cls, exp: Experiment, max_energy: float | None = None
    ) -> "CRSpectrumData":
        return CRSpectrumData(
            d=GenericExperimentData.load(
                exp=exp,
                suffix="all_energy",
                x_bounds=(1e3, max_energy or np.inf),
            ),
            element=None,
        )

    def plot_label(self) -> str:
        if self.element is None:
            element_label = "all"
        elif isinstance(self.element, tuple):
            element_label = "+".join(p.name for p in self.element)
        else:
            element_label = self.element.name

        if (self.element, self.d.experiment) in {
            (None, experiments.dampe),
            (Element.C, experiments.dampe),
            (Element.O, experiments.dampe),
        }:
            prelim_suffix = " (prelim.)"
        else:
            prelim_suffix = ""

        return f"{self.d.experiment.name} {element_label}{prelim_suffix}" + energy_shift_suffix(
            self.energy_scale_shift
        )

    def plot(
        self,
        scale: float,
        ax: Axes | None = None,
        color: Any | None = None,
        add_legend_label: bool = True,
        is_fitted: bool = True,
        marker_size: float = DEFAULT_MARKER_SIZE,
    ) -> Axes:
        axes = self.d.plot(
            ax=ax,
            color=color
            or (
                self.element.color
                if isinstance(self.element, Element)
                else ("black" if self.element is None else None)
            ),
            scale=scale,
            is_fitted=is_fitted,
            marker_size=marker_size,
            label_override=self.plot_label(),
            add_legend_label=add_legend_label,
        )
        label_energy_flux(axes, scale)
        return axes


@dataclass
class DataConfig:
    experiments_elements: list[Experiment | tuple[Experiment, list[Element]]] = dataclasses.field(
        default_factory=list
    )
    experiments_all_particle: list[Experiment] = dataclasses.field(default_factory=list)
    experiments_lnA: list[Experiment] = dataclasses.field(default_factory=list)

    elements: list[Element] = dataclasses.field(default_factory=Element.regular)
    elements_R_bounds: tuple[float, float] = (5e2, 1e8)

    def __post_init__(self) -> None:
        self.elements_by_exp: dict[Experiment, list[Element]] = {}
        for exp_or_pair in self.experiments_elements:
            if isinstance(exp_or_pair, Experiment):
                self.elements_by_exp[exp_or_pair] = self.elements
            else:
                experiment, elements = exp_or_pair
                self.elements_by_exp[experiment] = elements
        self.elements_by_exp = {exp: els for exp, els in self.elements_by_exp.items() if els}

    @property
    def experiments_spectrum(self) -> list[Experiment]:
        return list(
            set(
                itertools.chain(
                    (exp_or_exp_elements for exp_or_exp_elements in self.elements_by_exp.keys()),
                    self.experiments_all_particle,
                )
            )
        )

    def excluding(self, other: "DataConfig") -> "DataConfig":
        return DataConfig(
            experiments_elements=[
                (exp, [el for el in elements if el not in other.elements_by_exp.get(exp, [])])
                for exp, elements in self.elements_by_exp.items()
            ],
            experiments_all_particle=list(
                set(self.experiments_all_particle).difference(other.experiments_all_particle)
            ),
            experiments_lnA=list(set(self.experiments_lnA).difference(other.experiments_lnA)),
            elements=self.elements.copy(),
            elements_R_bounds=self.elements_R_bounds,
        )


@dataclass
class Data:
    """Top-level container for a set of experimental data"""

    element_spectra: dict[Experiment, dict[Element, CRSpectrumData]]
    all_particle_spectra: dict[Experiment, CRSpectrumData]
    lnA: dict[Experiment, GenericExperimentData]

    config: DataConfig

    def experiments(self, spectra_only: bool = False) -> list[Experiment]:
        all = set[Experiment]()
        for e in itertools.chain(
            [exp for exp, elements in self.element_spectra.items() if elements],
            self.all_particle_spectra.keys(),
        ):
            all.add(e)
        if not spectra_only:
            for e in self.lnA.keys():
                all.add(e)
        return sorted(all)

    def is_empty(self) -> bool:
        return len(self.experiments()) > 0

    def all_spectra(self) -> Iterable[CRSpectrumData]:
        for element_spectra in self.element_spectra.values():
            yield from element_spectra.values()
        yield from self.all_particle_spectra.values()

    def E_min(self) -> float:
        return min([s.E.min() for s in self.all_spectra()])

    def E_max(self) -> float:
        return max([s.E.max() for s in self.all_spectra()])

    @classmethod
    def empty(cls) -> "Data":
        return Data(
            element_spectra={},
            all_particle_spectra={},
            lnA={},
            config=DataConfig(),
        )

    @classmethod
    def load(cls, config: DataConfig, verbose: bool = False) -> "Data":
        def log(s: str) -> None:
            if verbose:
                print(s)

        allparticle = {}
        for exp in config.experiments_all_particle:
            try:
                allparticle[exp] = CRSpectrumData.load_all_particle(exp)
                log(f"Loaded all particle data for {exp}...")
            except Exception as e:
                log(f"Failed to load all particle spectrum data for {exp}: {e}")

        lnA = {}
        for exp in config.experiments_lnA:
            try:
                lnA[exp] = GenericExperimentData.load(
                    exp,
                    suffix="lnA_energy",
                    x_bounds=(0, np.inf),
                    custom_label=LN_A_LABEL,
                )
                log(f"Loaded lnA data for {exp}...")
            except Exception as e:
                log(f"Failed to load lnA for {exp}: {e}")

        element_spectra: dict[Experiment, dict[Element, CRSpectrumData]] = {}
        for exp, elements in config.elements_by_exp.items():
            exp_data = dict[Element, CRSpectrumData]()
            for element in elements:
                try:
                    exp_data[element] = CRSpectrumData.load(exp, element, config.elements_R_bounds)
                    log(f"Loaded {element.name} data for {exp}...")
                except Exception as e:
                    log(f"Failed to load {element.name} spectrum from {exp}: {e}")
            element_spectra[exp] = exp_data

        return Data(
            element_spectra=element_spectra,
            all_particle_spectra=allparticle,
            lnA=lnA,
            config=config,
        )

    def plot_spectra(
        self, scale: float, describe: bool, is_fitted: bool, ax: Axes, legend: bool = True
    ):
        print_ = print if describe else lambda _: None
        print_("Data by element:")
        for exp, ps in self.element_spectra.items():
            print_(exp.name)
            for p, s in ps.items():
                print_(f"  {p.name}: {s.E.size} points from {s.E.min():.1e} to {s.E.max():.1e} GeV")
                s.plot(scale=scale, ax=ax, add_legend_label=False, is_fitted=is_fitted)
        print_("All particle data:")
        for exp, s in self.all_particle_spectra.items():
            print_(f"    {exp.name}: {s.E.size} points from {s.E.min():.1e} to {s.E.max():.1e} GeV")
            s.plot(scale=scale, ax=ax, add_legend_label=False, is_fitted=is_fitted)

        ax.set_xscale("log")
        ax.set_yscale("log")
        if legend:
            legend_with_added_items(
                ax,
                [
                    (exp.legend_artist(is_fitted=is_fitted), exp.name)
                    for exp in sorted(self.experiments(spectra_only=True))
                ],
                fontsize="x-small",
            )

    def plot_lnA(self, describe: bool, is_fitted: bool, ax: Axes):
        print_ = print if describe else lambda _: None
        print_("lnA data:")
        for exp, lnA_data in self.lnA.items():
            print_(
                f"    {exp.name}: {lnA_data.x.size} points from {lnA_data.x.min():.1e} to {lnA_data.x.max():.1e} GeV"
            )
            lnA_data.plot(ax=ax, is_fitted=is_fitted)
        ax.set_xscale("log")
        label_energy_flux(ax, scale=0)
        ax.set_ylabel(LN_A_LABEL)
        ax.legend(fontsize="xx-small")

    def plot(
        self,
        scale: float,
        describe: bool = False,
        is_fitted: bool = True,
        figure: Figure | None = None,
    ) -> Figure:
        if figure is None:
            if self.lnA:
                fig, axes = plt.subplots(ncols=2, figsize=(16, 6))
                axes = cast(Sequence[Axes], axes)
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                axes = [ax]
        else:
            fig = figure
            axes = fig.axes

        self.plot_spectra(scale=scale, describe=describe, is_fitted=is_fitted, ax=axes[0])
        if len(axes) > 1:
            self.plot_lnA(describe=describe, is_fitted=is_fitted, ax=axes[1])

        return fig
