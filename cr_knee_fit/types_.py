import abc
from matplotlib.axes import Axes
import enum
from dataclasses import dataclass
from typing import Any, Generic, Type, TypeVar

import matplotlib.pyplot as plt
import numpy as np

from cr_knee_fit.plotting import label_energy_flux
from model.utils import load_data

T = TypeVar("T")
LayoutInfo = TypeVar("LayoutInfo")


class Packable(Generic[LayoutInfo], abc.ABC):
    def ndim(self) -> int:
        return self.pack().size

    @abc.abstractmethod
    def pack(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def labels(self) -> list[str]:
        ...

    @abc.abstractmethod
    def layout_info(self) -> LayoutInfo:
        ...

    @classmethod
    @abc.abstractmethod
    def unpack(cls: Type[T], theta: np.ndarray, layout_info: LayoutInfo) -> T:
        ...

    def validate_packing(self) -> None:
        packed = self.pack()
        labels = self.labels()
        assert len(packed) == len(labels)
        assert len(packed) == self.ndim()
        assert self.unpack(packed, layout_info=self.layout_info()) == self


_PRIMARY_CMAP = plt.colormaps["turbo"]


class Primary(enum.IntEnum):
    H = 1
    He = 2
    C = 6
    O = 8
    Mg = 12
    Si = 14
    Fe = 26

    @property
    def Z(self) -> int:
        return self.value

    @property
    def color(self) -> Any:
        all = sorted(Primary)
        idx = all.index(self)
        return _PRIMARY_CMAP(idx / (len(all) - 1))


class Experiment(enum.StrEnum):
    AMS02 = "AMS-02"
    CALET = "CALET"
    DAMPE = "DAMPE"
    CREAM = "CREAM"

    def available_primaries(self) -> list[Primary]:
        match self:
            case Experiment.AMS02:
                return list(Primary)
            case Experiment.CALET:
                return [
                    Primary.H,
                    # Primary.He,
                    Primary.C,
                    Primary.O,
                    Primary.Fe,
                ]
            case Experiment.DAMPE:
                return [Primary.H, Primary.He]
            case Experiment.CREAM:
                return list(Primary)

    def marker(self) -> str:
        match self:
            case Experiment.AMS02:
                return "o"
            case Experiment.CALET:
                return "s"
            case Experiment.DAMPE:
                return "v"
            case Experiment.CREAM:
                return "*"

    def load_spectra(
        self, primaries: list[Primary], R_bounds: tuple[float, float]
    ) -> dict[Primary, "CRSpectrumData"]:
        primaries = sorted(set(primaries).intersection(self.available_primaries()))
        return {p: CRSpectrumData.load(self, p, R_bounds=R_bounds) for p in primaries}


@dataclass
class CRSpectrumData:
    E: np.ndarray  # GeV

    F: np.ndarray  # (GeV m^2 s sr)^-1
    F_errlo: np.ndarray
    F_errhi: np.ndarray

    experiment: Experiment
    primary: Primary

    energy_scale_shift: float = 1.0

    def with_shifted_energy_scale(self, f: float) -> "CRSpectrumData":
        return CRSpectrumData(
            E=self.E * f,
            F=self.F / f,
            F_errlo=self.F_errlo / f,
            F_errhi=self.F_errhi / f,
            experiment=self.experiment,
            primary=self.primary,
            energy_scale_shift=self.energy_scale_shift * f,
        )

    @classmethod
    def load(cls, exp: Experiment, p: Primary, R_bounds: tuple[float, float]) -> "CRSpectrumData":
        data = load_data(
            filename=f"{exp.value}_{p.name}_energy.txt",
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
            primary=p,
        )

    def plot(
        self,
        scale: float,
        ax: Axes | None = None,
        color: Any | None = None,
    ) -> Axes:
        if ax is None:
            _, ax = plt.subplots()
        E_factor = self.E**scale
        label = f"{self.experiment.value} {self.primary.name}"
        if not np.isclose(self.energy_scale_shift, 1.0):
            label += f" $(E \\times {self.energy_scale_shift:.3g})$"
        ax.errorbar(
            self.E,
            E_factor * self.F,
            yerr=[E_factor * self.F_errlo, E_factor * self.F_errhi],
            color=color or self.primary.color,
            label=label,
            markersize=6.0,
            elinewidth=1.8,
            capthick=1.8,
            fmt=self.experiment.marker(),
        )
        label_energy_flux(ax, scale)
        ax.legend()
        return ax


@dataclass
class FitData:
    spectra: dict[Experiment, dict[Primary, CRSpectrumData]]
    R_bounds: tuple[float, float]

    @classmethod
    def load(
        cls,
        experiments: list[Experiment],
        primaries: list[Primary],
        R_bounds: tuple[float, float],
    ) -> "FitData":
        return FitData(
            spectra={e: e.load_spectra(primaries, R_bounds) for e in experiments},
            R_bounds=R_bounds,
        )
