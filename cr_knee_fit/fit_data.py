import abc
import enum
from dataclasses import dataclass
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from cr_knee_fit.experiments import Experiment
from cr_knee_fit.types_ import Primary
from cr_knee_fit.utils import label_energy_flux
from model.utils import load_data


@dataclass
class CRSpectrumData:
    E: np.ndarray  # GeV

    F: np.ndarray  # (GeV m^2 s sr)^-1
    F_errlo: np.ndarray
    F_errhi: np.ndarray

    experiment: Experiment
    primary: Primary | None

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
            filename=exp.primary_filename(p),
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

    @classmethod
    def load_all_particle(cls, exp: Experiment, max_energy: float | None = None) -> "CRSpectrumData":
        data = load_data(
            filename=exp.all_particle_filename(),
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
            primary=None,
        )

    def plot_label(self) -> str:
        label = f"{self.experiment.name} {self.primary.name if self.primary is not None else 'all'}"
        if not np.isclose(self.energy_scale_shift, 1.0):
            shift_percent = abs(100 * (self.energy_scale_shift - 1))
            shift_sign = "+" if self.energy_scale_shift > 1 else "-"
            label += f" $(E \\; {shift_sign} {shift_percent:.1g} \\%)$"
        return label

    def plot(
        self,
        scale: float,
        ax: Axes | None = None,
        color: Any | None = None,
        add_label: bool = True,
    ) -> Axes:
        if ax is None:
            _, ax = plt.subplots()
        E_factor = self.E**scale
        ax.errorbar(
            self.E,
            E_factor * self.F,
            yerr=[E_factor * self.F_errlo, E_factor * self.F_errhi],
            color=color or (self.primary.color if self.primary else "black"),
            label=self.plot_label() if add_label else None,
            markersize=4.0,
            elinewidth=0.75,
            capsize=2.0,
            fmt=self.experiment.marker,
        )
        label_energy_flux(ax, scale)
        if add_label:
            ax.legend()
        return ax


@dataclass
class FitData:
    spectra: dict[Experiment, dict[Primary, CRSpectrumData]]
    all_particle_spectra: dict[Experiment, CRSpectrumData]
    R_bounds: tuple[float, float]

    def all_spectra(self) -> Iterable[CRSpectrumData]:
        for primary_spectra in self.spectra.values():
            yield from primary_spectra.values()
        yield from self.all_particle_spectra.values()

    def E_min(self) -> float:
        return min([s.E.min() for s in self.all_spectra()])

    def E_max(self) -> float:
        return max([s.E.max() for s in self.all_spectra()])

    @classmethod
    def load(
        cls,
        experiments_detailed: list[Experiment],
        experiments_all_particle: list[Experiment],
        primaries: list[Primary],
        R_bounds: tuple[float, float],
    ) -> "FitData":
        return FitData(
            spectra={exp: load_spectra(exp, primaries, R_bounds) for exp in experiments_detailed},
            all_particle_spectra={
                exp: CRSpectrumData.load_all_particle(exp) for exp in experiments_all_particle
            },
            R_bounds=R_bounds,
        )


def load_spectra(
    experiment: Experiment,
    primaries: list[Primary],
    R_bounds: tuple[float, float],
) -> dict[Primary, "CRSpectrumData"]:
    res = dict[Primary, CRSpectrumData]()
    for p in primaries:
        try:
            res[p] = CRSpectrumData.load(experiment, p, R_bounds)
        except Exception:
            pass
    return res
