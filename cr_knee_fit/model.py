from dataclasses import dataclass
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from cr_knee_fit.galactic import GalacticCR, RigidityBreak
from cr_knee_fit.shifts import ExperimentEnergyScaleShifts
from cr_knee_fit.spectrum import PowerLaw
from cr_knee_fit.types_ import Experiment, FitData, Packable, Primary


@dataclass
class ModelConfig:
    primaries: list[Primary]
    shifted_experiments: list[Experiment]


@dataclass
class Model(Packable[ModelConfig]):
    cr: GalacticCR
    energy_shifts: ExperimentEnergyScaleShifts

    def pack(self) -> np.ndarray:
        return np.hstack((self.cr.pack(), self.energy_shifts.pack()))

    def labels(self) -> list[str]:
        return self.cr.labels() + self.energy_shifts.labels()

    def layout_info(self) -> ModelConfig:
        return ModelConfig(
            primaries=self.cr.primaries,
            shifted_experiments=self.energy_shifts.experiments,
        )

    @classmethod
    def unpack(cls, theta: np.ndarray, layout_info: ModelConfig) -> "Model":
        cr = GalacticCR.unpack(theta, layout_info=layout_info.primaries)
        energy_shifts = ExperimentEnergyScaleShifts.unpack(
            theta[cr.ndim() :],
            layout_info=layout_info.shifted_experiments,
        )
        return Model(cr=cr, energy_shifts=energy_shifts)

    def plot(self, fit_data: FitData, scale: float) -> Figure:
        fig, ax = plt.subplots()

        for exp, data_by_particle in fit_data.spectra.items():
            for _, data in data_by_particle.items():
                data.with_shifted_energy_scale(f=self.energy_shifts.f(exp)).plot(scale=scale, ax=ax)

        primaries = list(self.cr.components.keys())
        self.cr.plot(
            Emin=fit_data.R_bounds[0] * min(p.Z for p in primaries),
            Emax=fit_data.R_bounds[1] * max(p.Z for p in primaries),
            scale=scale,
            axes=ax,
        )

        ax.legend(fontsize="xx-small")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(self.cr.description(), {"fontsize": "x-small"})
        return fig

    def format_params(self) -> str:
        lines = [f"{label} = {value:.2e}" for label, value in zip(self.labels(), self.pack())]
        return "\n".join(lines)


if __name__ == "__main__":
    m = Model(
        cr=GalacticCR(
            components={
                p: PowerLaw(lgI=np.random.random(), alpha=np.random.random()) for p in Primary
            },
            dampe_break=RigidityBreak(
                lg_R=np.random.random(),
                d_alpha=np.random.random(),
                lg_sharpness=np.random.random(),
            ),
        ),
        energy_shifts=ExperimentEnergyScaleShifts(
            lg_shifts={e: np.random.random() for e in Experiment}
        ),
    )
    m.validate_packing()
