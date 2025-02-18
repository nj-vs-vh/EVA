from dataclasses import dataclass

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
from cr_knee_fit.types_ import Packable, Primary
from cr_knee_fit.utils import legend_with_added_items


@dataclass
class ModelConfig:
    cr_model_config: CosmicRaysModelConfig
    shifted_experiments: list[Experiment]


@dataclass
class Model(Packable[ModelConfig]):
    cr_model: CosmicRaysModel
    energy_shifts: ExperimentEnergyScaleShifts

    def pack(self) -> np.ndarray:
        return np.hstack((self.cr_model.pack(), self.energy_shifts.pack()))

    def labels(self, latex: bool) -> list[str]:
        return self.cr_model.labels(latex) + self.energy_shifts.labels(latex)

    def layout_info(self) -> ModelConfig:
        return ModelConfig(
            cr_model_config=self.cr_model.layout_info(),
            shifted_experiments=self.energy_shifts.experiments,
        )

    @classmethod
    def unpack(cls, theta: np.ndarray, layout_info: ModelConfig) -> "Model":
        cr = CosmicRaysModel.unpack(theta, layout_info=layout_info.cr_model_config)
        energy_shifts = ExperimentEnergyScaleShifts.unpack(
            theta[cr.ndim() :],
            layout_info=layout_info.shifted_experiments,
        )
        return Model(cr_model=cr, energy_shifts=energy_shifts)

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

        self.cr_model.plot(
            Emin=fit_data.E_min(),
            Emax=fit_data.E_max(),
            scale=scale,
            axes=ax,
            all_particle=len(fit_data.all_particle_spectra) > 0,
        )
        legend_with_added_items(
            ax,
            [(e.legend_artist(), e.name) for e in fit_data.all_experiments()],
            fontsize="x-small",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        return fig


if __name__ == "__main__":
    m = Model(
        cr_model=CosmicRaysModel(
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
        ),
        energy_shifts=ExperimentEnergyScaleShifts(
            lg_shifts={e: np.random.random() for e in [Experiment("a"), Experiment("b")]}
        ),
    )
    m.validate_packing()
