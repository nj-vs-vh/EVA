from dataclasses import dataclass
import numpy as np

from cr_knee_fit.types_ import Experiment, Packable


@dataclass
class ExperimentEnergyScaleShifts(Packable[list[Experiment]]):
    lg_shifts: dict[Experiment, float]

    def f(self, e: Experiment) -> float:
        lg_f = self.lg_shifts.get(e)
        if lg_f is None:
            return 1.0
        else:
            return 10**lg_f

    @property
    def experiments(self) -> list[Experiment]:
        return sorted(self.lg_shifts.keys())

    def ndim(self) -> int:
        return len(self.experiments)

    def pack(self) -> np.ndarray:
        return np.array([self.lg_shifts[exp] for exp in self.experiments])

    def labels(self, latex: bool = False) -> list[str]:
        if latex:
            return [f"\\lg(f_\\text{{{exp.name}}})" for exp in self.experiments]
        else:
            return [f"lg(f_{exp.name})" for exp in self.experiments]

    def layout_info(self) -> list[Experiment]:
        return self.experiments

    @classmethod
    def unpack(
        cls, theta: np.ndarray, layout_info: list[Experiment]
    ) -> "ExperimentEnergyScaleShifts":
        experiments = layout_info
        return ExperimentEnergyScaleShifts(
            lg_shifts={exp: shift for exp, shift in zip(experiments, theta)}
        )


if __name__ == "__main__":
    s = ExperimentEnergyScaleShifts({e: np.random.random() for e in Experiment})
    s.validate_packing()
