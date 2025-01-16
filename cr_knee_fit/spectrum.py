from dataclasses import dataclass
from typing import ClassVar
import numpy as np

from cr_knee_fit.types_ import Packable


@dataclass
class PowerLaw(Packable[None]):
    lgI: float  # log10(I / (GeV m^2 s sr)^-1) at R0
    alpha: float  # power law index

    R0: ClassVar[float] = 1e3  # reference rigidity

    def pack(self) -> np.ndarray:
        return np.array([self.lgI, self.alpha])

    def ndim(self) -> int:
        return 2

    def labels(self, latex: bool) -> list[str]:
        if latex:
            return ["\\lg(I)", "\\alpha"]
        else:
            return ["lg(I)", "alpha"]

    def layout_info(self) -> None:
        return None

    @classmethod
    def unpack(cls, theta: np.ndarray, layout_info: None) -> "PowerLaw":
        return PowerLaw(*theta[:2])
