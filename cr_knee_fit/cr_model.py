from dataclasses import dataclass
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from num2tex import num2tex

from cr_knee_fit.plotting import label_energy_flux
from cr_knee_fit.types_ import Packable, Primary


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


@dataclass
class RigidityBreak(Packable[None]):
    lg_R: float  # break rigidity, lg(R / GV)
    d_alpha: float  # PL index change at the break
    lg_sharpness: float  # 0 is very smooth, 10+ is very sharp

    def ndim(self) -> int:
        return 3

    def pack(self) -> np.ndarray:
        return np.array([self.lg_R, self.d_alpha, self.lg_sharpness])

    def labels(self, latex: bool) -> list[str]:
        if latex:
            return ["\\lg(R_b)", "\\Delta \\alpha", "\\lg(s)"]
        else:
            return ["lg(R_b)", "d_alpha", "lg(s)"]

    def layout_info(self) -> None:
        return None

    @classmethod
    def unpack(cls, theta: np.ndarray, layout_info: None) -> "RigidityBreak":
        return RigidityBreak(*theta[:3])

    def compute(self, R: np.ndarray) -> np.ndarray:
        R_b = 10**self.lg_R
        s = 10**self.lg_sharpness

        result = np.zeros_like(R)

        # calculation in different form for numerical stability
        below_break = R <= R_b
        result[below_break] = (1 + (R[below_break] / R_b) ** s) ** (-self.d_alpha / s)

        above_break = R > R_b
        R_b_to_R = R_b / R[above_break]
        result[above_break] = R_b_to_R**self.d_alpha * (R_b_to_R**s + 1) ** (-self.d_alpha / s)
        return result

    def description(self) -> str:
        parts: list[str] = []
        parts.append(
            f"$ R_b = {num2tex(10**self.lg_R, precision=2, exp_format='cdot')}~\\text{{GV}} $"
        )
        parts.append(f"$ \\Delta \\alpha = {self.d_alpha:.2f} $")
        parts.append(f"$ s = {10**self.lg_sharpness:.2f} $")
        return ", ".join(parts)


@dataclass
class CosmicRaysModelConfig:
    primaries: list[Primary]
    n_breaks: int


@dataclass
class CosmicRaysModel(Packable[CosmicRaysModelConfig]):
    components: dict[Primary, PowerLaw]
    breaks: list[RigidityBreak]

    def description(self) -> str:
        component_strs = [
            f"$ \\text{{{p.name}}} \\propto R ^{{-{pl.alpha:.2f}}} $"
            for p, pl in self.components.items()
        ]
        for break_ in self.breaks:
            component_strs.append(break_.description())
        return "; ".join(component_strs)

    def compute(self, E: np.ndarray, particle: Primary) -> np.ndarray:
        pl = self.components.get(particle)
        if pl is None:
            raise ValueError(
                f"Unsupported particle: {particle}. Supported values are {list(self.components.keys())}."
            )
        R = E / float(particle.Z)

        I = 10**pl.lgI
        flux = I * (R / pl.R0) ** -pl.alpha
        for rb in self.breaks:
            flux *= rb.compute(R)
        return flux

    def plot(self, Emin: float, Emax: float, scale: float, axes: Axes | None = None) -> Axes:
        if axes is not None:
            ax = axes
        else:
            _, ax = plt.subplots()
        E_grid = np.logspace(np.log10(Emin), np.log10(Emax), 300)
        E_factor = E_grid**scale
        for p in self.components.keys():
            ax.loglog(E_grid, E_factor * self.compute(E_grid, p), label=p.name, color=p.color)
        ax.legend()
        label_energy_flux(ax, scale)
        return ax

    @property
    def primaries(self) -> list[Primary]:
        return sorted(self.components.keys())

    def ndim(self) -> int:
        return sum(c.ndim() for c in self.components.values()) + sum(b.ndim() for b in self.breaks)

    def pack(self) -> np.ndarray:
        subvectors = [self.components[p].pack() for p in self.primaries] + [
            b.pack() for b in self.breaks
        ]
        return np.hstack(subvectors)

    def labels(self, latex: bool) -> list[str]:
        labels: list[str] = []
        for p in self.primaries:
            for param_label in self.components[p].labels(latex):
                labels.append(
                    f"{param_label}_\\text{{{p.name}}}" if latex else f"{param_label} ({p.name})"
                )
        for i, b in enumerate(self.breaks):
            break_idx = i + 1
            for param_label in b.labels(latex):
                labels.append(
                    f"{param_label}_{{{break_idx}}}" if latex else f"{param_label}_{break_idx}"
                )
        return labels

    def layout_info(self) -> CosmicRaysModelConfig:
        return CosmicRaysModelConfig(
            primaries=self.primaries,
            n_breaks=len(self.breaks),
        )

    @classmethod
    def unpack(cls, theta: np.ndarray, layout_info: CosmicRaysModelConfig) -> "CosmicRaysModel":
        components: dict[Primary, PowerLaw] = dict()
        offset = 0
        for p in layout_info.primaries:
            component = PowerLaw.unpack(theta[offset:], None)
            components[p] = component
            offset += component.ndim()
        breaks: list[RigidityBreak] = []
        for _ in range(layout_info.n_breaks):
            b = RigidityBreak.unpack(theta[offset:], None)
            breaks.append(b)
            offset += b.ndim()
        return CosmicRaysModel(
            components=components,
            breaks=breaks,
        )


if __name__ == "__main__":
    gcr = CosmicRaysModel(
        components={
            Primary.H: PowerLaw(3, 7),
            Primary.Fe: PowerLaw(1, 1),
            Primary.C: PowerLaw(5.5, 0.5),
        },
        breaks=[
            RigidityBreak(lg_R=5.0, d_alpha=-0.4, lg_sharpness=0.5),
            RigidityBreak(lg_R=5.0, d_alpha=-0.4, lg_sharpness=0.5),
            RigidityBreak(lg_R=5.0, d_alpha=-0.4, lg_sharpness=0.5),
        ],
    )
    gcr.validate_packing()
