from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from num2tex import num2tex
import numpy as np

from cr_knee_fit.plotting import label_energy_flux
from cr_knee_fit.spectrum import PowerLaw
from cr_knee_fit.types_ import Packable, Primary


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


@dataclass
class GalacticCR(Packable[list[Primary]]):
    components: dict[Primary, PowerLaw]
    dampe_break: RigidityBreak

    def description(self) -> str:
        component_strs = [
            f"$ \\text{{{p.name}}} \\propto R ^{{-{pl.alpha:.2f}}} $"
            for p, pl in self.components.items()
        ]
        component_strs.append(
            f"$ R_b = {num2tex(10**self.dampe_break.lg_R, precision=3, exp_format='cdot')}~\\text{{GV}} $"
        )
        component_strs.append(f"$ \\Delta \\alpha = {self.dampe_break.d_alpha:.2f} $")
        component_strs.append(f"$ s = {10**self.dampe_break.lg_sharpness:.2f} $")
        return ", ".join(component_strs)

    def compute(self, E: np.ndarray, particle: Primary) -> np.ndarray:
        pl = self.components.get(particle)
        if pl is None:
            raise ValueError(
                f"Unsupported particle: {particle}. Supported values are {list(self.components.keys())}."
            )
        R = E / float(particle.Z)

        I = 10**pl.lgI
        y = I * (R / pl.R0) ** -pl.alpha

        R_break = 10**self.dampe_break.lg_R
        s = 10**self.dampe_break.lg_sharpness
        y /= (1.0 + (R / R_break) ** s) ** (self.dampe_break.d_alpha / s)
        return y

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
        return sum(c.ndim() for c in self.components.values()) + self.dampe_break.ndim()

    def pack(self) -> np.ndarray:
        subvectors = [self.components[p].pack() for p in self.primaries]
        subvectors.append(self.dampe_break.pack())
        return np.hstack(subvectors)

    def labels(self, latex: bool) -> list[str]:
        labels: list[str] = []
        for p in self.primaries:
            for param_label in self.components[p].labels(latex):
                labels.append(
                    f"{param_label}_\\text{{{p.name}}}" if latex else f"{param_label} ({p.name})"
                )
        labels.extend(self.dampe_break.labels(latex))
        return labels

    def layout_info(self) -> list[Primary]:
        return self.primaries

    @classmethod
    def unpack(cls, theta: np.ndarray, layout_info: list[Primary]) -> "GalacticCR":
        components: dict[Primary, PowerLaw] = dict()
        offset = 0
        for particle in layout_info:
            component = PowerLaw.unpack(theta[offset:], None)
            components[particle] = component
            offset += component.ndim()
        dampe_break = RigidityBreak.unpack(theta[offset:], None)
        return GalacticCR(components=components, dampe_break=dampe_break)


if __name__ == "__main__":
    gcr = GalacticCR(
        components={
            Primary.H: PowerLaw(3, 7),
            Primary.Fe: PowerLaw(1, 1),
            Primary.C: PowerLaw(5.5, 0.5),
        },
        dampe_break=RigidityBreak(lg_R=5.0, d_alpha=-0.4, lg_sharpness=0.5),
    )
    gcr.validate_packing()
