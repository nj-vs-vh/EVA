import itertools
from dataclasses import dataclass
from typing import ClassVar, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from num2tex import num2tex  # type: ignore

from cr_knee_fit.types_ import Packable, Primary, most_abundant_stable_izotope_A
from cr_knee_fit.utils import label_energy_flux


@dataclass
class SharedPowerLaw(Packable[list[Primary]]):
    """
    Power law spectrum in rigidity with a single index and per-primary normalization
    """

    lgI_per_primary: dict[Primary, float]  # log10(I / (GV m^2 s sr)^-1) at R0
    alpha: float  # power law index

    R0: ClassVar[float] = 1e3  # reference rigidity

    @classmethod
    def single_primary(cls, p: Primary, lgI: float, alpha: float) -> "SharedPowerLaw":
        return SharedPowerLaw(lgI_per_primary={p: lgI}, alpha=alpha)

    def compute(self, R: np.ndarray, primary: Primary) -> np.ndarray:
        lgI = self.lgI_per_primary[primary]
        I = 10**lgI
        return I * (R / self.R0) ** -self.alpha

    @property
    def primaries(self) -> list[Primary]:
        return sorted(self.lgI_per_primary.keys())

    def pack(self) -> np.ndarray:
        return np.array([self.lgI_per_primary[p] for p in self.primaries] + [self.alpha])

    def ndim(self) -> int:
        return len(self.lgI_per_primary) + 1

    def labels(self, latex: bool) -> list[str]:
        if latex:
            return [f"\\lg(I)_\\text{{{p.name}}}" for p in self.primaries] + [
                f"\\alpha_\\text{{{component_label(self.primaries)}}}"
            ]
        else:
            return [f"lgI_{{{p.name}}}" for p in self.primaries] + [
                f"alpha_{{{component_label(self.primaries)}}}"
            ]

    def description(self) -> str:
        return f"$ \\text{{{component_label(self.primaries)}}} \\propto R ^{{-{self.alpha:.2f}}} $"

    def layout_info(self) -> list[Primary]:
        return self.primaries

    @classmethod
    def unpack(cls, theta: np.ndarray, layout_info: list[Primary]) -> "SharedPowerLaw":
        lgI_per_primary = dict(zip(layout_info, theta))
        return SharedPowerLaw(lgI_per_primary=lgI_per_primary, alpha=theta[len(lgI_per_primary)])


@dataclass
class RigidityBreakConfig:
    fixed_lg_sharpness: float | None


@dataclass
class RigidityBreak(Packable[RigidityBreakConfig]):
    lg_R: float  # break rigidity, lg(R / GV)
    d_alpha: float  # PL index change at the break
    lg_sharpness: float  # 0 is very smooth, 10+ is very sharp

    fix_sharpness: bool = False

    def ndim(self) -> int:
        return 2 if self.fix_sharpness else 3

    def pack(self) -> np.ndarray:
        return np.array([self.lg_R, self.d_alpha, self.lg_sharpness][: self.ndim()])

    def labels(self, latex: bool) -> list[str]:
        if latex:
            labels = ["\\lg(R_b)", "\\Delta \\alpha", "\\lg(s)"]
        else:
            labels = ["lg(R_b)", "d_alpha", "lg(s)"]
        return labels[: self.ndim()]

    def layout_info(self) -> RigidityBreakConfig:
        return RigidityBreakConfig(
            fixed_lg_sharpness=self.lg_sharpness if self.fix_sharpness else None
        )

    @classmethod
    def unpack(cls, theta: np.ndarray, layout_info: RigidityBreakConfig) -> "RigidityBreak":
        lg_R, d_alpha = theta[:2]
        return RigidityBreak(
            lg_R=lg_R,
            d_alpha=d_alpha,
            lg_sharpness=layout_info.fixed_lg_sharpness or theta[2],
            fix_sharpness=layout_info.fixed_lg_sharpness is not None,
        )

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
    components: Sequence[list[Primary]]
    breaks: Sequence[RigidityBreakConfig]
    rescale_all_particle: bool

    def __post_init__(self) -> None:
        assert len(self.primaries) == len(set(self.primaries))

    @property
    def add_unobserved_component(self) -> bool:
        return Primary.Unobserved in self.primaries

    @property
    def primaries(self) -> list[Primary]:
        return sorted(itertools.chain.from_iterable(self.components))


@dataclass
class CosmicRaysModel(Packable[CosmicRaysModelConfig]):
    base_spectra: list[SharedPowerLaw]
    breaks: list[RigidityBreak]

    all_particle_lg_shift: float | None  # sum of primaries * 10^shift = all particle spectrum
    unobserved_component_effective_Z: float | None

    def __post_init__(self) -> None:
        seen_primaries = set[Primary]()
        for s in self.base_spectra:
            if seen_primaries.intersection(s.primaries):
                raise ValueError(
                    "Ambiguous base spectra, at least one primary specified in several components"
                )
            seen_primaries.update(s.primaries)
        if (
            self.unobserved_component_effective_Z is not None
            and Primary.Unobserved not in self.layout_info().primaries
        ):
            raise ValueError(
                "Unobserved primary must be present as a primary if it's effective Z is used as a param"
            )

    def description(self) -> str:
        return "; ".join(
            itertools.chain(
                (spectrum.description() for spectrum in self.base_spectra),
                (break_.description() for break_ in self.breaks),
            )
        )

    def compute_rigidity(self, R: np.ndarray, primary: Primary) -> np.ndarray:
        matches = [pl for pl in self.base_spectra if primary in pl.primaries]
        if not matches:
            raise ValueError(
                f"Unsupported primary: {primary}, this model includes: {self.layout_info().primaries}"
            )
        spectrum = matches[0]
        flux = spectrum.compute(R, primary)
        for rb in self.breaks:
            flux *= rb.compute(R)
        return flux

    def _primary_Z(self, primary: Primary) -> float:
        if primary is Primary.Unobserved:
            if self.unobserved_component_effective_Z is None:
                raise ValueError(
                    f"Attempted to get unobserved primary Z but it's not included in the model"
                )
            return self.unobserved_component_effective_Z
        else:
            return primary.Z

    def compute(self, E: np.ndarray, primary: Primary) -> np.ndarray:
        Z = self._primary_Z(primary)
        R = E / Z
        dNdR = self.compute_rigidity(R, primary=primary)
        return dNdR / Z

    def compute_lnA(self, E: np.ndarray) -> np.ndarray:
        spectra = np.vstack([self.compute(E, primary) for primary in Primary])
        lnA = np.array([most_abundant_stable_izotope_A(round(self._primary_Z(p))) for p in Primary])
        return np.sum(spectra * np.expand_dims(lnA, axis=1), axis=0) / np.sum(spectra, axis=0)

    def compute_all_particle(self, E: np.ndarray) -> np.ndarray:
        flux = np.zeros_like(E)
        for primary in self.layout_info().primaries:
            flux += self.compute(E, primary=primary)
        if self.all_particle_lg_shift is not None:
            flux *= 10**self.all_particle_lg_shift
        return flux

    def plot(self, Emin: float, Emax: float, scale: float, axes: Axes | None = None) -> Axes:
        if axes is not None:
            ax = axes
        else:
            _, ax = plt.subplots()
        E_grid = np.logspace(np.log10(Emin), np.log10(Emax), 100)
        E_factor = E_grid**scale
        for p in self.layout_info().primaries:
            ax.plot(
                E_grid,
                E_factor * self.compute(E_grid, p),
                label=p.name,
                color=p.color,
            )
        if self.all_particle_lg_shift:
            ax.plot(
                E_grid,
                E_factor * self.compute_all_particle(E_grid),
                label="All particle",
                color="black",
            )
        return ax

    def ndim(self) -> int:
        return (
            sum(c.ndim() for c in self.base_spectra)
            + sum(b.ndim() for b in self.breaks)
            + int(self.all_particle_lg_shift is not None)
            + int(self.unobserved_component_effective_Z is not None)
        )

    def pack(self) -> np.ndarray:
        subvectors = [spectrum.pack() for spectrum in self.base_spectra] + [
            b.pack() for b in self.breaks
        ]
        if self.all_particle_lg_shift is not None:
            subvectors.append(np.array([self.all_particle_lg_shift]))
        if self.unobserved_component_effective_Z:
            subvectors.append(np.array([self.unobserved_component_effective_Z]))
        return np.hstack(subvectors)

    def labels(self, latex: bool) -> list[str]:
        labels: list[str] = []
        for spectrum in self.base_spectra:
            labels.extend(spectrum.labels(latex))
        for i, b in enumerate(self.breaks):
            break_idx = i + 1
            for param_label in b.labels(latex):
                labels.append(
                    f"{param_label}_{{{break_idx}}}" if latex else f"{param_label}_{break_idx}"
                )
        if self.all_particle_lg_shift is not None:
            labels.append("\\lg(K)" if latex else "lgK")
        if self.unobserved_component_effective_Z is not None:
            labels.append("Z_\\text{Unobs. eff}" if latex else "Z_Unobs")
        return labels

    def layout_info(self) -> CosmicRaysModelConfig:
        return CosmicRaysModelConfig(
            components=[spectrum.primaries for spectrum in self.base_spectra],
            breaks=[b.layout_info() for b in self.breaks],
            rescale_all_particle=self.all_particle_lg_shift is not None,
        )

    @classmethod
    def unpack(cls, theta: np.ndarray, layout_info: CosmicRaysModelConfig) -> "CosmicRaysModel":
        components: list[SharedPowerLaw] = []
        offset = 0
        for component in layout_info.components:
            spectrum = SharedPowerLaw.unpack(theta[offset:], component)
            components.append(spectrum)
            offset += spectrum.ndim()
        breaks: list[RigidityBreak] = []
        for break_config in layout_info.breaks:
            b = RigidityBreak.unpack(theta[offset:], break_config)
            breaks.append(b)
            offset += b.ndim()

        if layout_info.rescale_all_particle:
            all_particle_lg_shift = theta[offset]
            offset += 1
        else:
            all_particle_lg_shift = None

        if layout_info.add_unobserved_component:
            unobserved_component_eff_Z = theta[offset]
            offset += 1
        else:
            unobserved_component_eff_Z = None

        return CosmicRaysModel(
            base_spectra=components,
            breaks=breaks,
            all_particle_lg_shift=all_particle_lg_shift,
            unobserved_component_effective_Z=unobserved_component_eff_Z,
        )


def component_label(p_or_ps: Primary | Sequence[Primary]) -> str:
    return p_or_ps.name if isinstance(p_or_ps, Primary) else ", ".join(p.name for p in p_or_ps)


if __name__ == "__main__":
    gcr = CosmicRaysModel(
        base_spectra=[
            SharedPowerLaw.single_primary(Primary.H, np.random.random(), np.random.random()),
            SharedPowerLaw.single_primary(Primary.He, np.random.random(), np.random.random()),
            SharedPowerLaw(
                {
                    Primary.Mg: np.random.random(),
                    Primary.Fe: np.random.random(),
                    Primary.Unobserved: np.random.random(),
                },
                np.random.random(),
            ),
        ],
        breaks=[
            RigidityBreak(lg_R=5.0, d_alpha=-0.4, lg_sharpness=0.5),
            RigidityBreak(lg_R=5.0, d_alpha=-0.4, lg_sharpness=0.5, fix_sharpness=True),
            RigidityBreak(lg_R=5.0, d_alpha=-0.4, lg_sharpness=0.5),
        ],
        all_particle_lg_shift=np.random.random(),
        unobserved_component_effective_Z=np.random.random(),
    )
    gcr.validate_packing()
