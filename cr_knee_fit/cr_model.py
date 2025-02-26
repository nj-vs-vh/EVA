import itertools
from dataclasses import dataclass
from typing import ClassVar, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from num2tex import num2tex  # type: ignore

from cr_knee_fit.elements import (
    Element,
    element_name_to_Z_A,
    isotope_average_A,
    unresolved_element_names,
    unresolved_element_normalized_abundances_at_1TV,
    low_energy_CR_spectra,
)
from cr_knee_fit.types_ import Packable

# region: shared power law


@dataclass
class SpectralComponentConfig:
    elements: list[Element]
    scale_contrib_to_allpart: bool


R0 = 1e3  # GV; reference rigidity


@dataclass
class SharedPowerLawSpectrum(Packable[SpectralComponentConfig]):
    """
    Power law spectrum in rigidity with a single index and per-element normalization
    """

    lgI_per_element: dict[Element, float]  # log10(I / (GV m^2 s sr)^-1) at R0
    alpha: float  # power law index

    lg_scale_contrib_to_all: None | float = None

    @classmethod
    def single_element(cls, p: Element, lgI: float, alpha: float) -> "SharedPowerLawSpectrum":
        return SharedPowerLawSpectrum(lgI_per_element={p: lgI}, alpha=alpha)

    def compute(self, R: np.ndarray, element: Element) -> np.ndarray:
        lgI = self.lgI_per_element[element]
        I = 10.0**lgI
        return I * (R / R0) ** -self.alpha

    @property
    def elements(self) -> list[Element]:
        return sorted(self.lgI_per_element.keys())

    def pack(self) -> np.ndarray:
        packed = [self.lgI_per_element[p] for p in self.elements]
        packed.append(self.alpha)
        if self.lg_scale_contrib_to_all is not None:
            packed.append(self.lg_scale_contrib_to_all)
        return np.array(packed)

    def ndim(self) -> int:
        return len(self.lgI_per_element) + 1 + int(self.lg_scale_contrib_to_all is not None)

    def labels(self, latex: bool) -> list[str]:
        ps_label = component_label(self.elements)
        if latex:
            res = [f"\\lg(I_\\text{{{p.name}}})" for p in self.elements]
            res.append(f"\\alpha_\\text{{{ps_label}}}")
            if self.lg_scale_contrib_to_all:
                res.append(f"\\lg(K_\\text{{{ps_label}}})")
            return res
        else:
            res = [f"lgI_{{{p.name}}}" for p in self.elements] + [f"alpha_{{{ps_label}}}"]
            if self.lg_scale_contrib_to_all:
                res.append(f"lg(K)_{{{ps_label}}}")
            return res

    def description(self) -> str:
        return f"$ \\text{{{component_label(self.elements)}}} \\propto R ^{{-{self.alpha:.2f}}} $"

    def layout_info(self) -> SpectralComponentConfig:
        return SpectralComponentConfig(
            elements=self.elements,
            scale_contrib_to_allpart=self.lg_scale_contrib_to_all is not None,
        )

    @classmethod
    def unpack(
        cls, theta: np.ndarray, layout_info: SpectralComponentConfig
    ) -> "SharedPowerLawSpectrum":
        lgI_per_el = dict(zip(layout_info.elements, theta))
        offset = len(lgI_per_el)
        alpha = theta[offset]
        if layout_info.scale_contrib_to_allpart:
            offset += 1
            lg_scale_contrib_to_all = theta[offset]
        else:
            lg_scale_contrib_to_all = None
        return SharedPowerLawSpectrum(
            lgI_per_element=lgI_per_el,
            alpha=alpha,
            lg_scale_contrib_to_all=lg_scale_contrib_to_all,
        )


# endregion


@dataclass
class UnresolvedElementsSpectrum(Packable[None]):
    lgI: float

    def compute(self, R: np.ndarray, element_name: str) -> np.ndarray:
        I = 10.0**self.lgI * unresolved_element_normalized_abundances_at_1TV[element_name]
        alpha = low_energy_CR_spectra[element_name][1]
        return I * (R / R0) ** -alpha

    def pack(self) -> np.ndarray:
        return np.array([self.lgI])

    def ndim(self) -> int:
        return 1

    def labels(self, latex: bool) -> list[str]:
        subscript = "\\text{Unres.}" if latex else "Unres"
        if latex:
            return [f"\\lg(I_{subscript})"]
        else:
            return [f"lgI_{subscript}"]

    def layout_info(self) -> None:
        return None

    @classmethod
    def unpack(cls, theta: np.ndarray, layout_info: None) -> "UnresolvedElementsSpectrum":
        return UnresolvedElementsSpectrum(lgI=theta[0])


# region: breaks


BreakQuantity = Literal[
    "R",  # rigidity, GV
    "E",  # total energy, GeV
    "E_n",  # energy per nucleon, GeV
]


@dataclass
class SpectralBreakConfig:
    fixed_lg_sharpness: float | None
    quantity: BreakQuantity = "R"  # backcompat for legacy configs


@dataclass
class SpectralBreak(Packable[SpectralBreakConfig]):
    lg_break: float  # break position, in units of quantity
    d_alpha: float  # PL index change at the break
    lg_sharpness: float  # 0 is very smooth, 10+ is very sharp

    fix_sharpness: bool = False
    quantity: BreakQuantity = "R"

    def ndim(self) -> int:
        return 2 if self.fix_sharpness else 3

    def pack(self) -> np.ndarray:
        return np.array([self.lg_break, self.d_alpha, self.lg_sharpness][: self.ndim()])

    def labels(self, latex: bool) -> list[str]:
        if latex:
            labels = [f"\\lg({self.quantity}^\\text{{b}})", "\\Delta \\alpha", "\\lg(s)"]
        else:
            labels = [f"lg({self.quantity}^b)", "d_alpha", "lg(s)"]
        return labels[: self.ndim()]

    def layout_info(self) -> SpectralBreakConfig:
        return SpectralBreakConfig(
            fixed_lg_sharpness=self.lg_sharpness if self.fix_sharpness else None,
            quantity=self.quantity,
        )

    @classmethod
    def unpack(cls, theta: np.ndarray, layout_info: SpectralBreakConfig) -> "SpectralBreak":
        lg_R, d_alpha = theta[:2]
        return SpectralBreak(
            lg_break=lg_R,
            d_alpha=d_alpha,
            lg_sharpness=layout_info.fixed_lg_sharpness or theta[2],
            fix_sharpness=layout_info.fixed_lg_sharpness is not None,
            quantity=layout_info.quantity,
        )

    def compute(self, R: np.ndarray, Z: int, A: float) -> np.ndarray:
        match self.quantity:
            case "E":
                quantity = R * float(Z)
            case "R":
                quantity = R
            case "E_n":
                quantity = R * (Z / A)

        break_ = 10**self.lg_break
        s = 10**self.lg_sharpness

        result = np.zeros_like(quantity)

        # calculation in different form for numerical stability
        below_break = quantity <= break_
        result[below_break] = (1 + (quantity[below_break] / break_) ** s) ** (-self.d_alpha / s)

        above_break = quantity > break_
        R_b_to_R = break_ / quantity[above_break]
        result[above_break] = R_b_to_R**self.d_alpha * (R_b_to_R**s + 1) ** (-self.d_alpha / s)
        return result

    def description(self) -> str:
        parts: list[str] = []
        unit = "GV" if self.quantity == "R" else "GeV"
        parts.append(
            f"$ {self.quantity}^\\text{{b}} = {num2tex(10**self.lg_break, precision=2, exp_format='cdot')}~\\text{{{unit}}} $"
        )
        parts.append(f"$ \\Delta \\alpha = {self.d_alpha:.2f} $")
        if not self.fix_sharpness:
            parts.append(f"$ s = {10**self.lg_sharpness:.2f} $")
        return ", ".join(parts)


# endregion


@dataclass
class PopulationMetadata:
    name: str
    linestyle: str | None


@dataclass
class CosmicRaysModelConfig:
    components: Sequence[list[Element] | SpectralComponentConfig]
    breaks: Sequence[SpectralBreakConfig]
    rescale_all_particle: bool
    add_unresolved_elements: bool = False

    population_name: str | None = None
    population_meta: PopulationMetadata | None = None

    def __post_init__(self) -> None:
        assert len(self.resolved_elements) == len(
            set(self.resolved_elements)
        ), "Duplicate elements in components!"
        if self.population_name is not None:
            if self.population_meta is not None:
                raise ValueError("population name and metadata are mutually exclusive")
            self.population_meta = PopulationMetadata(
                name=self.population_name,
                linestyle=None,
            )

    @property
    def has_free_Z_component(self) -> bool:
        return Element.FreeZ in self.resolved_elements

    @property
    def component_configs(self) -> list[SpectralComponentConfig]:
        return [
            (
                conf_or_elements
                if isinstance(conf_or_elements, SpectralComponentConfig)
                else SpectralComponentConfig(conf_or_elements, scale_contrib_to_allpart=False)
            )
            for conf_or_elements in self.components
        ]

    @property
    def resolved_elements(self) -> list[Element]:
        return sorted(
            itertools.chain.from_iterable(
                c.elements if isinstance(c, SpectralComponentConfig) else c for c in self.components
            )
        )


@dataclass
class CosmicRaysModel(Packable[CosmicRaysModelConfig]):
    base_spectra: list[SharedPowerLawSpectrum]
    breaks: list[SpectralBreak]

    all_particle_lg_shift: float | None  # sum of elements* 10^shift = all particle spectrum
    free_Z: float | None

    unresolved_elements_spectrum: UnresolvedElementsSpectrum | None

    population_meta: PopulationMetadata | None = None

    def __post_init__(self) -> None:
        seen = set[Element]()
        for s in self.base_spectra:
            if seen.intersection(s.elements):
                raise ValueError(
                    "Ambiguous base spectra, at least one element specified in several components"
                )
            seen.update(s.elements)
        if self.free_Z is not None and Element.FreeZ not in self.resolved_elements:
            raise ValueError(
                "FreeZ element must be present as a element if it's effective Z is used as a param"
            )

    @property
    def resolved_elements(self) -> list[Element]:
        return self.layout_info().resolved_elements

    @property
    def all_elements(self) -> list[Element | str]:
        res: list[Element | str] = [e for e in self.layout_info().resolved_elements]
        if self.unresolved_elements_spectrum:
            res.extend(unresolved_element_names)
        return res

    def description(self) -> str:
        return "; ".join(
            itertools.chain(
                (spectrum.description() for spectrum in self.base_spectra),
                (break_.description() for break_ in self.breaks),
            )
        )

    def _get_component(self, element: Element) -> SharedPowerLawSpectrum | None:
        matches = [comp for comp in self.base_spectra if element in comp.elements]
        if not matches:
            return None
        if len(matches) > 1:
            raise RuntimeError(f"Element {element} matches more than one component: {matches}")
        return matches[0]

    def compute_rigidity(self, R: np.ndarray, element: Element | str) -> np.ndarray:
        if isinstance(element, str):  # unresolved element
            if self.unresolved_elements_spectrum is not None:
                flux = self.unresolved_elements_spectrum.compute(R, element_name=element)
            else:
                return np.zeros_like(R)
        else:
            spectrum = self._get_component(element)
            if spectrum is None:
                return np.zeros_like(R)
            flux = spectrum.compute(R, element)

        Z = round(self.element_Z(element))
        for break_ in self.breaks:
            flux *= break_.compute(R, Z=Z, A=isotope_average_A(Z))
        return flux

    def element_Z(self, element: Element | str) -> float:
        if element is Element.FreeZ:
            if self.free_Z is None:
                raise ValueError(
                    f"Attempted to get FreeZ element but it's Z is not included in the model"
                )
            return self.free_Z
        elif isinstance(element, str):
            return element_name_to_Z_A[element][0]
        else:
            return element.Z

    def element_name(self, element: Element | str) -> str:
        if element is Element.FreeZ:
            if self.free_Z is None:
                raise ValueError(
                    f"Attempted to get FreeZ element but it's Z is not included in the model"
                )
            return f"Z = {int(self.free_Z)}"
        elif isinstance(element, str):
            return element
        else:
            return element.name

    def compute(
        self,
        E: np.ndarray,
        element: Element | str,
        contrib_to_all_particle: bool = False,
    ) -> np.ndarray:
        Z = self.element_Z(element)
        R = E / Z
        dNdR = self.compute_rigidity(R, element=element)
        dNdE = dNdR / Z
        if contrib_to_all_particle and isinstance(element, Element):
            component = self._get_component(element)
            if component is not None and component.lg_scale_contrib_to_all:
                dNdE *= 10 ** (component.lg_scale_contrib_to_all)
        return dNdE

    def compute_lnA(self, E: np.ndarray) -> np.ndarray:
        elements = self.all_elements
        spectra = np.vstack([self.compute(E, el, contrib_to_all_particle=True) for el in elements])
        lnA = np.array([np.log(isotope_average_A(round(self.element_Z(el)))) for el in elements])
        return np.sum(spectra * np.expand_dims(lnA, axis=1), axis=0) / np.sum(spectra, axis=0)

    def compute_all_particle(self, E: np.ndarray) -> np.ndarray:
        flux = np.zeros_like(E)
        for element in self.all_elements:
            flux += self.compute(E, element=element, contrib_to_all_particle=True)
        if self.all_particle_lg_shift is not None:
            flux *= 10**self.all_particle_lg_shift
        return flux

    def compute_abundances(self, R: float) -> dict[Element | str, float]:
        return {
            element: float(self.compute_rigidity(np.array([R]), element=element)[0])
            for element in self.all_elements
        }

    @property
    def _linestyle(self) -> str | None:
        if self.population_meta is None:
            return None
        else:
            return self.population_meta.linestyle

    def compute_extra_all_particle_contribution(self, E: np.ndarray) -> np.ndarray:
        return self.compute_all_particle(E) - sum(
            (self.compute(E, element=element) for element in self.all_elements),
            np.zeros_like(E),
        )

    def plot(
        self,
        Emin: float,
        Emax: float,
        scale: float,
        axes: Axes | None = None,
        all_particle: bool = False,
        elements: list[Element] | None = None,
    ) -> Axes:
        if axes is not None:
            ax = axes
        else:
            _, ax = plt.subplots()

        E_grid = np.logspace(np.log10(Emin), np.log10(Emax), 100)
        E_factor = E_grid**scale
        label_prefix = self.population_prefix(latex=False)

        for element in elements or self.resolved_elements:
            ax.plot(
                E_grid,
                E_factor * self.compute(E_grid, element),
                label=label_prefix + self.element_name(element),
                color=element.color,
                linestyle=self._linestyle,
            )

        if self.unresolved_elements_spectrum:
            ax.plot(
                E_grid,
                (
                    E_factor
                    * sum(
                        (
                            self.compute(E_grid, element=unres_el)
                            for unres_el in unresolved_element_names
                        ),
                        start=np.zeros_like(E_grid),
                    )
                ),
                label=label_prefix + "Unresolved elements",
                color="magenta",
                linestyle=self._linestyle,
            )

        extra_all_particle_contrib = self.compute_extra_all_particle_contribution(E_grid)
        has_extra_allparticle_contrib = np.any(extra_all_particle_contrib > 0)
        if has_extra_allparticle_contrib:
            ax.plot(
                E_grid,
                E_factor * extra_all_particle_contrib,
                label=label_prefix + "Extra contribution to all-particle",
                color="gray",
                linestyle=self._linestyle,
            )
        if all_particle or has_extra_allparticle_contrib:
            ax.plot(
                E_grid,
                E_factor * self.compute_all_particle(E_grid),
                label=label_prefix + "All particle",
                color="black",
                linestyle=self._linestyle,
            )
        return ax

    def ndim(self) -> int:
        return (
            sum(c.ndim() for c in self.base_spectra)
            + sum(b.ndim() for b in self.breaks)
            + (
                self.unresolved_elements_spectrum.ndim()
                if self.unresolved_elements_spectrum is not None
                else 0
            )
            + int(self.all_particle_lg_shift is not None)
            + int(self.free_Z is not None)
        )

    def population_prefix(self, latex: bool) -> str:
        if self.population_meta is not None:
            return (
                f"\\text{{{self.population_meta.name}}}\\;"
                if latex
                else self.population_meta.name + " "
            )
        else:
            return ""

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
        if self.unresolved_elements_spectrum is not None:
            labels.extend(self.unresolved_elements_spectrum.labels(latex))
        if self.all_particle_lg_shift is not None:
            labels.append("\\lg(K)" if latex else "lgK")
        if self.free_Z is not None:
            labels.append("Z_\\text{Unobs. eff}" if latex else "Z_Unobs")

        prefix = self.population_prefix(latex)
        labels = [prefix + label for label in labels]

        return labels

    def pack(self) -> np.ndarray:
        subvectors = [spectrum.pack() for spectrum in self.base_spectra] + [
            b.pack() for b in self.breaks
        ]
        if self.unresolved_elements_spectrum is not None:
            subvectors.append(self.unresolved_elements_spectrum.pack())
        if self.all_particle_lg_shift is not None:
            subvectors.append(np.array([self.all_particle_lg_shift]))
        if self.free_Z:
            subvectors.append(np.array([self.free_Z]))
        return np.hstack(subvectors)

    def layout_info(self) -> CosmicRaysModelConfig:
        return CosmicRaysModelConfig(
            components=[spectrum.layout_info() for spectrum in self.base_spectra],
            breaks=[b.layout_info() for b in self.breaks],
            rescale_all_particle=self.all_particle_lg_shift is not None,
            add_unresolved_elements=self.unresolved_elements_spectrum is not None,
            population_meta=self.population_meta,
        )

    @classmethod
    def unpack(cls, theta: np.ndarray, layout_info: CosmicRaysModelConfig) -> "CosmicRaysModel":
        components: list[SharedPowerLawSpectrum] = []
        offset = 0
        for component in layout_info.components:
            component_config = (
                component
                if isinstance(component, SpectralComponentConfig)
                else SpectralComponentConfig(
                    elements=component,
                    scale_contrib_to_allpart=False,  # backwards compatibility
                )
            )
            spectrum = SharedPowerLawSpectrum.unpack(theta[offset:], component_config)
            components.append(spectrum)
            offset += spectrum.ndim()
        breaks: list[SpectralBreak] = []
        for break_config in layout_info.breaks:
            b = SpectralBreak.unpack(theta[offset:], break_config)
            breaks.append(b)
            offset += b.ndim()

        unresolved_elements_spectrum: UnresolvedElementsSpectrum | None = None
        if layout_info.add_unresolved_elements:
            unresolved_elements_spectrum = UnresolvedElementsSpectrum.unpack(theta[offset:], None)
            offset += unresolved_elements_spectrum.ndim()

        if layout_info.rescale_all_particle:
            all_particle_lg_shift = theta[offset]
            offset += 1
        else:
            all_particle_lg_shift = None

        if layout_info.has_free_Z_component:
            free_Z = theta[offset]
            offset += 1
        else:
            free_Z = None

        return CosmicRaysModel(
            base_spectra=components,
            breaks=breaks,
            unresolved_elements_spectrum=unresolved_elements_spectrum,
            all_particle_lg_shift=all_particle_lg_shift,
            free_Z=free_Z,
            population_meta=layout_info.population_meta,
        )


def component_label(p_or_ps: Element | Sequence[Element]) -> str:
    return p_or_ps.name if isinstance(p_or_ps, Element) else ", ".join(p.name for p in p_or_ps)


if __name__ == "__main__":
    gcr = CosmicRaysModel(
        base_spectra=[
            SharedPowerLawSpectrum.single_element(
                Element.H, np.random.random(), np.random.random()
            ),
            SharedPowerLawSpectrum.single_element(
                Element.He, np.random.random(), np.random.random()
            ),
            SharedPowerLawSpectrum(
                {
                    Element.Mg: np.random.random(),
                    Element.Fe: np.random.random(),
                    Element.FreeZ: np.random.random(),
                },
                np.random.random(),
            ),
        ],
        breaks=[
            SpectralBreak(lg_break=5.0, d_alpha=-0.4, lg_sharpness=0.5),
            SpectralBreak(lg_break=5.0, d_alpha=-0.4, lg_sharpness=0.5, fix_sharpness=True),
            SpectralBreak(lg_break=5.0, d_alpha=-0.4, lg_sharpness=0.5),
        ],
        unresolved_elements_spectrum=UnresolvedElementsSpectrum(
            lgI=np.random.random(),
        ),
        all_particle_lg_shift=np.random.random(),
        free_Z=np.random.random(),
        population_meta=None,
    )
    gcr.validate_packing()
