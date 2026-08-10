import functools
import itertools
import json
from collections.abc import Collection
from dataclasses import dataclass
from typing import Annotated, Any

import matplotlib.pyplot as plt
import numpy as np
import pydantic
from crams import (
    ELEMENT_NAMES as CRAMS_ELEMENT_NAMES,
)
from crams import (
    CramsRunner,
    InjectionParams,
    PropagationParams,
)
from matplotlib.axes import Axes

from cr_knee_fit.cr_model import PopulationMetadata
from cr_knee_fit.elements import (
    Element,
)
from cr_knee_fit.types_ import Packable

CRAMS_ELEMENTS = [Element[name] for name in CRAMS_ELEMENT_NAMES]


def pack_injection(ip: InjectionParams) -> np.ndarray:
    return np.array(list(itertools.chain(ip.abundances, ip.slopes)))


def unpack_injection(v: np.ndarray) -> InjectionParams:
    return InjectionParams(
        abundances=v[: len(CRAMS_ELEMENTS)],  # type: ignore
        slopes=v[len(CRAMS_ELEMENTS) :],  # type: ignore
    )


def pack_propagation(pp: PropagationParams) -> np.ndarray:
    return np.array(
        [
            pp.H_kpc,
            pp.v_A_km_sec,
            np.log10(pp.R_b_GV),
            pp.delta,
            pp.ddelta,
            np.log10(pp.D_0_cm2_sec),
            pp.X_src,
            pp.phi,
        ]
    )


PROPAGATION_BOUNDS = [
    (0.1, 100),
    (1e-4, 1e4),
    (np.log10(100), np.log10(1000)),
    (0.0, 1.5),
    (0.0, 1.0),
    (np.log10(1e26), np.log10(1e30)),
    None,
    (0.1, 1.0),
]


def unpack_propagation(v: np.ndarray) -> PropagationParams:
    return PropagationParams(
        H_kpc=v[0],
        v_A_km_sec=v[1],
        R_b_GV=10 ** v[2],
        delta=v[3],
        ddelta=v[4],
        D_0_cm2_sec=10 ** v[5],
        X_src=v[6],
        phi=v[7],
    )


ABUNDANCE_BOUND = (0.0, 10.0)
SLOPE_BOUND = (4.01, 4.99)  # (4, 5) is strictly enforced by CRAMS, here we add a small margin


def _serialize_crams(cr: CramsRunner) -> str:
    assert isinstance(cr, CramsRunner)
    return json.dumps(
        {
            "inelastic_model": cr._inelastic_model,
            "fragmentation_model": cr._fragmentation_model,
            "verbose": cr._verbose,
            "file_output": cr._file_output,
        }
    )


def _validate_crams(input: Any) -> CramsRunner:
    if isinstance(input, CramsRunner):
        return input
    raw = json.loads(input)
    return CramsRunner(
        inelastic_model=raw["inelastic_model"],
        fragmentation_model=raw["fragmentation_model"],
        verbose=raw["verbose"],
        file_output=raw["file_output"],
    )


# copied from CRAMS source code
DEFAULT_ABUNDANCES = {
    Element.H: 5.06605e-02,
    Element.He: 2.54369e-02,
    Element.Li: 0.0,
    Element.Be: 0.0,
    Element.B: 0.0,
    Element.C: 3.98879e-03,
    Element.N: 3.36117e-04,
    Element.O: 7.15129e-03,
    Element.F: 0.0,
    Element.Ne: 1.34031e-03,
    Element.Na: 0.5e-4,
    Element.Mg: 2.38948e-03,
    Element.Al: 2.7e-4,
    Element.Si: 2.77911e-03,
    Element.P: 1e-4,
    Element.S: 4.87000e-04,
    Element.Cl: 0.0,
    Element.Ar: 3e-4,
    Element.K: 0.0,
    Element.Ca: 4e-4,
    Element.Sc: 0.0,
    Element.Ti: 0.0,
    Element.V: 0.0,
    Element.Cr: 2.5e-4,
    Element.Mn: 0.0,
    Element.Fe: 6.80000e-03,
    Element.Co: 0.0,
    Element.Ni: 4e-4,
}


class CramsModelConfig(pydantic.BaseModel):
    crams: Annotated[
        CramsRunner,
        pydantic.PlainValidator(_validate_crams),
        pydantic.PlainSerializer(_serialize_crams, return_type=str),
    ]

    # fitted params are set to nan in these
    frozen_propagation: PropagationParams
    frozen_injection: InjectionParams

    population_meta: PopulationMetadata | None = None

    @functools.cached_property
    def frozen_injection_packed(self) -> np.ndarray:
        return pack_injection(self.frozen_injection)

    @functools.cached_property
    def frozen_propagation_packed(self) -> np.ndarray:
        return pack_propagation(self.frozen_propagation)

    @functools.cached_property
    def is_fitted_propagation(self) -> np.ndarray:
        return np.isnan(self.frozen_propagation_packed)

    @functools.cached_property
    def is_fitted_injection(self) -> np.ndarray:
        return np.isnan(self.frozen_injection_packed)

    @staticmethod
    def default(
        fit_abundances: Collection[Element] = [
            Element.H,
            Element.He,
            Element.C,
            Element.N,
            Element.O,
            Element.Ne,
            Element.Mg,
            Element.Si,
            Element.S,
            Element.Fe,
        ],
    ) -> "CramsModelConfig":
        return CramsModelConfig(
            crams=CramsRunner(
                # default CRAMS values, but explicitly specified
                inelastic_model="tripathi99",
                fragmentation_model="usinewebber03coste12",
            ),
            frozen_injection=InjectionParams(
                abundances=[
                    (np.nan if element in fit_abundances else DEFAULT_ABUNDANCES[element])
                    for element in CRAMS_ELEMENTS
                ],
                slopes=[np.nan] * 3,
            ),
            frozen_propagation=PropagationParams(
                H_kpc=7.0,
                v_A_km_sec=np.nan,
                R_b_GV=np.nan,
                delta=np.nan,
                ddelta=np.nan,
                D_0_cm2_sec=np.nan,
                X_src=-1.0,
                phi=np.nan,
            ),
            population_meta=PopulationMetadata(name="CRAMS", linestyle="-"),
        )

    @property
    def elements(self) -> list[Element]:
        return CRAMS_ELEMENTS


@dataclass
class CramsModel(Packable[CramsModelConfig]):
    propagation: PropagationParams
    injection: InjectionParams

    config: CramsModelConfig

    @staticmethod
    def default() -> "CramsModel":
        init_fitted_abundances = {
            Element.H: 4.14e-2,
            Element.He: 2.04e-2,
            Element.C: 4.00e-3,
            Element.N: 3.83e-4,
            Element.O: 7.21e-3,
            Element.Ne: 1.35e-3,
            Element.Mg: 2.38e-3,
            Element.Si: 2.83e-3,
            Element.S: 5.06e-4,
            Element.Fe: 7.53e-3,
        }
        return CramsModel(
            propagation=PropagationParams(
                H_kpc=7.0,
                D_0_cm2_sec=2.376e28,
                v_A_km_sec=3.32,
                R_b_GV=316.9,
                delta=0.54,
                ddelta=0.27,
                phi=0.47,
            ),
            injection=InjectionParams(
                abundances=[
                    init_fitted_abundances.get(element, DEFAULT_ABUNDANCES[element])
                    for element in CRAMS_ELEMENTS
                ],
                slopes=[4.37, 4.30, 4.36],
            ),
            config=CramsModelConfig.default(fit_abundances=list(init_fitted_abundances.keys())),
        )

    def description(self) -> str:
        prop_raw: str = self.propagation.to_input().describe().strip()
        prop = "\n".join(
            line
            for line in prop_raw.splitlines()
            # these parameters are actually set in the global CramsRunner object
            if not line.startswith(("flux solver", "inelastic model", "fragmentation model"))
        )
        abundances = "\n".join(
            [
                f"Q_{el} = {abundance}"
                for el, abundance in zip(CRAMS_ELEMENT_NAMES, self.injection.abundances)
            ]
        )
        slope_labels = [
            f"gamma_{el} = {slope}"
            for el, slope in zip(CRAMS_ELEMENT_NAMES, self.injection.slopes[:-1])
        ]
        if len(self.injection.slopes) == len(CRAMS_ELEMENT_NAMES):
            slope_labels.append(f"gamma_{CRAMS_ELEMENT_NAMES[-1]} = {self.injection.slopes[-1]}")
        else:
            slope_labels.append(f"gamma_nuclei = {self.injection.slopes[-1]}")
        slopes = "\n".join(slope_labels)
        return f"{prop}\n========\n{abundances}\n{slopes}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CramsModel):
            return False
        # NOTE: use with caution; this method ignores the embedded CramsRunner that might have different
        # configurations between self and other, and hence give different results
        return bool(
            np.isclose(self.propagation.H_kpc, other.propagation.H_kpc)
            and np.isclose(self.propagation.v_A_km_sec, other.propagation.v_A_km_sec)
            and np.isclose(self.propagation.R_b_GV, other.propagation.R_b_GV)
            and np.isclose(self.propagation.delta, other.propagation.delta)
            and np.isclose(self.propagation.ddelta, other.propagation.ddelta)
            and np.isclose(self.propagation.D_0_cm2_sec, other.propagation.D_0_cm2_sec)
            and np.isclose(self.propagation.X_src, other.propagation.X_src)
            and np.isclose(self.propagation.phi, other.propagation.phi)
            and np.allclose(self.injection.abundances, other.injection.abundances)
            and np.allclose(self.injection.slopes, other.injection.slopes)
        )

    @property
    def crams(self) -> CramsRunner:
        return self.config.crams

    @functools.cached_property
    def _crams_lg_output(self) -> np.ndarray:
        return np.log10(self.crams.compute(self.propagation, self.injection))

    @functools.cached_property
    def crams_lgR_grid(self) -> np.ndarray:
        return self._crams_lg_output[:, 0]

    def crams_lg_spectrum(self, element: Element) -> np.ndarray:
        return self._crams_lg_output[:, round(element.Z)]

    def compute_rigidity_spectrum(self, R: np.ndarray, element: Element) -> np.ndarray:
        return 10 ** (
            np.interp(
                np.log10(R),
                xp=self.crams_lgR_grid,
                fp=self.crams_lg_spectrum(element),
                left=np.nan,  # explicitly invalid for points out of computed bound
                right=np.nan,
            )
        )  # type: ignore

    def compute_spectrum(self, E: np.ndarray, element: Element) -> np.ndarray:
        Z = element.Z
        R = E / Z
        dNdR = self.compute_rigidity_spectrum(R, element=element)
        return dNdR / Z

    def compute_all_particle_spectrum(self, E: np.ndarray) -> np.ndarray:
        flux = np.zeros_like(E)
        for element in CRAMS_ELEMENTS:
            flux += self.compute_spectrum(E, element)
        return flux

    def compute_abundances(self, R: float) -> dict[Element | str, float]:
        return {
            element: float(self.compute_rigidity_spectrum(np.array([R]), element=element)[0])
            for element in CRAMS_ELEMENTS
        }

    @property
    def linestyle(self) -> str | None:
        if self.config.population_meta is None:
            return None
        else:
            return self.config.population_meta.linestyle

    def population_prefix(self, latex: bool) -> str:
        if self.config.population_meta is not None:
            return self.config.population_meta.plot_prefix(latex)
        else:
            return ""

    def plot(
        self,
        Emin: float,
        Emax: float,
        scale: float,
        axes: Axes | None = None,
        all_particle: bool = False,
        elements: list[Element] | None = None,
        grid_size: int = 100,
    ) -> Axes:
        if axes is not None:
            ax = axes
        else:
            _, ax = plt.subplots()

        E_grid = np.logspace(np.log10(Emin), np.log10(Emax), grid_size)
        E_factor = E_grid**scale
        label_prefix = self.population_prefix(latex=False)

        def with_prefix(name: str, preserve_capitalization: bool = False) -> str:
            if not label_prefix:
                return name.capitalize()
            else:
                return label_prefix + (name if preserve_capitalization else name.lower())

        for element in elements or CRAMS_ELEMENTS:
            ax.plot(
                E_grid,
                E_factor * self.compute_spectrum(E_grid, element),
                label=with_prefix(element.name, preserve_capitalization=True),
                color=element.color,
                linestyle=self.linestyle,
            )
        if all_particle:
            ax.plot(
                E_grid,
                E_factor * self.compute_all_particle_spectrum(E_grid),
                label=with_prefix("All particle"),
                color="black",
                linestyle=self.linestyle,
            )
        return ax

    def labels(self, latex: bool) -> list[str]:
        labels: list[str] = []

        if latex:
            propagation_labels = [
                "H_kpc",
                "v_A_km_sec",
                "R_b_GV",
                "delta",
                "ddelta",
                "D_0_cm2_sec",
                "X_src",
                "phi",
            ]
        else:
            propagation_labels = [
                r"$H \; / \; \text{kpc}$",
                r"$v_A \; / \; \text{km} \; \text{s}^{-1}$",
                r"$R_b \; / \; \text{GV}$",
                r"$\delta$",
                r"$\Delta \delta$",
                r"$D_0 \; / \; \text{cm}^2 \; \text{s}^{-1}$",
                r"$X_{\text{src}} \; / \; \text{g} \text{cm}^{-2}$",
                r"$\phi \; / \; \text{GV}$",
            ]
        labels.extend(
            [
                lbl
                for lbl, is_fitted in zip(propagation_labels, self.config.is_fitted_propagation)
                if is_fitted
            ]
        )

        injection_labels = [f"q_\\text{{{el.name}}}" for el in CRAMS_ELEMENTS]
        injection_labels.extend(
            [
                f"\\gamma_\\text{{{el}}}"
                for el, _ in zip(CRAMS_ELEMENT_NAMES, self.injection.slopes[:-1])
            ]
        )
        if len(self.injection.slopes) == len(CRAMS_ELEMENT_NAMES):
            injection_labels.append(f"\\gamma_\\text{{{CRAMS_ELEMENT_NAMES[-1]}}}")
        else:
            injection_labels.append("\\gamma_\\text{nuc}")
        labels.extend(
            [
                lbl
                for lbl, is_fitted in zip(injection_labels, self.config.is_fitted_injection)
                if is_fitted
            ]
        )

        prefix = self.population_prefix(latex)
        labels = [prefix + label for label in labels]
        return labels

    def pack(self) -> np.ndarray:
        return np.concatenate(
            (
                pack_propagation(self.propagation)[self.config.is_fitted_propagation],
                pack_injection(self.injection)[self.config.is_fitted_injection],
            )
        )

    def ml_bounds(self) -> list[tuple[float, float] | None] | None:
        injection_bounds = [ABUNDANCE_BOUND] * len(self.injection.abundances) + [SLOPE_BOUND] * len(
            self.injection.slopes
        )
        return [
            bound
            for bound, is_fitted in zip(PROPAGATION_BOUNDS, self.config.is_fitted_propagation)
            if is_fitted
        ] + [
            bound
            for bound, is_fitted in zip(injection_bounds, self.config.is_fitted_injection)
            if is_fitted
        ]

    def ndim(self) -> int:
        return np.sum(self.config.is_fitted_injection) + np.sum(self.config.is_fitted_propagation)

    def layout_info(self) -> CramsModelConfig:
        return self.config

    @classmethod
    def unpack(cls, theta: np.ndarray, layout_info: CramsModelConfig) -> "CramsModel":
        config = layout_info

        propagation_packed = config.frozen_propagation_packed
        n_prop = np.sum(config.is_fitted_propagation)
        propagation_packed[config.is_fitted_propagation] = theta[:n_prop]

        injection_packed = config.frozen_injection_packed
        n_inj = np.sum(config.is_fitted_injection)
        injection_packed[config.is_fitted_injection] = theta[n_prop : (n_prop + n_inj)]

        return CramsModel(
            injection=unpack_injection(injection_packed),
            propagation=unpack_propagation(propagation_packed),
            config=config,
        )


if __name__ == "__main__":
    ip = InjectionParams.default()
    ip2 = unpack_injection(pack_injection(ip))
    assert np.allclose(np.array(ip.abundances), ip2.abundances)
    assert np.allclose(np.array(ip.slopes), ip2.slopes)

    cm = CramsModel(
        injection=InjectionParams.default(),
        propagation=PropagationParams(),
        config=CramsModelConfig.default(),
    )
    print(cm.description())
    print(cm.pack())
    cm.validate_packing()
    print(cm.ml_bounds())
