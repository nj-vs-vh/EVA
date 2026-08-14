import functools
import json
from collections.abc import Collection
from dataclasses import dataclass
from typing import Annotated, Any

import matplotlib.pyplot as plt
import numpy as np
import pydantic
from crams import (
    CRAMS_DEFAULT_R_OUT_GRID,
    CRAMS_DEFAULT_T_SIM_GRID,
    CramsRunner,
    InjectionBreak,
    InjectionParams,
    LogGrid,
    PropagationParams,
)
from crams import (
    ELEMENT_NAMES as CRAMS_ELEMENT_NAMES,
)
from matplotlib.axes import Axes

from cr_knee_fit.cr_model import (
    CharacteristicQuantity,
    PopulationMetadata,
)
from cr_knee_fit.elements import Element
from cr_knee_fit.types_ import Packable
from cr_knee_fit.utils import q2R_factor

CRAMS_ELEMENTS = [Element[name] for name in CRAMS_ELEMENT_NAMES]


def pack_injection(ip: InjectionParams) -> np.ndarray:
    components = [ip.abundances, ip.slopes]
    if ip.feature is not None:
        components.append((np.log10(ip.feature.R_GV), ip.feature.delta_slope, ip.feature.omega))
    return np.concatenate(components)


def unpack_injection(v: np.ndarray, n_slopes: int, has_feature: bool) -> InjectionParams:
    abundances = v[: len(CRAMS_ELEMENTS)]
    offset = len(CRAMS_ELEMENTS)
    slopes = v[offset : offset + n_slopes]
    offset += n_slopes
    if has_feature:
        feature = InjectionBreak(
            R_GV=10 ** v[offset],
            delta_slope=v[offset + 1],
            omega=v[offset + 2],
        )
        offset += 2
    else:
        feature = None
    return InjectionParams(
        abundances=abundances,  # type: ignore
        slopes=slopes,  # type: ignore
        feature=feature,
    )


def pack_propagation(pp: PropagationParams) -> np.ndarray:
    return np.array(
        [
            pp.H_kpc,
            pp.v_A_km_sec,
            np.log10(pp.R_b_GV),
            pp.delta,
            pp.ddelta,
            pp.D_0_cm2_sec,
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
    (1e26, 1e30),
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
        D_0_cm2_sec=v[5],
        X_src=v[6],
        phi=v[7],
    )


ABUNDANCE_BOUND = (0.0, 10.0)
SLOPE_BOUND = (4.01, 4.99)  # (4, 5) is strictly enforced by CRAMS, here we add a small margin
SOURCE_BREAK_BOUNDS = [
    (np.log10(5e3), np.log10(30e3)),  # DAMPE break expected at ~13 TV
    (0.0, 2.0),  # break should harden the spectrum
    (0.01, 1.0),
]


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
        include_10TeV_break: bool,
        fit_abundances: Collection[Element] = (
            [Element.H, Element.He, Element.C, Element.N, Element.O]
            + [Element.Ne, Element.Mg, Element.Si, Element.S, Element.Fe]
        ),
    ) -> "CramsModelConfig":
        if include_10TeV_break:
            T_sim_grid = LogGrid(
                min=0.1,  # 0.1 GeV
                max=5e6,  # 3 PeV
                size=200,  # enough for numerical errors <0.1%
            )
            R_out_grid = LogGrid(
                min=5,  # GV
                max=1e6,  # PV
                size=150,
            )
        else:
            T_sim_grid = CRAMS_DEFAULT_T_SIM_GRID
            R_out_grid = CRAMS_DEFAULT_R_OUT_GRID

        return CramsModelConfig(
            crams=CramsRunner(T_sim_grid=T_sim_grid, R_out_grid=R_out_grid),
            frozen_injection=InjectionParams(
                abundances=[
                    (np.nan if element in fit_abundances else DEFAULT_ABUNDANCES[element])
                    for element in CRAMS_ELEMENTS
                ],
                slopes=[np.nan] * 3,
                feature=(
                    InjectionBreak(R_GV=np.nan, delta_slope=np.nan, omega=np.nan)
                    if include_10TeV_break
                    else None
                ),
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

    def unpack_injection(self, v: np.ndarray):
        return unpack_injection(
            v,
            n_slopes=len(self.frozen_injection.slopes),
            has_feature=self.frozen_injection.feature is not None,
        )


@dataclass
class CramsModel(Packable[CramsModelConfig]):
    propagation: PropagationParams
    injection: InjectionParams

    config: CramsModelConfig

    def __post_init__(self) -> None:
        assert len(self.injection.abundances) == len(self.config.frozen_injection.abundances)
        assert len(self.injection.slopes) == len(self.config.frozen_injection.slopes)
        assert type(self.injection.feature) == type(self.config.frozen_injection.feature)

    @staticmethod
    def default(include_10TeV_break: bool) -> "CramsModel":
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
                feature=(
                    InjectionBreak(R_GV=15e3, delta_slope=0.2, omega=0.2)
                    if include_10TeV_break
                    else None
                ),
            ),
            config=CramsModelConfig.default(
                include_10TeV_break=include_10TeV_break,
                fit_abundances=list(init_fitted_abundances.keys()),
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CramsModel):
            return False

        # NOTE: use with caution; this method ignores the embedded CramsRunner that might have different
        # configurations between self and other, and hence give different results

        if self.injection.feature is not None:
            if other.injection.feature is None:
                return False
            is_feature_same = (
                np.isclose(self.injection.feature.R_GV, other.injection.feature.R_GV)
                and np.isclose(
                    self.injection.feature.delta_slope, other.injection.feature.delta_slope
                )
                and np.isclose(self.injection.feature.omega, other.injection.feature.omega)
            )
        else:
            is_feature_same = np.True_

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
            and is_feature_same
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

    def _compute_rigidity_spectrum(self, R: np.ndarray, element: Element) -> np.ndarray:
        return 10 ** (
            np.interp(
                np.log10(R),
                xp=self.crams_lgR_grid,
                fp=self.crams_lg_spectrum(element),
                left=np.nan,  # explicitly invalid for points out of computed bound
                right=np.nan,
            )
        )  # type: ignore

    def compute_spectrum(
        self, Q: np.ndarray, element: Element, quantity: CharacteristicQuantity
    ) -> np.ndarray:
        q2R = q2R_factor(element.Z, element.A, quantity=quantity)
        dNdR = self._compute_rigidity_spectrum(R=q2R * Q, element=element)
        return dNdR * q2R

    def compute_all_particle_spectrum(self, E: np.ndarray) -> np.ndarray:
        flux = np.zeros_like(E)
        for element in CRAMS_ELEMENTS:
            flux += self.compute_spectrum(E, element, quantity="E")
        return flux

    def compute_abundances(self, R: float) -> dict[Element | str, float]:
        return {
            element: float(self._compute_rigidity_spectrum(np.array([R]), element=element)[0])
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
                E_factor * self.compute_spectrum(E_grid, element, quantity="E"),
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
        if not latex:
            all_prop_labels = [
                "H / kpc",
                "v_A / km/sec",
                "lg R_b / GV",
                "delta",
                "ddelta",
                "D_0 / cm2/sec",
                "X_src",
                "phi",
            ]
        else:
            all_prop_labels = [
                r"$H \; / \; \text{kpc}$",
                r"$v_A \; / \; \text{km} \; \text{s}^{-1}$",
                r"$\lg ( \mathcal{R}_b \; / \; \text{GV} )$",
                r"$\delta$",
                r"$\Delta \delta$",
                r"$D_0 \; / \; \text{cm}^2 \; \text{s}^{-1}$",
                r"$X_{\text{src}} \; / \; \text{g} \text{cm}^{-2}$",
                r"$\phi \; / \; \text{GV}$",
            ]

        if latex:
            all_inj_labels = [f"q_\\text{{{el.name}}}" for el in CRAMS_ELEMENTS]
            all_inj_labels.extend(
                [
                    f"\\gamma_\\text{{{el}}}"
                    for el, _ in zip(CRAMS_ELEMENT_NAMES, self.injection.slopes[:-1])
                ]
            )
            if len(self.injection.slopes) == len(CRAMS_ELEMENT_NAMES):
                all_inj_labels.append(f"\\gamma_\\text{{{CRAMS_ELEMENT_NAMES[-1]}}}")
            else:
                all_inj_labels.append("\\gamma_\\text{nuc}")
            if self.injection.feature is not None:
                all_inj_labels.extend(
                    [
                        r"\lg \mathcal{R}_{b, \text{src}} \; / \; \text{GV}",
                        r"\Delta \gamma",
                        r"\omega",
                    ]
                )
        else:
            all_inj_labels = [f"q_{el.name}" for el in CRAMS_ELEMENTS]
            all_inj_labels.extend(
                [f"gamma_{el}" for el, _ in zip(CRAMS_ELEMENT_NAMES, self.injection.slopes[:-1])]
            )
            if len(self.injection.slopes) == len(CRAMS_ELEMENT_NAMES):
                all_inj_labels.append(f"gamma_{CRAMS_ELEMENT_NAMES[-1]}")
            else:
                all_inj_labels.append("gamma_nuc")
            if self.injection.feature is not None:
                all_inj_labels.extend(
                    [
                        r"lg R_b, src / GV",
                        r"Delta gamma",
                        r"omega",
                    ]
                )

        labels = [
            pl
            for pl, is_fitted in zip(all_prop_labels, self.config.is_fitted_propagation)
            if is_fitted
        ] + [
            il
            for il, is_fitted in zip(all_inj_labels, self.config.is_fitted_injection)
            if is_fitted
        ]
        labels = [self.population_prefix(latex) + label for label in labels]
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
        if self.injection.feature is not None:
            injection_bounds.extend(SOURCE_BREAK_BOUNDS)

        return [
            prop_bound
            for prop_bound, is_fitted in zip(PROPAGATION_BOUNDS, self.config.is_fitted_propagation)
            if is_fitted
        ] + [
            inj_bound
            for inj_bound, is_fitted in zip(injection_bounds, self.config.is_fitted_injection)
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
            injection=layout_info.unpack_injection(injection_packed),
            propagation=unpack_propagation(propagation_packed),
            config=config,
        )


if __name__ == "__main__":
    for include_10TeV_break in (False, True):
        print("\n===\n")
        cm = CramsModel.default(include_10TeV_break=include_10TeV_break)
        print(cm.print_params())
        print(cm.pack())
        cm.validate_packing()
        print(cm.ml_bounds())
