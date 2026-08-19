import dataclasses
import functools
import json
from collections.abc import Collection
from dataclasses import dataclass
from typing import ClassVar, Literal

import matplotlib.pyplot as plt
import numpy as np
import pydantic
import scipy.interpolate  # type: ignore
from crams import (
    CRAMS_DEFAULT_R_OUT_GRID,
    CRAMS_DEFAULT_T_SIM_GRID,
    CramsError,
    CramsRunner,
    InjectionBreak,
    InjectionParams,
    LogGrid,
    LognormalRmaxDistribution,
    PropagationParams,
)
from crams import ELEMENT_NAMES as CRAMS_ELEMENT_NAMES
from matplotlib.axes import Axes
from scipy import stats

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
    match ip.feature:
        case None:
            pass
        case InjectionBreak() as br:
            components.append((np.log10(br.R_GV), br.delta_slope, br.omega))
        case LognormalRmaxDistribution() as dist:
            components.append((np.log10(dist.R_mean_GV), dist.sigma, dist.beta))
    return np.concatenate(components)


def unpack_injection(
    v: np.ndarray,
    n_slopes: int,
    feature_class: type[LognormalRmaxDistribution | InjectionBreak] | None,
) -> InjectionParams:
    abundances = v[: len(CRAMS_ELEMENTS)]
    offset = len(CRAMS_ELEMENTS)
    slopes = v[offset : offset + n_slopes]
    offset += n_slopes
    if feature_class is not None:
        feature = feature_class(10 ** v[offset], v[offset + 1], v[offset + 2])
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
            pp.D_0_cm2_sec / 1e28,
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
    (1e-2, 1e2),
    None,
    (0.1, 1.0),
]


def unpack_propagation(v: np.ndarray) -> PropagationParams:
    D_0_1e28_cm2_sec = v[5]
    # backwards compatibility with the old format, using plain D0, not divided by 10^28
    if D_0_1e28_cm2_sec > 1e20:
        D_0_cm2_sec = D_0_1e28_cm2_sec
    else:
        D_0_cm2_sec = D_0_1e28_cm2_sec * 1e28
    return PropagationParams(
        H_kpc=v[0],
        v_A_km_sec=v[1],
        R_b_GV=10 ** v[2],
        delta=v[3],
        ddelta=v[4],
        D_0_cm2_sec=D_0_cm2_sec,
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
LOGNORMAL_RMAX_DISTRIBUTION_BOUNDS = [
    SOURCE_BREAK_BOUNDS[0],
    (0.0, 1.0),
    (-5, 5),
]


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


SourceFeatureSpec = Literal["break", "erfc-cutoff", "none"]


FREE = np.nan  # used to signify non-frozen parameter within the config


@dataclass
class CramsRunnerConfig:
    """
    A small config class with all the data needed to create runners. The actual runners are stored as singletons
    in a class-level cache. This way we support multiprocessing, because each subprocess will have it's own
    CramsRunner object with no need to pass them between.
    """

    T_sim_grid: LogGrid
    R_out_grid: LogGrid

    _runner_cache: ClassVar[dict[str, CramsRunner]] = {}

    @functools.cached_property
    def cache_key(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    def implement(self) -> CramsRunner:
        if self.cache_key not in self._runner_cache:
            self._runner_cache[self.cache_key] = CramsRunner(
                T_sim_grid=self.T_sim_grid,
                R_out_grid=self.R_out_grid,
            )
        return self._runner_cache[self.cache_key]


class CramsModelConfig(pydantic.BaseModel):
    runner: CramsRunnerConfig

    # fitted params are set to nan in these
    frozen_propagation: PropagationParams
    frozen_injection: InjectionParams

    # NOTE: cubic spline needs work, now it's unpysical
    interpolation_method: Literal["numpy_linear", "scipy_linear", "scipy_cubic"] = "scipy_linear"

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
    def make(
        up2PeV: bool,
        source_feature: SourceFeatureSpec,
        fit_abundances: Collection[Element] = (
            [Element.H, Element.He, Element.C, Element.N, Element.O]
            + [Element.Ne, Element.Mg, Element.Si, Element.S, Element.Fe]
        ),
        freeze_propagation: bool = False,
        freeze_injection: bool = False,
    ) -> "CramsModelConfig":
        if up2PeV:
            T_sim_grid = LogGrid(
                min=0.1,  # 0.1 GeV
                max=5e7,  # 3 PeV
                size=200,  # enough for numerical errors <0.1%
            )
            R_out_grid = LogGrid(
                min=5,  # GV
                max=1e7,  # PV
                size=200,
            )
        else:
            T_sim_grid = CRAMS_DEFAULT_T_SIM_GRID
            R_out_grid = CRAMS_DEFAULT_R_OUT_GRID

        match source_feature:
            case "break":
                feature: InjectionBreak | LognormalRmaxDistribution | None = InjectionBreak(
                    R_GV=FREE,
                    delta_slope=FREE,
                    omega=FREE,
                )
            case "erfc-cutoff":
                feature = LognormalRmaxDistribution(FREE, FREE, 1.0)
            case "none":
                feature = None

        if freeze_injection:
            # best-fit params from CRAMS-only fit
            frozen_abundances = {
                Element.H: 0.0465,
                Element.He: 0.0223,
                Element.C: 0.00465,
                Element.N: 0.000452,
                Element.O: 0.00843,
                Element.Ne: 0.00161,
                Element.Mg: 0.00284,
                Element.Si: 0.00333,
                Element.S: 0.00061,
                Element.Fe: 0.00906,
            }
            slopes = [4.41, 4.33, 4.39]
        else:
            frozen_abundances = {}
            slopes = [FREE] * 3

        if freeze_propagation:
            frozen_propagation = PropagationParams(
                H_kpc=7.0,
                v_A_km_sec=0.000101,
                R_b_GV=10**2.58,
                delta=0.469,
                ddelta=0.319,
                D_0_cm2_sec=3.55e28,
                X_src=-1.0,
                phi=0.631,
            )
        else:
            frozen_propagation = PropagationParams(
                H_kpc=7.0,
                v_A_km_sec=FREE,
                R_b_GV=FREE,
                delta=FREE,
                ddelta=FREE,
                D_0_cm2_sec=FREE,
                X_src=-1.0,
                phi=FREE,
            )

        return CramsModelConfig(
            runner=CramsRunnerConfig(T_sim_grid=T_sim_grid, R_out_grid=R_out_grid),
            frozen_injection=InjectionParams(
                abundances=[
                    (
                        frozen_abundances.get(element, FREE)
                        if element in fit_abundances
                        else DEFAULT_ABUNDANCES[element]
                    )
                    for element in CRAMS_ELEMENTS
                ],
                slopes=slopes,
                feature=(feature),
            ),
            frozen_propagation=frozen_propagation,
            population_meta=PopulationMetadata(name="CRAMS", linestyle="-"),
        )

    @property
    def elements(self) -> list[Element]:
        return CRAMS_ELEMENTS

    def unpack_injection(self, v: np.ndarray):
        return unpack_injection(
            v,
            n_slopes=len(self.frozen_injection.slopes),
            feature_class=(
                type(self.frozen_injection.feature)
                if self.frozen_injection.feature is not None
                else None
            ),
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
    def make(
        up2PeV: bool,
        source_feature: SourceFeatureSpec,
        freeze_propagation: bool = False,
        freeze_injection: bool = False,
        randomize_init: bool = False,
    ) -> "CramsModel":

        def _maybe_randomized(v: float, sigma: float) -> float:
            if randomize_init:
                return stats.norm.rvs(v, sigma)
            else:
                return v

        init_fitted_abundances = {
            Element.H: _maybe_randomized(4.14e-2, 1e-4),
            Element.He: _maybe_randomized(2.04e-2, 1e-4),
            Element.C: _maybe_randomized(4.00e-3, 1e-4),
            Element.N: _maybe_randomized(3.83e-4, 1e-4),
            Element.O: _maybe_randomized(7.21e-3, 1e-4),
            Element.Ne: _maybe_randomized(1.35e-3, 1e-4),
            Element.Mg: _maybe_randomized(2.38e-3, 1e-4),
            Element.Si: _maybe_randomized(2.83e-3, 1e-4),
            Element.S: _maybe_randomized(5.06e-4, 1e-4),
            Element.Fe: _maybe_randomized(7.53e-3, 1e-4),
        }
        init_fitted_slopes = [
            _maybe_randomized(4.37, 0.05),
            _maybe_randomized(4.30, 0.05),
            _maybe_randomized(4.36, 0.05),
        ]
        init_prop = PropagationParams(
            H_kpc=_maybe_randomized(7.0, 0.1),
            D_0_cm2_sec=_maybe_randomized(2.376e28, 1e27),
            v_A_km_sec=_maybe_randomized(3.32, 0.1),
            R_b_GV=_maybe_randomized(316.9, 10),
            delta=_maybe_randomized(0.54, 0.05),
            ddelta=_maybe_randomized(0.27, 0.05),
            phi=_maybe_randomized(0.47, 0.05),
            X_src=-1.0,
        )

        feature: InjectionBreak | LognormalRmaxDistribution | None
        match source_feature:
            case "break":
                feature = InjectionBreak(
                    R_GV=_maybe_randomized(15e3, 1e3),
                    delta_slope=_maybe_randomized(0.2, 0.05),
                    omega=_maybe_randomized(0.2, 0.05),
                )
            case "erfc-cutoff":
                feature = LognormalRmaxDistribution(
                    R_mean_GV=_maybe_randomized(15e3, 1e3),
                    sigma=_maybe_randomized(0.5, 0.05),
                    beta=1.0,
                )
            case "none":
                feature = None

        config = CramsModelConfig.make(
            up2PeV=up2PeV,
            fit_abundances=list(init_fitted_abundances.keys()),
            source_feature=source_feature,
            freeze_propagation=freeze_propagation,
            freeze_injection=freeze_injection,
        )

        def _if_not_frozen(v: float, default: float) -> float:
            if np.isnan(default):
                return v
            else:
                return default

        return CramsModel(
            propagation=PropagationParams(
                H_kpc=_if_not_frozen(init_prop.H_kpc, config.frozen_propagation.H_kpc),
                D_0_cm2_sec=_if_not_frozen(
                    init_prop.D_0_cm2_sec, config.frozen_propagation.D_0_cm2_sec
                ),
                v_A_km_sec=_if_not_frozen(
                    init_prop.v_A_km_sec, config.frozen_propagation.v_A_km_sec
                ),
                R_b_GV=_if_not_frozen(init_prop.R_b_GV, config.frozen_propagation.R_b_GV),
                delta=_if_not_frozen(init_prop.delta, config.frozen_propagation.delta),
                ddelta=_if_not_frozen(init_prop.ddelta, config.frozen_propagation.ddelta),
                phi=_if_not_frozen(init_prop.phi, config.frozen_propagation.phi),
                X_src=_if_not_frozen(init_prop.X_src, config.frozen_propagation.X_src),
            ),
            injection=InjectionParams(
                abundances=[
                    _if_not_frozen(
                        init_fitted_abundances.get(element, DEFAULT_ABUNDANCES[element]),
                        frozen_abundance,
                    )
                    for element, frozen_abundance in zip(
                        CRAMS_ELEMENTS, config.frozen_injection.abundances, strict=True
                    )
                ],
                slopes=[
                    _if_not_frozen(sl, frozen)
                    for sl, frozen in zip(init_fitted_slopes, config.frozen_injection.slopes)
                ],
                feature=feature,
            ),
            config=config,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CramsModel):
            return False

        # NOTE: use with caution; this method ignores the embedded CramsRunner that might have different
        # configurations between self and other, and hence give different results

        match self.injection.feature:
            case None:
                is_feature_same = np.bool(other.injection.feature is None)
            case InjectionBreak() as b1:
                b2 = other.injection.feature
                if not isinstance(b2, InjectionBreak):
                    return False
                is_feature_same = (
                    np.isclose(b1.R_GV, b2.R_GV)
                    and np.isclose(b1.delta_slope, b2.delta_slope)
                    and np.isclose(b1.omega, b2.omega)
                )
            case LognormalRmaxDistribution() as d1:
                d2 = other.injection.feature
                if not isinstance(d2, LognormalRmaxDistribution):
                    return False
                is_feature_same = (
                    np.isclose(d1.R_mean_GV, d2.R_mean_GV)
                    and np.isclose(d1.sigma, d2.sigma)
                    and np.isclose(d1.beta, d2.beta)
                )

        flags = [
            np.isclose(self.propagation.H_kpc, other.propagation.H_kpc),
            np.isclose(self.propagation.v_A_km_sec, other.propagation.v_A_km_sec),
            np.isclose(self.propagation.R_b_GV, other.propagation.R_b_GV),
            np.isclose(self.propagation.delta, other.propagation.delta),
            np.isclose(self.propagation.ddelta, other.propagation.ddelta),
            np.isclose(self.propagation.D_0_cm2_sec, other.propagation.D_0_cm2_sec),
            np.isclose(self.propagation.X_src, other.propagation.X_src),
            np.isclose(self.propagation.phi, other.propagation.phi),
            np.allclose(self.injection.abundances, other.injection.abundances),
            np.allclose(self.injection.slopes, other.injection.slopes),
            is_feature_same,
        ]
        # print(flags)

        return bool(np.all(flags))  # type: ignore

    @property
    def crams(self) -> CramsRunner:
        return self.config.runner.implement()

    @functools.cached_property
    def _crams_lg_output(self) -> np.ndarray:
        return np.log10(self.crams.compute(self.propagation, self.injection))

    @functools.cached_property
    def crams_lgR_grid(self) -> np.ndarray:
        return self._crams_lg_output[:, 0]

    def crams_lg_spectrum(self, element: Element) -> np.ndarray:
        Z = round(element.Z)
        return self._crams_lg_output[:, Z]

    def _lg_spectrum_spline(self, element: Element) -> scipy.interpolate.BSpline | None:
        # TODO: add instance-level caching
        if element.Z > 28:
            return None
        lg_flux = self.crams_lg_spectrum(element)
        mask = np.isfinite(lg_flux)
        if np.count_nonzero(mask) < 3:
            return None  # no interpolator for all-zero data
        return scipy.interpolate.make_interp_spline(
            x=self.crams_lgR_grid[mask].squeeze(),
            y=lg_flux[mask].squeeze(),
            k=1 if self.config.interpolation_method == "scipy_linear" else 3,
        )

    def _compute_rigidity_spectrum(self, R: np.ndarray, element: Element) -> np.ndarray:
        try:
            match self.config.interpolation_method:
                case "numpy_linear":
                    lg_res = np.interp(
                        x=np.log10(R),
                        xp=self.crams_lgR_grid,
                        fp=self.crams_lg_spectrum(element),
                        left=np.nan,  # explicitly invalid for points out of computed bound
                        right=np.nan,
                    )
                case "scipy_linear" | "scipy_cubic":
                    lgR = np.log10(R)
                    if spline := self._lg_spectrum_spline(element):
                        lg_res = spline(lgR)
                    else:
                        return np.zeros_like(R)
            return 10**lg_res
        except CramsError:
            return np.zeros_like(R) * np.nan

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
        caption_elements: bool = True,
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
            if element not in CRAMS_ELEMENTS:
                continue
            ax.plot(
                E_grid,
                E_factor * self.compute_spectrum(E_grid, element, quantity="E"),
                label=(
                    with_prefix(element.name, preserve_capitalization=True)
                    if caption_elements
                    else None
                ),
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
                "Halo size",
                "v_A",
                "lg R_b",
                "delta",
                "Delta delta",
                "D_0",
                "X_src",
                "phi",
            ]
        else:
            all_prop_labels = [
                r"H \; / \; \text{kpc}",
                r"v_A \; / \; \text{km} \; \text{s}^{-1}",
                r"\lg ( \mathcal{R}_b \; / \; \text{GV} )",
                r"\delta",
                r"\Delta \delta",
                r"D_0 \; / \; 10^{28} \text{cm}^2 \; \text{s}^{-1}",
                r"X_{\text{src}} \; / \; \text{g} \text{cm}^{-2}",
                r"\phi \; / \; \text{GV}",
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
            match self.injection.feature:
                case InjectionBreak():
                    all_inj_labels.extend(
                        [
                            r"\lg \mathcal{R}_{b, \text{src}} \; / \; \text{GV}",
                            r"\Delta \gamma",
                            r"\omega",
                        ]
                    )
                case LognormalRmaxDistribution():
                    all_inj_labels.extend(
                        [
                            r"\langle \lg \mathcal{R}_{max} \; / \; \text{GV} \rangle",
                            r"\sigma(\lg \mathcal{R}_{max} \; / \; \text{GV})",
                            r"\beta",
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
            match self.injection.feature:
                case InjectionBreak():
                    all_inj_labels.extend(
                        [
                            r"lg R_b, src",
                            r"Delta gamma",
                            r"omega",
                        ]
                    )
                case LognormalRmaxDistribution():
                    all_inj_labels.extend(
                        [
                            r"<lg R_max>",
                            r"sigma(lg R_max)",
                            r"beta",
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
    for up2PeV in (False, True):
        for source_feature in ("break", "none", "erfc-cutoff"):
            for freeze_prop in (False, True):
                for freeze_inj in (False, True):
                    print("\n===\n")
                    cm = CramsModel.make(
                        up2PeV=up2PeV,
                        source_feature=source_feature,
                        freeze_propagation=freeze_prop,
                        freeze_injection=freeze_inj,
                    )
                    cm.print_params()
                    print(cm.pack())
                    cm.validate_packing()
                    print(cm.ml_bounds())
