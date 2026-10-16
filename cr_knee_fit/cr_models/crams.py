import dataclasses
import functools
import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal

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
    InjectionExpCutoff,
    InjectionFeature,
    InjectionLognormalDist,
    InjectionParams,
    LogGrid,
    PropagationParams,
)
from crams import ELEMENT_NAMES as CRAMS_ELEMENT_NAMES
from matplotlib.axes import Axes
from scipy import stats

from cr_knee_fit.cr_models.utils import PopulationMetadata
from cr_knee_fit.elements import Element
from cr_knee_fit.types_ import Packable
from cr_knee_fit.utils import CharacteristicQuantity, q2R_factor

CRAMS_ELEMENTS = [Element[name] for name in CRAMS_ELEMENT_NAMES]


def pack_injection(ip: InjectionParams) -> np.ndarray:
    components = [ip.abundances, ip.slopes]
    for feature in ip.features:
        match feature:
            case InjectionBreak() as br:
                components.append((np.log10(br.R_GV), br.delta_slope, br.omega))
            case InjectionLognormalDist() as dist:
                components.append((np.log10(dist.R_mean_GV), dist.sigma, dist.beta))
            case InjectionExpCutoff() as cutoff:
                components.append((np.log10(cutoff.R_cut_GV), cutoff.Delta, cutoff.is_lower))
    return np.concatenate(components)


def unpack_injection(
    v: np.ndarray,
    n_slopes: int,
    feature_classes: list[type[InjectionFeature]] | None,
) -> InjectionParams:
    abundances = v[: len(CRAMS_ELEMENTS)]
    offset = len(CRAMS_ELEMENTS)
    slopes = v[offset : offset + n_slopes]
    offset += n_slopes
    features: list[InjectionFeature] = []
    if feature_classes is not None:
        for cls in feature_classes:
            if cls in (InjectionBreak, InjectionLognormalDist, InjectionExpCutoff):
                feature = cls(10 ** v[offset], v[offset + 1], v[offset + 2])
                offset += 3
            else:
                raise ValueError(f"Unsupported feature class: {cls}")
            features.append(feature)
    return InjectionParams(
        abundances=abundances,  # type: ignore
        slopes=slopes,  # type: ignore
        features=features,
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
    (np.log10(5e3), np.log10(3e5)),  # DAMPE break expected at ~13 TV
    (0.0, 2.0),  # break should harden the spectrum
    (0.01, 1.0),
]
LOGNORMAL_RMAX_DISTRIBUTION_BOUNDS = [
    SOURCE_BREAK_BOUNDS[0],
    (0.0, 1.0),
    (-5, 5),
]
EXPCUTOFF_BOUNDS = [
    SOURCE_BREAK_BOUNDS[0],
    (0.0, 3.0),
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


SourceFeatureSpec = Literal["break", "erfc-cutoff", "exp-cutoff", "exp-cutoff-lower", "none"]


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


# HACK: pydantic sometimes dumps NaN as null, which breaks validation, so here we convert it back


def _dict_null2nan(value: Any):
    if isinstance(value, dict):
        return {k: v if v is not None else np.nan for k, v in value.items()}
    else:
        return value


def _list_null2nan(value: Any):
    if isinstance(value, list):
        return [item if item is not None else np.nan for item in value]
    else:
        return value


def _inj_null2nan(value: Any):
    if isinstance(value, dict):
        if abundances := value.get("abundances"):
            value["abundances"] = _list_null2nan(abundances)
        if slopes := value.get("slopes"):
            value["slopes"] = _list_null2nan(slopes)
        if feature := value.get("feature"):
            value["feature"] = _dict_null2nan(feature)
    return value


class CramsModelConfig(pydantic.BaseModel):
    runner: CramsRunnerConfig

    # fitted params are set to nan in these
    frozen_propagation: Annotated[PropagationParams, pydantic.BeforeValidator(_dict_null2nan)]
    frozen_injection: Annotated[InjectionParams, pydantic.BeforeValidator(_inj_null2nan)]

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
        source_feature: SourceFeatureSpec | InjectionFeature,
        fit_abundances: Collection[Element] = (
            [Element.H, Element.He, Element.C, Element.N, Element.O]
            + [Element.Ne, Element.Mg, Element.Si, Element.S, Element.Fe]
        ),
        freeze_propagation: bool | PropagationParams = False,
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

        features: list[InjectionFeature] = []
        match source_feature:
            case "break":
                features.append(InjectionBreak(R_GV=FREE, delta_slope=FREE, omega=FREE))
            case "erfc-cutoff":
                features.append(InjectionLognormalDist(FREE, FREE, beta=1.0))
            case "exp-cutoff":
                features.append(InjectionExpCutoff(FREE, FREE, is_lower=False))
            case "exp-cutoff-lower":
                features.append(InjectionExpCutoff(FREE, FREE, is_lower=True))
            case "none":
                pass
            case _:
                features.append(source_feature)

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

        match freeze_propagation:
            case PropagationParams():
                frozen_propagation = freeze_propagation
            case True:
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
            case False:
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
                features=features,
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
            feature_classes=[type(ftr) for ftr in self.frozen_injection.features],
        )


@dataclass
class CramsModel(Packable[CramsModelConfig]):
    propagation: PropagationParams
    injection: InjectionParams

    config: CramsModelConfig

    def __post_init__(self) -> None:
        assert len(self.injection.abundances) == len(self.config.frozen_injection.abundances)
        assert len(self.injection.slopes) == len(self.config.frozen_injection.slopes)
        assert len(self.injection.features) == len(self.config.frozen_injection.features)
        for inj_ftr, frozen_ftr in zip(
            self.injection.features, self.config.frozen_injection.features
        ):
            assert type(inj_ftr) == type(frozen_ftr)

    @staticmethod
    def make(
        up2PeV: bool,
        source_feature: SourceFeatureSpec | InjectionFeature,
        freeze_propagation: bool | PropagationParams = False,
        freeze_injection: bool = False,
        randomize_init: bool = False,
    ) -> "CramsModel":

        def _init(v: float, sigma: float) -> float:
            if randomize_init:
                return stats.norm.rvs(v, sigma)
            else:
                return v

        init_fitted_abundances = {
            Element.H: _init(4.14e-2, 1e-4),
            Element.He: _init(2.04e-2, 1e-4),
            Element.C: _init(4.00e-3, 1e-4),
            Element.N: _init(3.83e-4, 1e-4),
            Element.O: _init(7.21e-3, 1e-4),
            Element.Ne: _init(1.35e-3, 1e-4),
            Element.Mg: _init(2.38e-3, 1e-4),
            Element.Si: _init(2.83e-3, 1e-4),
            Element.S: _init(5.06e-4, 1e-4),
            Element.Fe: _init(7.53e-3, 1e-4),
        }
        init_fitted_slopes = [
            _init(4.37, 0.05),
            _init(4.30, 0.05),
            _init(4.36, 0.05),
        ]
        init_prop = PropagationParams(
            H_kpc=_init(7.0, 0.1),
            D_0_cm2_sec=_init(2.376e28, 1e27),
            v_A_km_sec=_init(3.32, 0.1),
            R_b_GV=_init(316.9, 10),
            delta=_init(0.54, 0.05),
            ddelta=_init(0.27, 0.05),
            phi=_init(0.47, 0.05),
            X_src=-1.0,
        )

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

        features: list[InjectionFeature] = []
        for feature in config.frozen_injection.features:
            match feature:
                case InjectionBreak() as b:
                    feature = InjectionBreak(
                        R_GV=_if_not_frozen(_init(15e3, 1e3), b.R_GV),
                        delta_slope=_if_not_frozen(_init(0.2, 0.05), b.delta_slope),
                        omega=_if_not_frozen(_init(0.2, 0.05), b.omega),
                    )
                case InjectionLognormalDist() as ln:
                    feature = InjectionLognormalDist(
                        R_mean_GV=_if_not_frozen(_init(15e3, 1e3), ln.R_mean_GV),
                        sigma=_if_not_frozen(_init(0.5, 0.05), ln.sigma),
                        beta=_if_not_frozen(_init(1.0, 0.05), ln.beta),
                    )
                case InjectionExpCutoff() as iec:
                    feature = InjectionExpCutoff(
                        R_cut_GV=_if_not_frozen(_init(15e3, 1e3), iec.R_cut_GV),
                        Delta=_if_not_frozen(_init(0.2, 0.05), iec.Delta),
                        is_lower=iec.is_lower,
                    )
            features.append(feature)

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
                features=features,
            ),
            config=config,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CramsModel):
            return False

        # NOTE: use with caution; this method ignores the embedded CramsRunner that might have different
        # configurations between self and other, and hence give different results

        is_feature_same = True
        for f1, f2 in zip(self.injection.features, other.injection.features):
            match f1:
                case None:
                    if f2 is not None:
                        return False
                case InjectionBreak() as b1:
                    if not isinstance(f2, InjectionBreak):
                        return False
                    is_feature_same &= bool(
                        np.isclose(b1.R_GV, f2.R_GV)
                        and np.isclose(b1.delta_slope, f2.delta_slope)
                        and np.isclose(b1.omega, f2.omega)
                    )
                case InjectionLognormalDist() as d1:
                    if not isinstance(f2, InjectionLognormalDist):
                        return False
                    is_feature_same &= bool(
                        np.isclose(d1.R_mean_GV, f2.R_mean_GV)
                        and np.isclose(d1.sigma, f2.sigma)
                        and np.isclose(d1.beta, f2.beta)
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
                        # we explicitly set extrapolation to low R to NaN
                        lg_res[lgR < self.crams_lgR_grid[0]] = np.nan
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
            for feature in self.injection.features:
                match feature:
                    case InjectionBreak():
                        all_inj_labels.extend(
                            [
                                r"\lg \mathcal{R}_{b, \text{src}} \; / \; \text{GV}",
                                r"\Delta \gamma",
                                r"\omega",
                            ]
                        )
                    case InjectionLognormalDist():
                        all_inj_labels.extend(
                            [
                                r"\langle \lg \mathcal{R}_{max} \; / \; \text{GV} \rangle",
                                r"\sigma(\lg \mathcal{R}_{max} \; / \; \text{GV})",
                                r"\beta",
                            ]
                        )
                    case InjectionExpCutoff():
                        all_inj_labels.extend(
                            [
                                r"R_\text{cut, src}",
                                r"\Delta \gamma",
                                r"\text{is_lower}",
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
            for feature in self.injection.features:
                match feature:
                    case InjectionBreak():
                        all_inj_labels.extend(
                            [
                                r"lg R_b, src",
                                r"Delta gamma",
                                r"omega",
                            ]
                        )
                    case InjectionLognormalDist():
                        all_inj_labels.extend(
                            [
                                r"<lg R_max>",
                                r"sigma(lg R_max)",
                                r"beta",
                            ]
                        )
                    case InjectionExpCutoff():
                        all_inj_labels.extend(
                            [
                                r"R_cut, src",
                                r"Delta gamma",
                                r"is_lower",
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

    def bounds(self) -> Sequence[tuple[float, float] | None]:
        injection_bounds = [ABUNDANCE_BOUND] * len(self.injection.abundances) + [SLOPE_BOUND] * len(
            self.injection.slopes
        )
        for feature in self.injection.features:
            if feature is not None:
                match feature:
                    case InjectionBreak():
                        injection_bounds.extend(SOURCE_BREAK_BOUNDS)
                    case InjectionLognormalDist():
                        injection_bounds.extend(LOGNORMAL_RMAX_DISTRIBUTION_BOUNDS)
                    case InjectionExpCutoff():
                        injection_bounds.extend(EXPCUTOFF_BOUNDS)

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
                    print(cm.bounds())
