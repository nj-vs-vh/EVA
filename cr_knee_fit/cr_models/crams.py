import copy
import dataclasses
import functools
import json
from collections.abc import Sequence
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

from cr_knee_fit.cr_models.utils import Param, PopulationMetadata, get_frozen_scheme, sample_init
from cr_knee_fit.elements import Element
from cr_knee_fit.types_ import Packable
from cr_knee_fit.utils import CharacteristicQuantity, q2R_factor

CRAMS_ELEMENTS = [Element[name] for name in CRAMS_ELEMENT_NAMES]


def pack_injection(ip: InjectionParams) -> np.ndarray:
    components = [np.log10(ip.abundances), ip.slopes]
    for feature in ip.features:
        match feature:
            case InjectionBreak() as br:
                components.append((np.log10(br.R_GV), br.delta_slope, br.omega))
            case InjectionLognormalDist() as dist:
                components.append((np.log10(dist.R_mean_GV), dist.sigma, dist.beta))
            case InjectionExpCutoff() as cutoff:
                components.append((np.log10(cutoff.R_cut_GV), cutoff.Delta, cutoff.is_lower))
    return np.concatenate(components)  # type: ignore


def unpack_injection(
    v: np.ndarray,
    n_slopes: int,
    feature_classes: list[type[InjectionFeature]] | None,
    abundances_are_precomputed_normalizations: bool = False,
) -> InjectionParams:
    abundances = 10 ** v[: len(CRAMS_ELEMENTS)]
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
        abundances_are_precomputed_normalizations=abundances_are_precomputed_normalizations,
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
    (0.1, 1.0),
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


# sentinel params that should be used in frozen_propagation when in fact the propagation
# params are shared between multiple CramsModel instances coordinated by Model
SHARED_PROPAGATION = PropagationParams(
    H_kpc=Param.frozen(0.0),
    v_A_km_sec=Param.frozen(0.0),
    R_b_GV=Param.frozen(0.0),
    delta=Param.frozen(0.0),
    ddelta=Param.frozen(0.0),
    D_0_cm2_sec=Param.frozen(0.0),
    phi=Param.frozen(0.0),
    smoothness=Param.frozen(0.0),
    X_src=Param.frozen(0.0),
)
# sentinel params that should be used in frozen_injection when in the nuclei abundances must
# be taken from another population
SHARED_NUCLEI_INJECTION = [Param.frozen(np.inf) for _ in range(len(CRAMS_ELEMENTS) - 2)]

DEFAULT_INIT_PROPAGATION = PropagationParams(
    H_kpc=Param.frozen(7.0),
    D_0_cm2_sec=Param.fitted(2.376e28, 1e27),
    v_A_km_sec=Param.fitted(3.32, 0.1),
    R_b_GV=Param.fitted(316.9, 10),
    delta=Param.fitted(0.54, 0.05),
    ddelta=Param.fitted(0.27, 0.05),
    smoothness=Param.frozen(0.1),
    phi=Param.fitted(0.47, 0.05),
    X_src=Param.frozen(-1.0),
)


def init_injection(
    fitted_abundances: dict[Element, float],
    slopes: list[float],
    features: list[InjectionFeature] | None = None,
    frozen_abundances: dict[Element, float] = DEFAULT_ABUNDANCES,
) -> InjectionParams:
    _init_abundances: list[float] = []
    for element in CRAMS_ELEMENTS:
        if fitted := fitted_abundances.get(element):
            if isinstance(fitted, Param):
                _init_abundances.append(fitted)
            else:
                _init_abundances.append(Param.fitted_relative(fitted, 1e-2))
        else:
            _init_abundances.append(Param.frozen(frozen_abundances[element]))

    return InjectionParams(
        abundances=_init_abundances,
        slopes=[
            slope if isinstance(slope, Param) else Param.fitted(slope, 0.02) for slope in slopes
        ],
        features=features or [],
    )


DEFAULT_INIT_INJECTION = init_injection(
    fitted_abundances={
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
    },
    slopes=[4.41, 4.33, 4.39],
)


class CramsModelConfig(pydantic.BaseModel):
    runner: CramsRunnerConfig

    # fitted params are set to nan in these
    frozen_propagation: Annotated[PropagationParams, pydantic.BeforeValidator(_dict_null2nan)]
    frozen_injection: Annotated[InjectionParams, pydantic.BeforeValidator(_inj_null2nan)]

    # NOTE: cubic spline needs work, now it's unpysical
    interpolation_method: Literal["numpy_linear", "scipy_linear", "scipy_cubic"] = "scipy_linear"

    population_meta: PopulationMetadata | None = None

    scale_nuclei_abundances: bool = False  # NOTE: this is useful if nuclei abundances are frozen/shared, but need to be scaled coherently

    @property
    def frozen_injection_packed(self) -> np.ndarray:
        return pack_injection(self.frozen_injection)

    @property
    def frozen_propagation_packed(self) -> np.ndarray:
        return pack_propagation(self.frozen_propagation)

    @property
    def is_fitted_propagation(self) -> np.ndarray:
        return np.isnan(self.frozen_propagation_packed)

    @property
    def is_fitted_injection(self) -> np.ndarray:
        return np.isnan(self.frozen_injection_packed)

    def has_shared_propagation(self) -> bool:
        return self.frozen_propagation == SHARED_PROPAGATION

    def has_shared_nuclei_injection(self) -> bool:
        return self.frozen_injection.abundances[2:] == SHARED_NUCLEI_INJECTION

    @staticmethod
    def make(
        to_high_E: bool,
        frozen_propagation: PropagationParams,
        frozen_injection: InjectionParams,
        population_meta: PopulationMetadata,
    ) -> "CramsModelConfig":
        if to_high_E:
            T_sim_grid = LogGrid(
                min=0.1,  # 0.1 GeV
                max=1e9,  # 100 PeV
                size=300,  # enough for numerical errors <0.1%
            )
            R_out_grid = LogGrid(
                min=5,
                max=5e8,
                size=300,
            )
        else:
            T_sim_grid = CRAMS_DEFAULT_T_SIM_GRID
            R_out_grid = CRAMS_DEFAULT_R_OUT_GRID

        return CramsModelConfig(
            runner=CramsRunnerConfig(T_sim_grid=T_sim_grid, R_out_grid=R_out_grid),
            frozen_injection=frozen_injection or DEFAULT_INIT_INJECTION,
            frozen_propagation=frozen_propagation or DEFAULT_INIT_PROPAGATION,
            population_meta=population_meta,
        )

    @property
    def elements(self) -> list[Element]:
        return CRAMS_ELEMENTS

    @property
    def all_elements(self) -> list[Element]:  # compat with CosmicRaysModelConfig
        return self.elements

    def unpack_injection(self, v: np.ndarray):
        return unpack_injection(
            v,
            n_slopes=len(self.frozen_injection.slopes),
            feature_classes=[type(ftr) for ftr in self.frozen_injection.features],
            abundances_are_precomputed_normalizations=self.frozen_injection.abundances_are_precomputed_normalizations,
        )


@dataclass
class CramsModel(Packable[CramsModelConfig]):
    propagation: PropagationParams
    injection: InjectionParams

    config: CramsModelConfig

    nuclei_abundance_scale: float = 1.0

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
        to_high_E: bool,
        init_propagation: PropagationParams | None = None,
        init_injection: InjectionParams | None = None,
        randomize_init: bool = False,
        population_meta: PopulationMetadata | None = None,
    ) -> "CramsModel":
        init_propagation = init_propagation or DEFAULT_INIT_PROPAGATION
        init_injection = init_injection or DEFAULT_INIT_INJECTION
        return CramsModel(
            propagation=sample_init(init_propagation) if randomize_init else init_propagation,
            injection=sample_init(init_injection) if randomize_init else init_injection,
            config=CramsModelConfig.make(
                to_high_E=to_high_E,
                frozen_propagation=get_frozen_scheme(init_propagation),
                frozen_injection=get_frozen_scheme(init_injection),
                population_meta=population_meta or PopulationMetadata(name="CRAMS", linestyle="-"),
            ),
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

        flags: list[bool | np.bool_] = [
            np.allclose(self.injection.abundances, other.injection.abundances),
            np.allclose(self.injection.slopes, other.injection.slopes),
            np.isclose(self.propagation.H_kpc, other.propagation.H_kpc),
            np.isclose(self.propagation.v_A_km_sec, other.propagation.v_A_km_sec),
            np.isclose(self.propagation.R_b_GV, other.propagation.R_b_GV),
            np.isclose(self.propagation.delta, other.propagation.delta),
            np.isclose(self.propagation.ddelta, other.propagation.ddelta),
            np.isclose(self.propagation.D_0_cm2_sec, other.propagation.D_0_cm2_sec),
            np.isclose(self.propagation.X_src, other.propagation.X_src),
            np.isclose(self.propagation.phi, other.propagation.phi),
            is_feature_same,
            np.isclose(self.nuclei_abundance_scale, other.nuclei_abundance_scale),
        ]

        # print(self, other)
        # print(flags)

        return bool(np.all(flags))  # type: ignore

    @property
    def crams(self) -> CramsRunner:
        return self.config.runner.implement()

    @functools.cached_property
    def _crams_lg_output(self) -> np.ndarray:
        if self.nuclei_abundance_scale:
            injection = copy.deepcopy(self.injection)
            injection.abundances = [
                # nuclei heavier than He are scaled
                (self.nuclei_abundance_scale if i > 1 else 1.0) * a
                for i, a in enumerate(injection.abundances)
            ]
        else:
            injection = self.injection
        return np.log10(self.crams.compute(self.propagation, injection))

    @functools.cached_property
    def crams_lgR_grid(self) -> np.ndarray:
        return self._crams_lg_output[:, 0]

    def clear_crams_cache(self) -> None:
        if hasattr(self, "_crams_lg_output"):
            del self._crams_lg_output
        if hasattr(self, "crams_lgR_grid"):
            del self.crams_lgR_grid

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
        self,
        Q: np.ndarray,
        element: Element,
        quantity: CharacteristicQuantity,
        contrib_to_all_particle: bool = True,  # noop, compatibility with legacy pheno signature
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
    def population_meta(self) -> PopulationMetadata | None:
        return self.config.population_meta

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
        add_labels: bool = True,
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
                    with_prefix(element.name, preserve_capitalization=True) if add_labels else None
                ),
                color=element.color,
                linestyle=self.linestyle,
            )
        if all_particle:
            ax.plot(
                E_grid,
                E_factor * self.compute_all_particle_spectrum(E_grid),
                label=with_prefix("All particle") if add_labels else None,
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
            all_inj_labels = [f"\\lg q_\\text{{{el.name}}}" for el in CRAMS_ELEMENTS]
            all_inj_labels.extend(
                [
                    f"\\gamma_\\text{{{el}}}"
                    for el, _ in zip(CRAMS_ELEMENT_NAMES, self.injection.slopes[:-1])
                ]
            )
            if len(self.injection.slopes) == len(CRAMS_ELEMENT_NAMES):
                all_inj_labels.append(f"\\gamma_\\text{{{CRAMS_ELEMENT_NAMES[-1]}}}")
            elif len(self.injection.slopes) == 1:
                all_inj_labels.append("\\gamma")
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
            all_inj_labels = [f"lg q_{el.name}" for el in CRAMS_ELEMENTS]
            all_inj_labels.extend(
                [f"gamma_{el}" for el, _ in zip(CRAMS_ELEMENT_NAMES, self.injection.slopes[:-1])]
            )
            if len(self.injection.slopes) == len(CRAMS_ELEMENT_NAMES):
                all_inj_labels.append(f"gamma_{CRAMS_ELEMENT_NAMES[-1]}")
            elif len(self.injection.slopes) == 1:
                all_inj_labels.append("gamma")
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
        if self.config.scale_nuclei_abundances:
            if latex:
                labels.append("\\lg K_\\text{nuc}")
            else:
                labels.append("lg K_nuc")

        labels = [self.population_prefix(latex) + label for label in labels]
        return labels

    def pack(self) -> np.ndarray:
        components = [
            pack_propagation(self.propagation)[self.config.is_fitted_propagation],
            pack_injection(self.injection)[self.config.is_fitted_injection],
        ]
        if self.config.scale_nuclei_abundances:
            components.append([np.log10(self.nuclei_abundance_scale)])  # type: ignore
        return np.concatenate(components)

    def bounds(self) -> Sequence[tuple[float | None, float | None]]:
        injection_bounds = [
            (
                (-8.0, np.log10(300.0))
                if self.injection.abundances_are_precomputed_normalizations
                else (-10.0, np.log10(2.0))
            )
        ] * len(self.injection.abundances) + [
            (
                (0.0, 6.0)
                if self.injection.abundances_are_precomputed_normalizations
                # (4, 5) is strictly enforced by CRAMS, here we add a small margin
                # unless we specify precomputed normalizations
                else (4.001, 4.999)
            )
        ] * len(self.injection.slopes)
        # no feature is expected below ~5 TV range (DAMPE break is at ~13 TV)
        feature_position_bounds = (np.log10(5e3), np.log10(1e7))
        for feature in self.injection.features:
            if feature is not None:
                match feature:
                    case InjectionBreak():
                        injection_bounds.extend(
                            (
                                feature_position_bounds,
                                (-1.0, 4.0),
                                (0.01, 1.0),
                            )
                        )
                    case InjectionLognormalDist():
                        injection_bounds.extend(
                            (
                                feature_position_bounds,
                                (0.1, 1.5),
                                (-5, 5),
                            )
                        )
                    case InjectionExpCutoff():
                        injection_bounds.extend(
                            (feature_position_bounds, (0.3, 3.0)),
                        )

        return (
            [
                prop_bound
                for prop_bound, is_fitted in zip(
                    PROPAGATION_BOUNDS, self.config.is_fitted_propagation
                )
                if is_fitted
            ]
            + [
                inj_bound
                for inj_bound, is_fitted in zip(injection_bounds, self.config.is_fitted_injection)
                if is_fitted
            ]
            + ([(-4, 2)] if self.config.scale_nuclei_abundances else [])
        )

    def ndim(self) -> int:
        return (
            np.sum(self.config.is_fitted_injection)
            + np.sum(self.config.is_fitted_propagation)
            + (1 if self.config.scale_nuclei_abundances else 0)
        )

    def layout_info(self) -> CramsModelConfig:
        return self.config

    @classmethod
    def unpack(cls, theta: np.ndarray, layout_info: CramsModelConfig) -> "CramsModel":
        config = layout_info
        assert not config.has_shared_propagation(), (
            "Attempted to unpack a CramsModel with shared propagation; "
            + "it is a placeholder which needs to be replaced with actual values before use"
        )
        assert not config.has_shared_nuclei_injection(), (
            "Attempted to unpack a CramsModel with shared nuclei injection; "
            + "it is a placeholder which needs to be replaced with actual values before use"
        )

        propagation_packed = config.frozen_propagation_packed
        n_prop = np.sum(config.is_fitted_propagation)
        propagation_packed[config.is_fitted_propagation] = theta[:n_prop]

        injection_packed = config.frozen_injection_packed
        n_inj = np.sum(config.is_fitted_injection)
        injection_packed[config.is_fitted_injection] = theta[n_prop : (n_prop + n_inj)]

        if config.scale_nuclei_abundances:
            nuclei_abundance_scale = 10 ** theta[n_prop + n_inj]
        else:
            nuclei_abundance_scale = 1.0

        return CramsModel(
            injection=layout_info.unpack_injection(injection_packed),
            propagation=unpack_propagation(propagation_packed),
            config=config,
            nuclei_abundance_scale=nuclei_abundance_scale,  # type: ignore
        )

    # pheno model compatibility methods

    @property
    def all_elements(self) -> list[Element]:
        return self.layout_info().all_elements
