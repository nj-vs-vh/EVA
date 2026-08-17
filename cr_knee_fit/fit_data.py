import dataclasses
import functools
import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from logging import warning
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.typing import ColorType

from cr_knee_fit import experiments
from cr_knee_fit.constants import NON_FITTED_ALPHA
from cr_knee_fit.elements import Element
from cr_knee_fit.experiments import Experiment
from cr_knee_fit.utils import (
    DATA_DIR,
    E_GEV_LABEL,
    LN_A_LABEL,
    R_GV_LABEL,
    CharacteristicQuantity,
    color_average,
    energy_shift_suffix,
    label_energy_flux,
    legend_with_added_items,
    quantity_label,
    quantity_unit,
)

DEFAULT_MARKER_SIZE = 4.0


def load_data(
    filename: str,
    x_bounds: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = str(DATA_DIR / filename)
    cols = (0, 1, 2, 3, 4, 5)
    data = np.loadtxt(path, usecols=cols)
    x = data[:, 0]
    if x_bounds is not None:
        mask = (x > x_bounds[0]) & (x < x_bounds[1])
    else:
        mask = np.ones_like(x, dtype=bool)
    return (
        x[mask],
        data[mask, 1],  # y
        data[mask, 2:4],  # err stat
        data[mask, 4:6],  # err syst
    )


@dataclass(frozen=True)
class GenericExperimentData:
    x: np.ndarray  # 1D

    y: np.ndarray

    err_stat: np.ndarray  # 2D, size: (npoints, 2); columns are (lower, upper)
    err_syst: np.ndarray

    experiment: Experiment

    custom_label: str | None = None

    def __post_init__(self) -> None:
        assert self.x.ndim == 1, "X must be 1-dimensional"
        assert self.y.ndim == 1, "Y must be 1-dimensional"
        assert self.x.size == self.size()
        assert self.y.size == self.size(), f"Bad Y size: {self.y.size} =/= {self.size()}"
        assert self.err_stat.shape == (self.size(), 2), (
            f"Bad stat error size: {self.err_stat.shape} =/= {(self.size(), 2)}"
        )
        assert self.err_syst.shape == (self.size(), 2), (
            f"Bad syst error size: {self.err_syst.shape} =/= {(self.size(), 2)}"
        )

    def with_shifted_grid(self, f: float) -> "GenericExperimentData":
        """Simple grid shift on a multiplicative factor f"""
        return dataclasses.replace(self, x=self.x * f)

    def size(self) -> int:
        return self.x.size

    def err_cov(self, corr_length: float, log_space_correlation: bool = True):
        """
        Covariance matrix of data errors. By default uses an ad hoc assumptions that statistical
        uncertainties are uncorrelated and systematic ones have correlations ~exp(-delta / corr_len).
        Delta is calculated in log10 space by default. Effectively, corr len >> x range leads
        to the global correlation of all systematics, and corr len << x step leads to uncorrelated
        systematics. Note that asymmetric errors are symmetrized to calculate the covariance matrix.
        """
        # it's reasonable to average upper and lower errors as is, not in quadratures!
        stat_symmetrized = np.sum(self.err_stat, axis=1) / 2
        stat_cov = np.diag(stat_symmetrized**2)

        x = np.log10(self.x) if log_space_correlation else self.x
        delta = np.abs(np.subtract.outer(x, x))
        syst_corr = np.exp(-delta / corr_length)
        syst_symmetrized = (np.sum(self.err_syst, axis=1) / 2).reshape((-1, 1))
        syst_cov = (syst_symmetrized @ syst_symmetrized.T) * syst_corr

        return stat_cov + syst_cov

    def total_relative_error(self) -> np.ndarray:
        err_total = np.sqrt(self.err_stat**2 + self.err_syst**2)
        mean_err_total = np.mean(err_total, axis=1)
        return mean_err_total / self.y

    def summary(self, description: str, x_quantity: str) -> str:
        e = self.total_relative_error()
        e = e[np.isfinite(e)]
        return (
            f"{self.experiment.name} {description}: {self.size()} points "
            + f"from {self.x.min():.1e} to {self.x.max():.1e} {x_quantity}; "
            + f"total relative error from {e.min():.2%} to {e.max():.2%}"
        )

    @functools.cached_property
    def log_space_err_cov_inv(self) -> np.ndarray:
        # TODO: does 1.0 correlation length make sense in all cases?
        return np.linalg.inv(self.err_cov(corr_length=1.0, log_space_correlation=True))

    @classmethod
    def load(
        cls,
        exp: Experiment,
        suffix: str,
        x_bounds: tuple[float, float] | None = None,
        custom_label: str | None = None,
    ) -> "GenericExperimentData":
        x, y, stat, syst = load_data(
            filename=f"{exp.filename_prefix}_{suffix}.txt", x_bounds=x_bounds
        )
        return GenericExperimentData(
            x=x, y=y, err_stat=stat, err_syst=syst, experiment=exp, custom_label=custom_label
        )

    def scale_factor(self, scale: float) -> np.ndarray:
        return self.x**scale

    def plot(
        self,
        ax: Axes | None = None,
        color: Any | None = None,
        scale: float = 0,
        is_fitted: bool = True,
        marker_size: float = DEFAULT_MARKER_SIZE,
        label_override: str | None = None,
        add_legend_label: bool = True,
        x_factor: float = 1.0,
    ) -> Axes:
        if ax is None:
            _, ax = plt.subplots()
        factor = self.scale_factor(scale=scale)

        if label_override is not None:
            label = label_override
        else:
            label = self.experiment.name
            if self.custom_label is not None:
                label += " " + self.custom_label
        factor_2D = np.expand_dims(factor, axis=-1)

        alpha = 1.0 if is_fitted else NON_FITTED_ALPHA
        lines = ax.errorbar(
            x_factor * self.x,
            factor * self.y,
            yerr=(factor_2D * self.err_stat).T,
            color=color,
            markersize=marker_size,
            elinewidth=0.75,
            # capsize=1.5,
            label=label if add_legend_label else None,
            linestyle="none",
            marker=self.experiment.marker,
            alpha=alpha,
        )
        ax.errorbar(
            x_factor * self.x,
            factor * self.y,
            yerr=(factor_2D * self.err_syst).T,
            color=lines[0].get_color(),
            alpha=alpha * 0.33,
            elinewidth=marker_size,  # making systematic error bar as wide as the marker
            linestyle="none",
            marker="none",
        )
        return ax


type AllparticleSpectrum = None
type ElementSum = tuple[Element, ...]
type SpectrumDataSpec = Element | ElementSum | AllparticleSpectrum


def spectrum_data_spec_label(spec: SpectrumDataSpec) -> str:
    if spec is None:
        return "all"
    elif isinstance(spec, tuple):
        return "+".join(p.name for p in spec)
    else:
        return spec.name


@dataclass(frozen=True)
class CRSpectrumData:
    d: GenericExperimentData
    spec: SpectrumDataSpec
    energy_scale_shift: float = 1.0

    # to avoid recomputing on energy scale shifts
    precomputed_err_cov: np.ndarray | None = None
    precomputed_err_cov_inv: np.ndarray | None = None

    def __post_init__(self) -> None:
        assert self.E.size > 0, "Empty spectrum data"

    @property
    def experiment(self) -> Experiment:
        return self.d.experiment

    @property
    def E(self) -> np.ndarray:
        """Energy in GeV"""
        return self.d.x

    @property
    def F(self) -> np.ndarray:
        """Flux in GeV^-1 m^-2 s^-1 sr^-1"""
        return self.d.y

    @property
    def F_err_stat(self) -> np.ndarray:
        return self.d.err_stat

    @property
    def F_err_syst(self) -> np.ndarray:
        return self.d.err_syst

    def size(self) -> int:
        return self.d.size()

    def summary(self) -> str:
        return self.d.summary(self.element_label(), "GeV")

    @functools.cached_property
    def err_cov(self) -> np.ndarray:
        if self.precomputed_err_cov is not None:
            return self.precomputed_err_cov

        # 1 decade rougly follows the use in DAMPE's paper of 4 nuisance parameter shifts per 4 decades of spectrum
        corr_length = 1.0 if self.d.experiment == experiments.dampe else 1.0  # noqa: RUF034

        return self.d.err_cov(
            corr_length=corr_length,
            log_space_correlation=True,
        )

    @functools.cached_property
    def err_cov_inv(self) -> np.ndarray:
        if self.precomputed_err_cov_inv is not None:
            return self.precomputed_err_cov_inv
        return np.linalg.inv(self.err_cov)

    def scaled_flux(self, scale: float) -> np.ndarray:
        return self.F * (self.E**scale)

    def with_shifted_energy_scale(self, f: float) -> "CRSpectrumData":
        return CRSpectrumData(
            d=GenericExperimentData(
                x=self.d.x * f,
                y=self.d.y / f,
                err_stat=self.d.err_stat / f,
                err_syst=self.d.err_syst / f,
                experiment=self.d.experiment,
            ),
            spec=self.spec,
            energy_scale_shift=self.energy_scale_shift * f,
            precomputed_err_cov=self.err_cov / (f**2),
            precomputed_err_cov_inv=self.err_cov_inv * (f**2),
        )

    @classmethod
    def load(
        cls,
        exp: Experiment,
        element: Element | ElementSum,
        R_bounds: tuple[float, float] = (0, np.inf),
    ) -> "CRSpectrumData":
        match element:
            case Element() as element:
                elements_suffix = element.name
                min_Z = max_Z = element.Z
            case tuple() as elements:
                elements_suffix = "_".join(e.name for e in sorted(elements, key=lambda el: el.Z))
                min_Z = min(e.Z for e in elements)
                max_Z = max(e.Z for e in elements)

        return CRSpectrumData(
            d=GenericExperimentData.load(
                exp=exp,
                suffix=f"{elements_suffix}_energy",
                x_bounds=(R_bounds[0] * min_Z, R_bounds[1] * max_Z),
            ),
            spec=element,
        )

    @classmethod
    def load_all_particle(
        cls, exp: Experiment, E_bounds: tuple[float, float] | None = None
    ) -> "CRSpectrumData":
        return CRSpectrumData(
            d=GenericExperimentData.load(
                exp=exp,
                suffix="all_energy",
                x_bounds=E_bounds,
            ),
            spec=None,
        )

    def element_label(self) -> str:
        return spectrum_data_spec_label(self.spec)

    def plot_label(self) -> str:
        if (self.spec, self.d.experiment) in {
            (None, experiments.dampe),
        }:
            prelim_suffix = " (prelim.)"
        else:
            prelim_suffix = ""

        return (
            f"{self.d.experiment.name} {self.element_label()}{prelim_suffix}"
            + energy_shift_suffix(self.energy_scale_shift)
        )

    def plot_color(self) -> ColorType:
        match self.spec:
            case Element():
                return self.spec.color
            case tuple():
                return color_average([el.color for el in self.spec])
            case None:
                return "black"

    def plot(
        self,
        scale: float,
        ax: Axes | None = None,
        color: Any | None = None,
        add_legend_label: bool = True,
        is_fitted: bool = True,
        marker_size: float = DEFAULT_MARKER_SIZE,
    ) -> Axes:
        axes = self.d.plot(
            ax=ax,
            color=color or self.plot_color(),
            scale=scale,
            is_fitted=is_fitted,
            marker_size=marker_size,
            label_override=self.plot_label(),
            add_legend_label=add_legend_label,
        )
        label_energy_flux(axes, scale)
        return axes


@dataclass(frozen=True)
class FluxRatio:
    num: Element
    denom: Element

    def __str__(self) -> str:
        return f"{self.num.name} / {self.denom.name}"

    def color(self) -> ColorType:
        return color_average([self.num.color, self.denom.color])


@dataclass(frozen=True)
class FluxRatioData:
    d: GenericExperimentData
    ratio: FluxRatio
    quantity: CharacteristicQuantity

    def __post_init__(self) -> None:
        assert self.Q.size > 0, "Empty flux ratio data"

    @property
    def Q(self) -> np.ndarray:
        """Rigidity in GV"""
        return self.d.x

    @property
    def value(self) -> np.ndarray:
        """Dimensionless flux ratio"""
        return self.d.y

    @property
    def ratio_err_stat(self) -> np.ndarray:
        return self.d.err_stat

    @property
    def ratio_err_syst(self) -> np.ndarray:
        return self.d.err_syst

    def size(self) -> int:
        return self.d.size()

    def summary(self) -> str:
        return self.d.summary(self.ratio_label(), quantity_unit(self.quantity))

    @classmethod
    def load(
        cls,
        exp: Experiment,
        spec: FluxRatio,
        R_bounds: tuple[float, float] = (0, np.inf),
    ) -> "FluxRatioData":
        num_name = spec.num.name if spec.num != Element.H else "p"
        denom_name = spec.denom.name

        try:
            d = GenericExperimentData.load(
                exp=exp,
                suffix=f"{num_name}_{denom_name}_ratio_rigidity",
                x_bounds=R_bounds,
            )
            quantity: CharacteristicQuantity = "R"
        except FileNotFoundError:
            d = GenericExperimentData.load(
                exp=exp,
                suffix=f"{num_name}_{denom_name}_ratio_energy_per_nucleon",
                x_bounds=R_bounds,
            )
            quantity = "E_n"

        return FluxRatioData(d=d, ratio=spec, quantity=quantity)

    def ratio_label(self) -> str:
        return str(self.ratio)

    def plot_label(self) -> str:
        return f"{self.d.experiment.name} {self.ratio_label()}"

    def plot_in_R(
        self,
        ax: Axes | None = None,
        color: Any | None = None,
        add_legend_label: bool = True,
        is_fitted: bool = True,
        marker_size: float = DEFAULT_MARKER_SIZE,
    ) -> Axes | None:
        match self.quantity:
            case "E_n":
                # 2 * E_n roughly gives rigidity, so we can put data in energy per nucleon on rigidity plot
                x_factor = 2.0
                label_suffix = " (shifted from $E_n$)"
            case "R":
                x_factor = 1.0
                label_suffix = ""
            case "E":
                warning("Can't plot E data in R")  # noqa: LOG015
                return None
        axes = self.d.plot(
            ax=ax,
            color=color or self.ratio.color(),
            is_fitted=is_fitted,
            marker_size=marker_size,
            label_override=self.plot_label() + label_suffix,
            add_legend_label=add_legend_label,
            x_factor=x_factor,
        )
        axes.set_xlabel(R_GV_LABEL)
        axes.set_ylabel("Flux ratio")
        return axes

    def plot(
        self,
        ax: Axes | None = None,
        color: Any | None = None,
        add_legend_label: bool = True,
        is_fitted: bool = True,
        marker_size: float = DEFAULT_MARKER_SIZE,
    ) -> Axes:
        axes = self.d.plot(
            ax=ax,
            color=color or self.ratio.color(),
            is_fitted=is_fitted,
            marker_size=marker_size,
            label_override=self.plot_label(),
            add_legend_label=add_legend_label,
        )
        axes.set_xlabel(quantity_label(self.quantity))
        axes.set_ylabel("Flux ratio")
        return axes


@dataclass(frozen=True)
class SpectrumDataConfig:
    experiment: Experiment
    spec: SpectrumDataSpec
    bounds: tuple[float, float] = (
        0,
        np.inf,
    )  # in R for elemental spectra, in E for all-particle and sums

    @staticmethod
    def allparticle(exp: Experiment) -> "SpectrumDataConfig":
        return SpectrumDataConfig(exp, None)


@dataclass(frozen=True)
class FluxRatioDataConfig:
    experiment: Experiment
    ratio: FluxRatio
    Q_bounds: tuple[float, float] = (0, np.inf)


@dataclass
class DataConfig:
    spectra: Sequence[SpectrumDataConfig] = dataclasses.field(default_factory=list)
    lnA: Sequence[Experiment] = dataclasses.field(default_factory=list)
    flux_ratios: Sequence[FluxRatioDataConfig] = dataclasses.field(default_factory=list)

    @property
    def experiments_spectra(self) -> list[Experiment]:
        return sorted({s.experiment for s in self.spectra})

    @property
    def elements(self) -> list[Element]:
        all = {sp.spec for sp in self.spectra if isinstance(sp.spec, Element)}
        return sorted(all)

    def excluding(self, other: "DataConfig") -> "DataConfig":
        other_spectra_specs = {(sp.experiment, sp.spec): sp for sp in other.spectra}

        return DataConfig(
            spectra=[
                sp for sp in self.spectra if (sp.experiment, sp.spec) not in other_spectra_specs
            ],
            lnA=list(set(self.lnA).difference(other.lnA)),
            flux_ratios=list(set(self.flux_ratios).difference(other.flux_ratios)),
        )

    def remove_subdominant_spectra_constrained_by_flux_ratios(self):
        flux_ordering_100GeV = (
            [1, 2, 8, 6, 26, 12, 14, 10, 7, 5, 16, 20, 13, 24]
            + [22, 18, 11, 25, 4, 19, 23, 9, 15, 17, 21]
            + [28, 27, 29, 3]
        )  # as computed by CRAMS, roughly accurate

        for fr in self.flux_ratios:
            dom, sub = fr.ratio.num, fr.ratio.denom
            if flux_ordering_100GeV.index(dom) > flux_ordering_100GeV.index(sub):
                sub, dom = dom, sub

            experiment_elements = {sp.spec for sp in self.spectra if sp.experiment == fr.experiment}
            if not (dom in experiment_elements and sub in experiment_elements):
                continue

            print(
                f"Removing {fr.experiment.name} {sub.name} as it is constrained by {fr.ratio} ratio and {dom.name} is dominant..."
            )
            self.spectra = [
                sp for sp in self.spectra if not (sp.experiment == fr.experiment and sp.spec == sub)
            ]


@dataclass
class Data:
    """Top-level container for a set of experimental data"""

    config: DataConfig

    spectra: list[CRSpectrumData] = dataclasses.field(default_factory=list)
    lnA: list[GenericExperimentData] = dataclasses.field(default_factory=list)
    flux_ratios: list[FluxRatioData] = dataclasses.field(default_factory=list)

    # backwards-compatible properties
    @property
    def element_spectra(self) -> dict[Experiment, dict[Element, CRSpectrumData]]:
        outu: dict[Experiment, dict[Element, CRSpectrumData]] = {}
        for exp, data in itertools.groupby(self.spectra, key=lambda d: d.experiment):
            outu[exp] = {d.spec: d for d in data if isinstance(d.spec, Element)}
        return outu

    @property
    def all_particle_spectra(self) -> dict[Experiment, CRSpectrumData]:
        out: dict[Experiment, CRSpectrumData] = {}
        for exp, data in itertools.groupby(self.spectra, key=lambda d: d.experiment):
            allparticle = [d for d in data if d.spec is None]
            if allparticle:
                out[exp] = allparticle[0]
        return out

    def __post_init__(self):
        self.spectra.sort(key=lambda d: d.experiment)
        self.lnA.sort(key=lambda d: d.experiment)
        self.flux_ratios.sort(key=lambda frd: frd.d.experiment)

    def experiments(self, spectra_only: bool = False) -> list[Experiment]:
        all = {sp.d.experiment for sp in self.spectra}
        if not spectra_only:
            all.update(lnA.experiment for lnA in self.lnA)
            all.update(fr.d.experiment for fr in self.flux_ratios)
        return sorted(all)

    def elements(self) -> list[Element]:
        all = {sp.spec for sp in self.spectra if isinstance(sp.spec, Element)}
        return sorted(all)

    def is_empty(self) -> bool:
        return len(self.experiments()) == 0

    def all_spectra(self) -> Iterable[CRSpectrumData]:
        yield from self.spectra  # legacy iterator

    def E_min(self) -> float:
        return min([s.E.min() for s in self.all_spectra()])

    def E_max(self) -> float:
        return max([s.E.max() for s in self.all_spectra()])

    @classmethod
    def empty(cls) -> "Data":
        return Data(
            spectra=[],
            lnA=[],
            flux_ratios=[],
            config=DataConfig(),
        )

    @classmethod
    def load(cls, config: DataConfig, verbose: bool = False) -> "Data":
        def log_loaded(exp: Experiment, param: str, error: Exception | None = None) -> None:
            if not verbose:
                return
            if error is None:
                print(f"✅ {exp.name} {param}")
            else:
                print(f"❌ {exp.name} {param}: {error}")

        spectra: list[CRSpectrumData] = []
        for sc in config.spectra:
            log_label = "<unknown>"
            spec = sc.spec
            try:
                match spec:
                    case Element() | tuple():
                        log_label = (
                            spec.name
                            if isinstance(spec, Element)
                            else "+".join([s.name for s in spec])
                        )
                        spectra.append(
                            CRSpectrumData.load(sc.experiment, element=spec, R_bounds=sc.bounds)
                        )
                    case None:
                        log_label = "all particle"
                        spectra.append(
                            CRSpectrumData.load_all_particle(sc.experiment, E_bounds=sc.bounds)
                        )
                log_loaded(sc.experiment, log_label)
            except Exception as e:  # noqa: BLE001
                log_loaded(sc.experiment, log_label, e)

        lnA: list[GenericExperimentData] = []
        for exp in config.lnA:
            try:
                lnA.append(
                    GenericExperimentData.load(
                        exp,
                        suffix="lnA_energy",
                        x_bounds=(0, np.inf),
                        custom_label=LN_A_LABEL,
                    )
                )
                log_loaded(exp, "lnA")
            except Exception as e:  # noqa: BLE001
                log_loaded(exp, "lnA", e)

        flux_ratios: list[FluxRatioData] = []
        for fr_conf in config.flux_ratios:
            try:
                flux_ratios.append(
                    FluxRatioData.load(
                        fr_conf.experiment,
                        fr_conf.ratio,
                        R_bounds=fr_conf.Q_bounds,
                    )
                )
                log_loaded(fr_conf.experiment, str(fr_conf.ratio))
            except Exception as e:  # noqa: BLE001
                log_loaded(fr_conf.experiment, str(fr_conf.ratio), e)

        return Data(spectra=spectra, lnA=lnA, flux_ratios=flux_ratios, config=config)

    def plot_spectra(
        self, scale: float, describe: bool, is_fitted: bool, ax: Axes, legend: bool = True
    ):
        print_ = print if describe else lambda _: None
        print_("Spectra:")
        for experiment, spectra in itertools.groupby(self.spectra, key=lambda d: d.experiment):
            print_(experiment.name)
            for s in spectra:
                print_("    " + s.summary())
                s.plot(scale=scale, ax=ax, add_legend_label=False, is_fitted=is_fitted)

        ax.set_xscale("log")
        ax.set_yscale("log")
        if legend:
            legend_with_added_items(
                ax,
                [
                    (exp.legend_artist(is_fitted=is_fitted), exp.name)
                    for exp in sorted(self.experiments(spectra_only=True))
                ],
                fontsize="x-small",
            )

    def plot_lnA(self, describe: bool, is_fitted: bool, ax: Axes):
        print_ = print if describe else lambda _: None
        print_("lnA data:")
        for lnA_data in self.lnA:
            print_("    " + lnA_data.summary("<lnA>", "GeV"))
            lnA_data.plot(ax=ax, is_fitted=is_fitted)
        ax.set_xscale("log")
        ax.set_xlabel(E_GEV_LABEL)
        ax.set_ylabel(LN_A_LABEL)
        ax.legend(fontsize="xx-small")

    def plot_flux_ratios(self, describe: bool, is_fitted: bool, ax: Axes):
        print_ = print if describe else lambda _: None
        print_("Flux ratio data:")
        for fr in self.flux_ratios:
            print_("    " + fr.summary())
            fr.plot_in_R(ax=ax, is_fitted=is_fitted)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(R_GV_LABEL)
        ax.set_ylabel("Flux ratio")
        ax.legend(fontsize="xx-small")

    def plot(
        self,
        scale: float,
        describe: bool = False,
        is_fitted: bool = True,
    ) -> Figure | None:
        n_subplots = bool(self.spectra) + bool(self.lnA) + bool(self.flux_ratios)
        if n_subplots == 0:
            return None

        fig, axes = plt.subplots(nrows=n_subplots, figsize=(6, 4 * n_subplots))
        if n_subplots == 1:
            axes = [axes]
        else:
            axes = cast(Sequence[Axes], axes)

        offset = 0
        if self.spectra:
            self.plot_spectra(scale=scale, describe=describe, is_fitted=is_fitted, ax=axes[offset])
            offset += 1
        if self.lnA:
            self.plot_lnA(describe=describe, is_fitted=is_fitted, ax=axes[offset])
            offset += 1
        if self.flux_ratios:
            self.plot_flux_ratios(describe=describe, is_fitted=is_fitted, ax=axes[offset])
            offset += 1

        fig.tight_layout()
        return fig


EMPTY_DATA = Data(config=DataConfig())
