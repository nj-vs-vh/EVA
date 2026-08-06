import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker

from cr_knee_fit.analysis import FitConfig
from cr_knee_fit.fit_data import CRSpectrumData, Data
from cr_knee_fit.model import Model
from cr_knee_fit.utils import (
    LegendItem,
    add_log_margin,
    clamp_log_margin,
    legend_artist_line,
    legend_with_added_items,
    merged_lims,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--plot", default="output.png", type=Path)
    parser.add_argument("--export", default="export.txt", type=Path)
    parser.add_argument("--lgEmin", type=float)
    parser.add_argument("--lgEmax", type=float)
    parser.add_argument("--ngrid", type=int, default=100)
    args = parser.parse_args()

    dir: Path = args.dir

    fc = FitConfig.model_validate_json((dir / "config-dump.json").read_text())
    theta_sample = np.loadtxt(dir / "theta.txt")
    theta_best_fit = np.loadtxt(dir / "posterior-ml.txt")

    best_fit_model = Model.unpack(theta_best_fit, layout_info=fc.model)
    fit_data = Data.load(fc.fit_data_config)
    validation_data = (
        Data.load(fc.plots.validation_data_config)
        if fc.plots.validation_data_config is not None
        else Data.empty()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    spectra_scale = 2.6

    element_legend_items: list[LegendItem] = []
    experiment_legend_item_by_label: dict[str, LegendItem] = {}
    plotted_elem_spectra: list[CRSpectrumData] = []
    for data, is_fitted in ((fit_data, True), (validation_data, False)):
        for exp, data_by_particle in data.element_spectra.items():
            f_exp = best_fit_model.energy_shifts.f(exp)
            for _, spec_data in data_by_particle.items():
                spec_data = spec_data.with_shifted_energy_scale(f=f_exp)
                plotted_elem_spectra.append(spec_data)
                spec_data.plot(
                    scale=spectra_scale, ax=ax, add_legend_label=False, is_fitted=is_fitted
                )
            experiment_legend_item_by_label.setdefault(
                exp.name, (exp.legend_artist(True), exp.name)
            )
    elements = best_fit_model.layout_info().elements(only_fixed_Z=False)
    comp_data_ylim = merged_lims([sp.scaled_flux(spectra_scale) for sp in plotted_elem_spectra])
    comp_data_Elim = merged_lims([sp.E for sp in plotted_elem_spectra])
    data_Elim = add_log_margin(*comp_data_Elim)

    Elim = (
        10 ** (args.lgEmin or np.log10(data_Elim[0])),
        10 ** (args.lgEmax or np.log10(data_Elim[1])),
    )

    E_grid = np.geomspace(*Elim, args.ngrid)
    E_factor = E_grid**spectra_scale

    export_cols = [E_grid]
    export_labels = ["E"]

    for element in elements:
        flux_bestfit = best_fit_model.compute_spectrum(E_grid, element=element)
        ax.plot(E_grid, E_factor * flux_bestfit, color=element.color)

        model_sample = [Model.unpack(theta, layout_info=fc.model) for theta in theta_sample]
        observable_sample = np.vstack(
            [model.compute_spectrum(E_grid, element=element) for model in model_sample]
        )
        cl = 0.9
        quantile = (1 - cl) / 2
        flux_lower = np.quantile(observable_sample, q=quantile, axis=0)
        flux_upper = np.quantile(observable_sample, q=1 - quantile, axis=0)

        export_cols.extend((flux_bestfit, flux_lower, flux_upper))
        export_labels.extend(
            [
                f"{element.name} {value}"
                for value in (
                    "best fit",
                    f"lower ({100 * quantile:.0f}%)",
                    f"upper ({100 * (1 - quantile):.0f}%)",
                )
            ]
        )

        ax.fill_between(
            E_grid,
            E_factor * flux_lower,
            E_factor * flux_upper,
            color=element.color,
            alpha=0.2,
            edgecolor="none",
        )

        element_legend_items.append((legend_artist_line(element.color), element.name))

    clamp_log_margin(ax, comp_data_ylim, 0.1)
    ax.set_xlim(*Elim)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())  # never caption minor ticks!

    legend_items = element_legend_items.copy()
    legend_items += list(experiment_legend_item_by_label.values())
    legend_with_added_items(
        ax,
        legend_items,
        fontsize="small",
        bbox_to_anchor=(0.00, 1.05, 1.0, 0.0),
        loc="lower left",
        fancybox=True,
        shadow=True,
        ncol=7,
    )

    fig.tight_layout()
    fig.savefig(args.plot)

    np.savetxt(
        fname=args.export,
        X=np.array(export_cols).T,
        header=(
            "Energies in GeV, fluxes in (GeV m^2 s sr)^-1\n"
            + " ".join(f"{label: ^24}" for label in export_labels)
        ),
    )
