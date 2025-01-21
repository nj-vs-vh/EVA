from matplotlib.axes import Axes


def add_log_margin(min: float, max: float, log_margin: float = 0.1) -> tuple[float, float]:
    frac = max / min
    margin = frac**log_margin
    return min / margin, max * margin


def label_energy_flux(ax: Axes, scale: float) -> None:
    ax.set_xlabel("$E$ / $\\text{GeV}$")
    ax.set_ylabel(
        f"$ E^{{{scale}}} F $ / $ \\text{{GeV}}^{{{scale - 1:.2g}}} \\; \\text{{m}}^{{-2}} \\; \\text{{s}}^{{-1}} \\; \\text{{sr}}^{{-1}} $"
    )
