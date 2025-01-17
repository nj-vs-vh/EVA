from matplotlib.axes import Axes


def label_energy_flux(ax: Axes, scale: float) -> None:
    ax.set_xlabel("$E$ / $\\text{GeV}$")
    ax.set_ylabel(
        f"$ E^{{{scale}}} F $ / $ \\text{{GeV}}^{{{scale - 1:.2g}}} \\; \\text{{m}}^{{-2}} \\; \\text{{s}}^{{-1}} \\; \\text{{sr}}^{{-1}} $"
    )
