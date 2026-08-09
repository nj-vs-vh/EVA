from matplotlib import pyplot as plt

plt.rcParams.update(
    {
        # No external LaTeX
        "text.usetex": False,
        # Serif text on macOS, with safe fallbacks
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        # Make MathText look consistent with the serif text
        "mathtext.fontset": "stix",
        # Roughly document-like 11 pt sizing
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        # Optional but usually better for papers
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)
