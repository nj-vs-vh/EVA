import itertools
from dataclasses import dataclass

from matplotlib import lines

markers_iter = itertools.cycle(["o", "v", "^", "<", ">", "s", "p", "P", "8", "X", "D", "d"])
markers_cache: dict[str, str] = {}


@dataclass
class Experiment:
    name: str
    filename_stem: str

    def __post_init__(self) -> None:
        marker = markers_cache.get(self.name)
        if not marker:
            marker = next(markers_iter)
            markers_cache[self.name] = marker
        self.marker = marker

    def __gt__(self, other: "Experiment") -> bool:
        return self.name > other.name

    def __hash__(self):
        return hash(self.name)

    def legend_artist(self, is_fitted: bool = True):
        return lines.Line2D(
            [],  # type: ignore
            [],  # type: ignore
            color="black",
            marker=self.marker,
            linestyle="none",
            alpha=1.0 if is_fitted else 0.3,
        )

    @property
    def filename_prefix(self) -> str:
        return self.filename_stem


ams02 = Experiment("AMS-02", filename_stem="AMS-02")
calet = Experiment("CALET", filename_stem="CALET")
dampe = Experiment("DAMPE", filename_stem="DAMPE")
cream = Experiment("CREAM", filename_stem="CREAM")
iss_cream = Experiment("ISS-CREAM", filename_stem="ISS-CREAM")

DIRECT = [ams02, calet, dampe, cream, iss_cream]

hawc = Experiment("HAWC (QGSJet II-04)", filename_stem="HAWC_QGSJET-II-04")
grapes = Experiment("GRAPES-3", filename_stem="GRAPES")
tale = Experiment("TALE (QGSJet II-04)", filename_stem="TALE_QGSJET-II-04")
gamma = Experiment("GAMMA (Sibyll)", filename_stem="GAMMA_SIBYLL")
lhaaso_epos = Experiment("LHAASO (EPOS)", filename_stem="LHAASO_EPOS-LHC")
lhaaso_sibyll = Experiment("LHAASO (Sibyll 2.3)", filename_stem="LHAASO_SIBYLL-23")
lhaaso_qgsjet = Experiment("LHAASO (QGSJET II-04)", filename_stem="LHAASO_QGSJET-II-04")
ice_top_sibyll = Experiment("IceTop (Sibyll 2.1)", filename_stem="ICETOP_SIBYLL_21")
kascade_sibyll = Experiment("KASCADE (Sibyll 2.1)", filename_stem="KASCADE_SIBYLL_21")
kascade_re_qgsjet = Experiment(
    "KASCADE re-analysis (QGSJet II-04)", filename_stem="KASCADE_re_QGSJET-II-04"
)
kascade_grande_sibyll = Experiment("KASCADE-Grande (Sibyll 2.3)", filename_stem="KGRANDE_SIBYLL_23")

INDIRECT = [
    hawc,
    grapes,
    tale,
    gamma,
    lhaaso_epos,
    lhaaso_sibyll,
    lhaaso_qgsjet,
    ice_top_sibyll,
    kascade_sibyll,
    kascade_re_qgsjet,
    kascade_grande_sibyll,
]

ALL = DIRECT + INDIRECT
