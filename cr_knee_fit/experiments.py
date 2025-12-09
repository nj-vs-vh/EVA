import itertools
from dataclasses import dataclass

from matplotlib import lines

from cr_knee_fit.constants import NON_FITTED_ALPHA

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
        # self.marker = "$" + self.name[0].upper() + "$"

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
            alpha=1.0 if is_fitted else NON_FITTED_ALPHA,
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

hawc = Experiment("HAWC (Q)", filename_stem="HAWC_QGSJET-II-04")
grapes = Experiment("GRAPES-3", filename_stem="GRAPES")
tale = Experiment("TALE (Q)", filename_stem="TALE_QGSJET-II-04")
gamma = Experiment("GAMMA (S)", filename_stem="GAMMA_SIBYLL")
lhaaso_epos = Experiment("LHAASO (E)", filename_stem="LHAASO_EPOS-LHC")
lhaaso_sibyll = Experiment("LHAASO (S)", filename_stem="LHAASO_SIBYLL-23")
lhaaso_qgsjet = Experiment("LHAASO (Q)", filename_stem="LHAASO_QGSJET-II-04")
ice_top_sibyll = Experiment("IceTop (S)", filename_stem="ICETOP_SIBYLL_21")
kascade_sibyll = Experiment("KASCADE (S)", filename_stem="KASCADE_SIBYLL_21")
kascade_qgsjet = Experiment("KASCADE (Q)", filename_stem="KASCADE_QGSJET-II-04")
kascade_re_qgsjet = Experiment("KASCADE (2024, Q)", filename_stem="KASCADE_re_QGSJET-II-04")
kascade_grande_sibyll = Experiment("KASCADE-Grande (S)", filename_stem="KGRANDE_SIBYLL_23")

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
    kascade_qgsjet,
    kascade_re_qgsjet,
    kascade_grande_sibyll,
]

ALL = DIRECT + INDIRECT

# experiments used for ICRC25 analysis
ICRC25 = DIRECT + [hawc, grapes, lhaaso_qgsjet, kascade_re_qgsjet]
