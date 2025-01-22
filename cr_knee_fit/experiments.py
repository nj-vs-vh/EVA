from dataclasses import dataclass
import itertools

from cr_knee_fit.types_ import Primary


markers_iter = itertools.cycle(["o", "v", "^", "<", ">", "s", "p", "P", "8", "X", "D"])
markers_cache: dict[str, str] = {}


@dataclass
class Experiment:
    name: str
    filename_stem: str | None = None

    def __post_init__(self):
        marker = markers_cache.get(self.name)
        if not marker:
            marker = next(markers_iter)
            markers_cache[self.name] = marker
        self.marker = marker

    def __hash__(self):
        return hash(self.name)

    @property
    def filename_stem_(self) -> str:
        return self.filename_stem or self.name

    def primary_filename(self, primary: Primary) -> str:
        return f"{self.filename_stem_}_{primary.name}_energy.txt"

    def all_particle_filename(self) -> str:
        return f"{self.filename_stem_}_all_energy.txt"


ams02 = Experiment("AMS-02")
calet = Experiment("CALET")
dampe = Experiment("DAMPE")
cream = Experiment("CREAM")
iss_cream = Experiment("ISS-CREAM")

direct_experiments = [ams02, calet, dampe, cream, iss_cream]

hawc = Experiment("HAWC", filename_stem="HAWC_QGSJET-II-04")
grapes = Experiment("GRAPES")
tale = Experiment("TALE (QGS)", filename_stem="TALE_QGSJET-II-04")
gamma = Experiment("GAMMA (SIBYLL)", filename_stem="GAMMA_SIBYLL")

indirect_experiments = [grapes, tale, gamma]
