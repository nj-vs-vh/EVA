from dataclasses import dataclass


@dataclass
class PopulationMetadata:
    name: str
    linestyle: str | None = None

    is_apriori_energy_dominant: bool | None = None

    def plot_prefix(self, latex: bool) -> str:
        if self.name:
            return f"\\text{{{self.name}}} \\; " if latex else self.name + " "
        else:
            return ""
