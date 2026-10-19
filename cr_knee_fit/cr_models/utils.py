import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

FREE = np.nan  # float used in the frozen scheme to indicate free parameters


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


class Param(float):
    @staticmethod
    def frozen(value: float) -> "Param":
        if isinstance(value, Param):
            value.freeze()
            return value

        v = Param(value)
        v._param_is_frozen = True
        return v

    @staticmethod
    def fitted(value: float, init_std: float) -> "Param":
        return Param.gaussian(value, init_std)

    @staticmethod
    def gaussian(value: float, init_std: float) -> "Param":
        if isinstance(value, Param):
            v = value
        else:
            v = Param(value)
        v.unfreeze()
        v._param_init_std = init_std  # type: ignore
        return v

    def freeze(self) -> None:
        self._param_is_frozen = True
        self._param_init_std = None

    def unfreeze(self) -> None:
        self._param_is_frozen = False
        self._param_init_std = 0.0  # type: ignore

    @property
    def is_frozen(self) -> bool:
        return getattr(self, "_param_is_frozen", False)  # type: ignore

    def sample(self) -> float:
        std = getattr(self, "_param_init_std", None)
        if std is None:
            return float(self)
        return float(np.random.normal(loc=self, scale=std))


def get_frozen_scheme[T: Any](data: T) -> T:
    match data:
        case Param():
            return data if data.is_frozen else FREE  # type: ignore
        case struct if dataclasses.is_dataclass(struct):
            return dataclasses.replace(
                struct,  # type: ignore
                **{
                    f.name: get_frozen_scheme(getattr(struct, f.name))
                    for f in dataclasses.fields(struct)
                },
            )
        case list() | tuple():
            return type(data)(get_frozen_scheme(x) for x in data)  # type: ignore
        case dict():
            return {k: get_frozen_scheme(v) for k, v in data.items()}  # type: ignore
        case _:
            return data


def sample_init[T: Any](data: T) -> T:
    match data:
        case Param():
            return data.sample()  # type: ignore
        case struct if dataclasses.is_dataclass(struct):
            return dataclasses.replace(
                struct,  # type: ignore
                **{
                    f.name: sample_init(getattr(struct, f.name)) for f in dataclasses.fields(struct)
                },
            )
        case list() | tuple():
            return type(data)(sample_init(x) for x in data)  # type: ignore
        case dict():
            return {k: sample_init(v) for k, v in data.items()}  # type: ignore
        case _:
            return data
