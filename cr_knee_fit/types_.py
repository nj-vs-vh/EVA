import abc
import enum
from typing import Any, Generic, Type, TypeVar

import matplotlib.pyplot as plt
import numpy as np

T = TypeVar("T")
LayoutInfo = TypeVar("LayoutInfo")


class Packable(Generic[LayoutInfo], abc.ABC):
    def ndim(self) -> int:
        return self.pack().size

    @abc.abstractmethod
    def pack(self) -> np.ndarray: ...

    @abc.abstractmethod
    def labels(self, latex: bool) -> list[str]: ...

    @abc.abstractmethod
    def layout_info(self) -> LayoutInfo: ...

    @classmethod
    @abc.abstractmethod
    def unpack(cls: Type[T], theta: np.ndarray, layout_info: LayoutInfo) -> T: ...

    def validate_packing(self) -> None:
        packed = self.pack()
        labels = self.labels(latex=False)
        assert len(packed) == len(labels)
        assert len(packed) == self.ndim()
        assert self.unpack(packed, layout_info=self.layout_info()) == self


_PRIMARY_CMAP = plt.colormaps["turbo"]


class Primary(enum.IntEnum):
    H = 1
    He = 2
    C = 6
    O = 8
    Mg = 12
    Si = 14
    Fe = 26

    @property
    def Z(self) -> int:
        return self.value

    @property
    def color(self) -> Any:
        all = sorted(Primary)
        idx = all.index(self)
        return _PRIMARY_CMAP(idx / (len(all) - 1))


class Experiment(enum.StrEnum):
    AMS02 = "AMS-02"
    CALET = "CALET"
    DAMPE = "DAMPE"
    CREAM = "CREAM"

    GRAPES = "GRAPES"

    def available_primaries(self) -> list[Primary]:
        match self:
            case Experiment.AMS02:
                return list(Primary)
            case Experiment.CALET:
                return [
                    Primary.H,
                    # Primary.He,
                    Primary.C,
                    Primary.O,
                    Primary.Fe,
                ]
            case Experiment.DAMPE:
                return [Primary.H, Primary.He]
            case Experiment.CREAM:
                return list(Primary)
            case Experiment.GRAPES:
                return [Primary.H]

    def marker(self) -> str:
        match self:
            case Experiment.AMS02:
                return "o"
            case Experiment.CALET:
                return "s"
            case Experiment.DAMPE:
                return "v"
            case Experiment.CREAM:
                return "d"
            case Experiment.GRAPES:
                return "x"
