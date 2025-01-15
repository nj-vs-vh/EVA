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
    def pack(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def labels(self) -> list[str]:
        ...

    @abc.abstractmethod
    def layout_info(self) -> LayoutInfo:
        ...

    @classmethod
    @abc.abstractmethod
    def unpack(cls: Type[T], theta: np.ndarray, layout_info: LayoutInfo) -> T:
        ...

    def validate_packing(self) -> None:
        packed = self.pack()
        labels = self.labels()
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
