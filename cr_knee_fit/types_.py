import abc
import enum
from typing import Annotated, Any, Generic, Type, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pydantic

T = TypeVar("T")
LayoutInfo = TypeVar("LayoutInfo")


class Packable(Generic[LayoutInfo], abc.ABC):
    def ndim(self) -> int:
        return self.pack().size

    @abc.abstractmethod
    def pack(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def labels(self, latex: bool) -> list[str]:
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
        labels = self.labels(latex=False)
        assert len(packed) == len(labels)
        assert len(packed) == self.ndim()
        assert self.unpack(packed, layout_info=self.layout_info()) == self

    def format_params(self) -> str:
        lines: list[str] = []
        for i, (label, value) in enumerate(zip(self.labels(False), self.pack())):
            lines.append(f"{i + 1: >3}. {label: >32} = {value:.2e}")
        return "\n".join(lines)

    def print_params(self):
        print(self.format_params())


_PRIMARY_CMAP = plt.colormaps["turbo"]


class Primary(enum.IntEnum):
    H = 1
    He = 2
    C = 6
    O = 8
    Mg = 12
    Si = 14
    Fe = 26

    Unobserved = 1000

    @classmethod
    def all(cls) -> "list[Primary]":
        return sorted([p for p in Primary if p is not Primary.Unobserved])

    @property
    def Z(self) -> float:
        if self is Primary.Unobserved:
            raise ValueError("Unobserved primary Z must be introduced as a free parameter")
        return self.value

    @property
    def A(self) -> int:
        return most_abundant_stable_izotope_A(round(self.Z))

    @property
    def color(self) -> Any:
        if self is Primary.Unobserved:
            return "gray"
        else:
            idx = sorted(Primary).index(self)
            return _PRIMARY_CMAP(idx / (len(Primary) - 1))


def most_abundant_stable_izotope_A(Z: int) -> int:
    Z_clamped = Z
    if Z < 1:
        Z_clamped = 1
    elif Z > 26:
        Z_clamped = 26
    return {
        1: 1,
        2: 4,
        3: 7,
        4: 9,
        5: 11,
        6: 12,
        7: 14,
        8: 16,
        9: 19,
        10: 20,
        11: 23,
        12: 24,
        13: 27,
        14: 28,
        15: 31,
        16: 32,
        17: 35,
        18: 40,
        19: 39,
        20: 40,
        21: 45,
        22: 48,
        23: 51,
        24: 52,
        25: 55,
        26: 56,
    }[Z_clamped]
