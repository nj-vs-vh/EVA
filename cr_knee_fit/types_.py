import abc
from typing import Generic, Type, TypeVar

import numpy as np

from cr_knee_fit.elements import Element, isotope_average_A

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

    def format_params(self) -> str:
        lines: list[str] = []
        for i, (label, value) in enumerate(zip(self.labels(False), self.pack())):
            lines.append(f"{i + 1: >3}. {label: >32} = {value:.2e}")
        return "\n".join(lines)

    def print_params(self):
        print(self.format_params())
