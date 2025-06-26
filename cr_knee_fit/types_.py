import abc
import datetime
from pathlib import Path
from typing import Generic, Self, Type, TypeVar

import numpy as np

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
    def unpack(cls: Type[Self], theta: np.ndarray, layout_info: LayoutInfo) -> Self: ...

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

    def save(self, path: Path, header: list[str] | None = None) -> None:
        np.savetxt(
            path,
            self.pack(),
            header="\n".join(
                [
                    f"Dumped on: {datetime.datetime.now()}",
                    f"Layout info: {self.layout_info()}",
                    *(header or []),
                ]
            ),
        )

    @classmethod
    def load(cls: Type[Self], path: Path, layout_info: LayoutInfo) -> Self | None:
        try:
            theta = np.loadtxt(path)
            return cls.unpack(theta, layout_info=layout_info)
        except FileNotFoundError:
            return None
