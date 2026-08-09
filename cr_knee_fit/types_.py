import abc
import datetime
from pathlib import Path
from typing import Self, TypeVar

import numpy as np

LayoutInfo = TypeVar("LayoutInfo")


class Packable[LayoutInfo](abc.ABC):
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
    def unpack(cls: type[Self], theta: np.ndarray, layout_info: LayoutInfo) -> Self: ...

    def ml_bounds(self) -> list[tuple[float, float] | None] | None:
        """Bounds to be used during optimization. By default, no bounds are specified."""
        return None

    def validate_packing(self) -> None:
        packed = self.pack()
        assert len(packed) == self.ndim()

        for pad_at_the_end in (0, 1, 5):
            packed_ = packed.copy()
            packed_ = np.concatenate((packed_, np.zeros((pad_at_the_end,))))
            assert self.unpack(packed_, layout_info=self.layout_info()) == self

        if bounds := self.ml_bounds():
            assert len(bounds) == self.ndim()

        assert len(self.labels(latex=False)) == self.ndim()

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
                    f"Dumped on: {datetime.datetime.now()}",  # noqa: DTZ005
                    f"Layout info: {self.layout_info()}",
                    *(header or []),
                ]
            ),
        )

    @classmethod
    def load(cls: type[Self], path: Path, layout_info: LayoutInfo) -> Self | None:
        try:
            theta = np.loadtxt(path)
            return cls.unpack(theta, layout_info=layout_info)
        except FileNotFoundError:
            return None
