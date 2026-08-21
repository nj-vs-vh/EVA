import abc
import datetime
import functools
import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import Self, TypeVar

import numpy as np
from scipy import stats  # type: ignore

LayoutInfo = TypeVar("LayoutInfo")


class Packable[LayoutInfo](abc.ABC):
    def ndim(self) -> int:
        return self.pack().size

    @functools.cached_property
    def size(self) -> int:
        """Cached property version of ndim"""
        return self.ndim()

    @abc.abstractmethod
    def pack(self) -> np.ndarray: ...

    @property
    def children(self) -> "list[Packable | None]":
        return []

    def pack_children(self) -> np.ndarray:
        chunks = [child.pack() for child in self.children if child is not None]
        return np.hstack(chunks)

    def children_bounds(self) -> list[tuple[float, float] | None]:
        bounds = [child.bounds() for child in self.children if child is not None]
        return list(itertools.chain.from_iterable(bounds))

    def children_labels(self, latex: bool) -> list[str]:
        labels = [child.labels(latex) for child in self.children if child is not None]
        return list(itertools.chain.from_iterable(labels))

    @abc.abstractmethod
    def labels(self, latex: bool) -> list[str]: ...

    @abc.abstractmethod
    def layout_info(self) -> LayoutInfo: ...

    @classmethod
    @abc.abstractmethod
    def unpack(cls: type[Self], theta: np.ndarray, layout_info: LayoutInfo) -> Self: ...

    def bounds(self) -> Sequence[tuple[float, float] | None]:
        """Parameter bounds to be used during inference. By default, no bounds are specified."""
        return [None] * self.size

    def validate_packing(self, quiet: bool = False) -> None:
        packed = self.pack()
        assert len(packed) == self.size

        for pad_at_the_end in (0, 1, 5):
            packed_ = packed.copy()
            packed_ = np.concatenate((packed_, np.zeros((pad_at_the_end,))))
            unpacked = self.unpack(packed_, layout_info=self.layout_info())
            if unpacked != self:
                if not quiet:
                    print("Packing validation failed")
                    print("Source:")
                    self.print_params()
                    print("\nUnpacked:")
                    unpacked.print_params()
                raise RuntimeError("Packing validation failed")

        if len(self.bounds()) != self.size:
            print(f"Packing validation failed, wrong bounds size; expected {self.size}, found:")
            print(self.bounds())

        for latex in (False, True):
            assert len(self.labels(latex)) == self.size

    def format_param_lines(self) -> list[str]:
        labels = self.labels(latex=False)
        if not labels:
            return []
        longest_label = max(len(lbl) for lbl in labels)
        lines: list[str] = []
        for i, (label, value) in enumerate(zip(labels, self.pack())):
            lines.append(f"{i + 1: >3}. {label: >{longest_label + 1}} = {value:.3g}")
        return lines

    def format_params(self) -> str:
        lines = self.format_param_lines()
        return "\n".join(lines) if lines else "<no params>"

    def print_params(self):
        print(self.format_params())

    def set_errcov(self, errcov: np.ndarray) -> None:
        assert errcov.shape == (self.size, self.size), (
            f"Errcov matrix set with incorrect shape, expected {self.size}x{self.size}, found {errcov.shape}"
        )
        self._errcov = errcov

    @property
    def errcov(self) -> np.ndarray | None:
        try:
            return self._errcov
        except AttributeError:
            return None

    def sample_errcov(self) -> np.ndarray | None:
        cov = self.errcov
        if cov is None:
            return None
        return stats.multivariate_normal.rvs(mean=self.pack(), cov=cov)  # type: ignore

    def save(self, path: Path, header: list[str] | None = None) -> None:
        dumped_on = f"Dumped on: {datetime.datetime.now()}"  # noqa: DTZ005
        np.savetxt(
            path,
            self.pack(),
            header="\n".join(
                [
                    dumped_on,
                    f"Layout info: {self.layout_info()}",
                    *(header or []),
                    "Human readable:",
                    *[ln for ln in self.format_param_lines()],
                ]
            ),
        )
        if self.errcov is not None:
            np.savetxt(
                _errcov_path(path),
                self.errcov,
                header="\n".join(
                    [
                        dumped_on,
                        f"Error covariance for the model stored in {path.resolve()}",
                    ]
                ),
            )

    @classmethod
    def load(cls: type[Self], path: Path, layout_info: LayoutInfo) -> Self | None:
        try:
            theta = np.loadtxt(path)
            res = cls.unpack(theta, layout_info=layout_info)
            errcov_path = _errcov_path(path)
            if errcov_path.exists():
                res.set_errcov(np.loadtxt(errcov_path))
            return res
        except FileNotFoundError:
            return None


def _errcov_path(path: Path) -> Path:
    return path.with_stem(path.stem + ".errcov")
