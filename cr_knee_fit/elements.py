import enum
from typing import Any

import crdb  # type: ignore
import crdb.experimental  # type: ignore
import matplotlib.pyplot as plt
from matplotlib import lines

_ELEMENT_CMAP = plt.colormaps["rainbow_r"]


class Element(enum.IntEnum):
    H = 1
    He = 2
    C = 6
    O = 8
    Mg = 12
    Si = 14
    Fe = 26

    FreeZ = -1

    @classmethod
    def special(cls) -> "list[Element]":
        return [Element.FreeZ]

    @classmethod
    def regular(cls) -> "list[Element]":
        return sorted([p for p in Element if p not in cls.special()])

    @property
    def Z(self) -> float:
        if self is Element.FreeZ:
            raise ValueError(
                "Z for Element.FreeZ must be introduced as a free parameter in the model"
            )
        return self.value

    @property
    def A(self) -> float:
        return izotope_average_A(round(self.Z))

    @property
    def color(self) -> Any:
        if self is Element.FreeZ:
            return "gray"
        else:
            idx = sorted(Element.regular()).index(self)
            return _ELEMENT_CMAP(idx / (len(Element.regular()) - 1))

    def legend_artist(self):
        return lines.Line2D([], [], color=self.color, marker="none")


Z_to_element_name = {Z: name for name, Z in crdb.ELEMENTS.items()}
name_to_Z_A = crdb.experimental.energy_conversion_numbers()


def izotope_average_A(Z: int) -> float:
    Z_clamped = min(28, max(1, Z))
    element = Z_to_element_name[Z_clamped]
    return name_to_Z_A[element][1]
