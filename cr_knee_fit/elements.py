import enum
from typing import Any

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.typing import ColorType


class Element(enum.IntEnum):
    H = 1
    He = 2
    Li = 3
    Be = 4
    B = 5
    C = 6
    N = 7
    O = 8
    F = 9
    Ne = 10
    Na = 11
    Mg = 12
    Al = 13
    Si = 14
    P = 15
    S = 16
    Cl = 17
    Ar = 18
    K = 19
    Ca = 20
    Sc = 21
    Ti = 22
    V = 23
    Cr = 24
    Mn = 25
    Fe = 26
    Co = 27
    Ni = 28
    Cu = 29

    FreeZ = -1

    @classmethod
    def special(cls) -> "list[Element]":
        return [Element.FreeZ]

    @classmethod
    def regular(cls) -> "list[Element]":
        return sorted([p for p in Element if p not in cls.special()])

    @classmethod
    def nuclei(cls) -> "list[Element]":
        return [e for e in Element.regular() if e not in {Element.H, Element.He}]

    @property
    def Z(self) -> float:
        if self is Element.FreeZ:
            raise ValueError(
                "Z for Element.FreeZ must be introduced as a free parameter in the model"
            )
        return self.value

    @property
    def A(self) -> float:
        return isotope_average_A(round(self.Z))

    @property
    def lnA(self) -> float:
        return np.log(self.A)

    @property
    def lnA_normalized(self) -> float:
        LN_A_MIN = Element.H.lnA
        LN_A_MAX = Element.Cu.lnA
        return (self.lnA - LN_A_MIN) / (LN_A_MAX - LN_A_MIN)

    @property
    def color_custom(self) -> ColorType | None:
        return {
            Element.H: "#D52B2B",
            Element.He: "#C2C23F",
            Element.Be: "#317E30",
            Element.C: "#35cfa9",
            Element.O: "#3a889d",
            Element.Mg: "#398be9",
            Element.Si: "#31259f",
            Element.Fe: "#7f00ff",
            Element.Cu: "#E33EB7",
            Element.FreeZ: "gray",
        }.get(self)

    @property
    def color(self) -> ColorType:
        return ELEMENTS_CMAP(self.lnA_normalized)

    def __truediv__(self, other: Any):
        if not isinstance(other, Element):
            return NotImplemented
        from cr_knee_fit.fit_data import FluxRatio

        return FluxRatio(self, other)


element_name_to_Z_A = {
    Element.H: 1.000019399047775,
    Element.He: 3.999834043764567,
    Element.Li: 6.9241031188029565,
    Element.Be: 9.0,
    Element.B: 10.801789838337182,
    Element.C: 12.011077178638928,
    Element.N: 14.003662808317134,
    Element.O: 16.004372580664562,
    Element.F: 19.0,
    Element.Ne: 20.13894018679578,
    Element.Na: 23.0,
    Element.Mg: 24.320196078431373,
    Element.Al: 27.0,
    Element.Si: 28.10857,
    Element.P: 31.0,
    Element.S: 32.092486276766486,
    Element.Cl: 35.48462860416269,
    Element.Ar: 36.30858536585366,
    Element.K: 39.13473464157374,
    Element.Ca: 40.11566174225073,
    Element.Sc: 45.0,
    Element.Ti: 47.918662262592896,
    Element.V: 50.997503467406375,
    Element.Cr: 52.055365474339034,
    Element.Mn: 55.0,
    Element.Fe: 55.909928400954655,
    Element.Co: 59.0,
    Element.Ni: 58.75944096909848,
}


def isotope_average_A(Z: int) -> float:
    Z_clamped = min(28, max(1, Z))
    element = Element(Z_clamped)
    return element_name_to_Z_A[element]


ELEMENTS_CMAP = LinearSegmentedColormap.from_list(
    "elements",
    [
        (el.lnA_normalized, el.color_custom)
        for el in Element.regular()
        if el.color_custom is not None
    ],
)
