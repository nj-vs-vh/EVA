import enum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_FALLBACK_CMAP = plt.colormaps["rainbow_r"]


class Element(enum.IntEnum):
    H = 1
    He = 2
    C = 6
    O = 8
    Mg = 12
    Si = 14
    # Ti = 22
    # Cr = 24
    Fe = 26

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
    def color(self) -> Any:
        if color := {
            Element.H: "#ff0000",
            Element.He: "#8F9827",
            Element.C: "#2edaaf",
            Element.O: "#3a889d",
            Element.Mg: "#398be9",
            Element.Si: "#31259f",
            Element.Fe: "#7f00ff",
            Element.FreeZ: "gray",
        }.get(self):
            return color
        else:
            # idx = sorted(Element.regular()).index(self)
            # return _ELEMENT_CMAP(idx / (len(Element.regular()) - 1))
            return _FALLBACK_CMAP(self.lnA / self.Fe.lnA)  # type: ignore


element_names = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
]
Z_to_element_name = {idx + 1: name for idx, name in enumerate(element_names)}
element_name_to_Z_A = {
    "H": (1, 1.000019399047775),
    "He": (2, 3.999834043764567),
    "Li": (3, 6.9241031188029565),
    "Be": (4, 9.0),
    "B": (5, 10.801789838337182),
    "C": (6, 12.011077178638928),
    "N": (7, 14.003662808317134),
    "O": (8, 16.004372580664562),
    "F": (9, 19.0),
    "Ne": (10, 20.13894018679578),
    "Na": (11, 23.0),
    "Mg": (12, 24.320196078431373),
    "Al": (13, 27.0),
    "Si": (14, 28.10857),
    "P": (15, 31.0),
    "S": (16, 32.092486276766486),
    "Cl": (17, 35.48462860416269),
    "Ar": (18, 36.30858536585366),
    "K": (19, 39.13473464157374),
    "Ca": (20, 40.11566174225073),
    "Sc": (21, 45.0),
    "Ti": (22, 47.918662262592896),
    "V": (23, 50.997503467406375),
    "Cr": (24, 52.055365474339034),
    "Mn": (25, 55.0),
    "Fe": (26, 55.909928400954655),
    "Co": (27, 59.0),
    "Ni": (28, 58.75944096909848),
}


def isotope_average_A(Z: int) -> float:
    Z_clamped = min(28, max(1, Z))
    element = Z_to_element_name[Z_clamped]
    return element_name_to_Z_A[element][1]


# element -> (spectrum normalization at R = 1 TV, power law spectral index)
low_energy_CR_spectra = {
    "H": (7.68049907475221e-05, 2.7780262371892057),
    "He": (2.1694018490738908e-05, 2.685178909671116),
    "Li": (2.7438617719929544e-08, 3.056975412242948),
    "Be": (1.534758103968766e-08, 3.020504875499327),
    "B": (4.048166803672866e-08, 3.033101182101726),
    "C": (6.519811112758188e-07, 2.69033528684777),
    "N": (8.586273486965895e-08, 2.8415453248189646),
    "O": (6.848995794448408e-07, 2.6792700607758615),
    "F": (3.829286411671765e-09, 2.946425241920921),
    "Ne": (1.278287067401656e-07, 2.63590538561923),
    "Na": (9.791216005229198e-09, 2.8375274443126446),
    "Mg": (1.6136065344805168e-07, 2.6305057692716756),
    "Al": (1.7650128638643698e-08, 2.740453808656748),
    "Si": (1.3644773296144594e-07, 2.6252851294491877),
    "P": (2.140995680619696e-09, 2.772129488435299),
    "S": (2.5884695480693985e-08, 2.6250760373070046),
    "Cl": (1.2913579846106572e-09, 2.9053485944514996),
    "Ar": (1.6583806004537737e-08, 2.387656299189939),
    "K": (3.1745896419273668e-09, 2.707988107646582),
    "Ca": (1.338711328284328e-08, 2.5918610812017557),
    "Sc": (1.6479186991619496e-09, 2.701446218365392),
    "Ti": (7.405699213999413e-09, 2.622745948020359),
    "V": (4.030988748331536e-09, 2.623095342910532),
    "Cr": (1.073651173624369e-08, 2.556173669897529),
    "Mn": (8.772548336172527e-09, 2.5497896559500903),
    "Fe": (1.2915738955314627e-07, 2.5334684220361394),
    "Co": (4.197296567776818e-10, 2.5874554358732684),
    "Ni": (9.265969844291408e-09, 2.4373626150852807),
}
resolved_elements = [el.name for el in Element.regular()]
unresolved_element_names = [
    el
    for el in element_names
    if el in low_energy_CR_spectra and el not in resolved_elements
    # and element_name_to_Z_A[el][0] < 27
]

_total = sum(low_energy_CR_spectra[el][0] for el in unresolved_element_names)
unresolved_element_normalized_abundances_at_1TV = {
    el: low_energy_CR_spectra[el][0] / _total for el in unresolved_element_names
}
