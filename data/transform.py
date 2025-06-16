import logging
import re
from pathlib import Path
from typing import Sequence

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def geom_mean(min_values, max_values):
    """Compute the mean energy as the geometric mean of min and max values."""
    return np.sqrt(min_values * max_values)


def readfile(filepath):
    """Read data from a given file and return extracted columns."""
    try:
        logging.info("Reading data from %s", filepath)
        x_min, x_max, y, y_sta_lo, y_sta_up, y_sys_lo, y_sys_up = np.loadtxt(
            filepath, usecols=(0, 1, 2, 3, 4, 5, 6), unpack=True
        )
        logging.info("Successfully read data from %s", filepath)
        return x_min, x_max, y, y_sta_lo, y_sta_up, y_sys_lo, y_sys_up
    except Exception as e:
        logging.error("Failed to read data from %s: %s", filepath, e)
        raise


def transform_R2E(R_min, R_max, I_R, e_sta_lo, e_sta_up, e_sys_lo, e_sys_up, Z=1.0):
    """Convert rigidity data to energy data."""
    R_mean = geom_mean(R_min, R_max)
    E_mean = R_mean * Z
    I_E = I_R / Z
    return [E_mean, I_E, e_sta_lo / Z, e_sta_up / Z, e_sys_lo / Z, e_sys_up / Z]


def transform_T2E(T_min, T_max, I_T, e_sta_lo, e_sta_up, e_sys_lo, e_sys_up, A=1.0):
    """Convert kinetic energy per nucleon data to energy data."""
    T_mean = geom_mean(T_min, T_max)
    E_mean = T_mean * A
    I_E = I_T / A
    return [E_mean, I_E, e_sta_lo / A, e_sta_up / A, e_sys_lo / A, e_sys_up / A]


def dump(data: np.ndarray | Sequence[np.ndarray], filename: str) -> None:
    """Write transformed data to a file in the output directory."""
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    logging.info("Dumping data to %s", filepath)
    try:
        E_mean, I_E, I_sta_lo, I_sta_up, I_sys_lo, I_sys_up = data
        with filepath.open("w") as f:
            for i, j, k, l, m, n in zip(E_mean, I_E, I_sta_lo, I_sta_up, I_sys_lo, I_sys_up):
                f.write(f"{i:5.3e} {j:5.3e} {k:5.3e} {l:5.3e} {m:5.3e} {n:5.3e}\n")
        logging.info("Data successfully written to %s", filepath)
    except Exception as e:
        logging.error("Failed to write data to %s: %s", filepath, e)
        raise


def transform_crdb_generic(filename: str) -> None:
    x_min, x_max, value, e_sta_lo, e_sta_up, e_sys_lo, e_sys_up = readfile(f"crdb/{filename}")
    dump([geom_mean(x_min, x_max), value, e_sta_lo, e_sta_up, e_sys_lo, e_sys_up], filename)


def transform_AMS02():
    """Transform and dump AMS02 data."""
    datasets = [
        ("PAMELA_H_rigidity.txt", "PAMELA_H_energy.txt", 1),
        ("AMS-02_H_rigidity.txt", "AMS-02_H_energy.txt", 1),
        ("AMS-02_He_rigidity.txt", "AMS-02_He_energy.txt", 2),
        ("PAMELA_He_rigidity.txt", "PAMELA_He_energy.txt", 2),
        ("AMS-02_C_rigidity.txt", "AMS-02_C_energy.txt", 6),
        ("AMS-02_O_rigidity.txt", "AMS-02_O_energy.txt", 8),
        ("AMS-02_Mg_rigidity.txt", "AMS-02_Mg_energy.txt", 12),
        ("AMS-02_Si_rigidity.txt", "AMS-02_Si_energy.txt", 14),
        ("AMS-02_Fe_rigidity.txt", "AMS-02_Fe_energy.txt", 26),
    ]

    for rigidity_file, energy_file, Z in datasets:
        R_min, R_max, I_R, e_sta_lo, e_sta_up, e_sys_lo, e_sys_up = readfile(
            f"crdb/{rigidity_file}"
        )
        data = transform_R2E(R_min, R_max, I_R, e_sta_lo, e_sta_up, e_sys_lo, e_sys_up, Z=Z)
        dump(data, energy_file)

    transform_crdb_generic("AMS-02_p_He_ratio_rigidity.txt")


def transform_CALET():
    """Transform and dump CALET data."""
    datasets = [
        ("CALET_H_kEnergy.txt", "CALET_H_energy.txt", 1),
        ("CALET_He_kEnergy.txt", "CALET_He_energy.txt", 1),
        ("CALET_C_kEnergyPerNucleon.txt", "CALET_C_energy.txt", 12),
        ("CALET_O_kEnergyPerNucleon.txt", "CALET_O_energy.txt", 16),
        ("CALET_Fe_kEnergyPerNucleon.txt", "CALET_Fe_energy.txt", 56),
    ]

    for energy_file, output_file, A in datasets:
        E_min, E_max, I_E, e_sta_lo, e_sta_up, e_sys_lo, e_sys_up = readfile(f"crdb/{energy_file}")
        if A == 1:
            data = [geom_mean(E_min, E_max), I_E, e_sta_lo, e_sta_up, e_sys_lo, e_sys_up]
        else:
            data = transform_T2E(E_min, E_max, I_E, e_sta_lo, e_sta_up, e_sys_lo, e_sys_up, A=A)
        dump(data, output_file)

    transform_crdb_generic("CALET_p_He_ratio_rigidity.txt")


def transform_DAMPE():
    """Transform and dump DAMPE data."""
    datasets = [
        ("DAMPE_H_kEnergy.txt", "DAMPE_H_energy.txt"),
        ("DAMPE_He_kEnergy.txt", "DAMPE_He_energy.txt"),
    ]

    for energy_file, output_file in datasets:
        E_min, E_max, I_E, e_sta_lo, e_sta_up, e_sys_lo, e_sys_up = readfile(f"crdb/{energy_file}")
        data = [geom_mean(E_min, E_max), I_E, e_sta_lo, e_sta_up, e_sys_lo, e_sys_up]
        dump(data, output_file)


def transform_CREAM():
    """Transform and dump CREAM data."""
    datasets: list[tuple[str, str, float, float | None]] = [
        ("CREAM_H_kEnergy.txt", "CREAM_H_energy.txt", 1, None),
        ("ISS-CREAM_H_kEnergy.txt", "ISS-CREAM_H_energy.txt", 1, None),
        ("CREAM_He_kEnergyPerNucleon.txt", "CREAM_He_energy.txt", 4, None),
        ("CREAM_C_kEnergyPerNucleon.txt", "CREAM_C_energy.txt", 12, None),
        ("CREAM_O_kEnergyPerNucleon.txt", "CREAM_O_energy.txt", 16, None),
        ("CREAM_Mg_kEnergyPerNucleon.txt", "CREAM_Mg_energy.txt", 24, 2e5),
        ("CREAM_Si_kEnergyPerNucleon.txt", "CREAM_Si_energy.txt", 28, None),
        ("CREAM_Fe_kEnergyPerNucleon.txt", "CREAM_Fe_energy.txt", 56, None),
    ]

    for file, output_file, A, max_Etot in datasets:
        E_min, E_max, I_E, e_sta_lo, e_sta_up, e_sys_lo, e_sys_up = readfile(f"crdb/{file}")

        if max_Etot is not None:
            max_Epernucl = max_Etot / A
            mask = E_max < max_Epernucl
            E_min = E_min[mask]
            E_max = E_max[mask]
            I_E = I_E[mask]
            e_sta_lo = e_sta_lo[mask]
            e_sta_up = e_sta_up[mask]
            e_sys_lo = e_sys_lo[mask]
            e_sys_up = e_sys_up[mask]

        if A == 1:
            data = [geom_mean(E_min, E_max), I_E, e_sta_lo, e_sta_up, e_sys_lo, e_sys_up]
        else:
            data = transform_T2E(E_min, E_max, I_E, e_sta_lo, e_sta_up, e_sys_lo, e_sys_up, A=A)
        dump(data, output_file)


def transform_allParticles():
    """Transform and dump ALL PARTICLES data."""
    datasets = [
        "NUCLEON_all_energy.txt",
        "GAMMA_SIBYLL_all_energy.txt",
        "TALE_QGSJET-II-04_all_energy.txt",
        "TUNKA-133_QGSJET-01_all_energy.txt",
        "KASCADE_QGSJET-01_all_energy.txt",
        "KASCADE_SIBYLL_21_all_energy.txt",
        "KGRANDE_QGSJET-II-04_all_energy.txt",
        "KGRANDE_SIBYLL_23_all_energy.txt",
        "ICETOP_QGSJET-II-04_all_energy.txt",
        "ICETOP_SIBYLL_21_all_energy.txt",
        "ICECUBE_SIBYLL_21_all_energy.txt",
        "AUGER_all_energy.txt",
        "TA_all_energy.txt",
        "TIBET_light_energy.txt",
        "NUCLEON_H_energy.txt",
    ]

    for file in datasets:
        E_min, E_max, I_E, e_sta_lo, e_sta_up, e_sys_lo, e_sys_up = readfile(f"crdb/{file}")
        data = [geom_mean(E_min, E_max), I_E, e_sta_lo, e_sta_up, e_sys_lo, e_sys_up]
        dump(data, file)


def transform_Cagnoli2024():
    file = "allpart_HET_wGen26_onlyBGO_RGWeights_SatCorrp_5UnfCycle_100GeV_1PeV_2016-23.txt"
    E_min, E_max, I_E, e_sta_lo, e_sta_up = np.loadtxt(
        f"lake/{file}", usecols=(2, 3, 4, 5, 6), unpack=True
    )
    # dropping the last 3 points that are conflicting with LHAASO data
    mask = np.ones_like(E_min, dtype=bool)
    mask[-3:] = False
    data = [
        geom_mean(E_min[mask], E_max[mask]),
        I_E[mask],
        e_sta_lo[mask],
        e_sta_up[mask],
        e_sta_lo[mask],
        e_sta_up[mask],
    ]
    dump(data, "DAMPE_all_energy.txt")


def transform_TIBET_all():
    datasets = [
        ("Tibet_QGSJET+HD_allParticle_totalEnergy.txt", "TIBET_QGSJET+HD_all_energy.txt"),
        ("Tibet_QGSJET+PD_allParticle_totalEnergy.txt", "TIBET_QGSJET+PD_all_energy.txt"),
        ("Tibet_SIBYLL+HD_allParticle_totalEnergy.txt", "TIBET_SIBYLL+HD_all_energy.txt"),
    ]
    for file, output_file in datasets:
        E, I_E, e_sta = np.loadtxt(f"lake/{file}", usecols=(0, 1, 2), unpack=True)
        data = [E, I_E, e_sta, e_sta, 0.0 * e_sta, 0.0 * e_sta]
        dump(data, output_file)


def transform_DAMPE_light():
    file = "DAMPE_light_totalEnergy.txt"
    E_min, E_max, I_E, e_sta, e_sys_ana, e_sys_had = np.loadtxt(
        f"lake/{file}", usecols=(1, 2, 3, 4, 5, 6), unpack=True
    )
    data = [geom_mean(E_min, E_max), I_E, e_sta, e_sta, e_sys_ana, e_sys_ana]
    dump(data, "DAMPE_light_energy.txt")


def transform_CREAM_light():
    H = np.loadtxt("crdb/CREAM_H_kEnergy.txt", usecols=(0, 1, 2, 3, 4, 5, 6), unpack=True)
    He = np.loadtxt(
        "crdb/CREAM_He_kEnergyPerNucleon.txt", usecols=(0, 1, 2, 3, 4, 5, 6), unpack=True
    )
    E_min = H[0]
    E_max = H[1]
    I_E = H[2] + He[2] / 4.0
    e_sta = H[3] + He[3] / 4.0
    e_sys = H[5] + He[5] / 4.0
    data = [geom_mean(E_min, E_max), I_E, e_sta, e_sta, e_sys, e_sys]
    dump(data, "CREAM_light_energy.txt")


def transform_LHAASO():
    filename = "lake/LHAASO_all_energy.txt"
    logEmin, logEmax, I_E, e_sta, e_sys = np.loadtxt(filename, usecols=(0, 1, 2, 3, 4), unpack=True)
    E = np.power(10.0, 0.5 * (logEmin + logEmax))
    data = [E, I_E, e_sta, e_sta, e_sys, e_sys]
    dump(data, "LHAASO_QGSJET-II-04_all_energy.txt")

    I_E, e_sta, e_sys = np.loadtxt(filename, usecols=(5, 6, 7), unpack=True)
    data = [E, I_E, e_sta, e_sta, e_sys, e_sys]
    dump(data, "LHAASO_EPOS-LHC_all_energy.txt")

    I_E, e_sta, e_sys = np.loadtxt(filename, usecols=(8, 9, 10), unpack=True)
    data = [E, I_E, e_sta, e_sta, e_sys, e_sys]
    dump(data, "LHAASO_SIBYLL-23_all_energy.txt")

    filename = "lake/LHAASO_lnA_energy.txt"
    logEmin, logEmax, lnA, e_sta, e_sys = np.loadtxt(filename, usecols=(0, 1, 2, 3, 4), unpack=True)
    E = np.power(10.0, 0.5 * (logEmin + logEmax))
    data = [E, lnA, e_sta, e_sta, e_sys, e_sys]
    dump(data, "LHAASO_QGSJET-II-04_lnA_energy.txt")

    lnA, e_sta, e_sys = np.loadtxt(filename, usecols=(5, 6, 7), unpack=True)
    data = [E, lnA, e_sta, e_sta, e_sys, e_sys]
    dump(data, "LHAASO_EPOS-LHC_lnA_energy.txt")

    lnA, e_sta, e_sys = np.loadtxt(filename, usecols=(8, 9, 10), unpack=True)
    data = [E, lnA, e_sta, e_sta, e_sys, e_sys]
    dump(data, "LHAASO_SIBYLL-23_lnA_energy.txt")


def transform_GRAPES():
    filename = "lake/GRAPES_H_totalEnergy.txt"
    E, I_E, e_sta, e_sys_up, e_sys_lo = np.loadtxt(filename, usecols=(0, 1, 2, 3, 4), unpack=True)
    data = [E, I_E, e_sta, e_sta, e_sys_lo, e_sys_up]
    dump(data, "GRAPES_H_energy.txt")


def transform_LHAASO_protons() -> None:
    source = Path("lake/LHAASO_H_table_S4.txt")
    row_re = re.compile(
        r"(-?[\d.]+) ~ (-?[\d.]+) \d+ \(([\d.]+)±([\d.]+)±([\d.]+) ([\d.]+)±([\d.]+)±([\d.]+)\) x10-(\d+)"
    )
    table: list[list[float]] = []
    for line in source.read_text().splitlines():
        m = row_re.match(line)
        assert m is not None
        table.append([float(v) for v in m.groups()])

    # print(*table, sep="\n")
    # LHAASO energies are in lg(E / PeV), so we shift by 6 orders to get GeV
    E = [10 ** ((lgE_min + lgE_max) / 2 + 6) for (lgE_min, lgE_max, *_) in table]

    def dump_table(table: list[tuple[float, ...]], filename: str):
        dump(np.array(table).T, filename)

    dump_table(
        [
            (
                E_bin,
                flux * (10**-lg_flux_mult) / 1e6,
                stat * 10**-lg_flux_mult / 1e6,
                stat * 10**-lg_flux_mult / 1e6,
                syst * 10**-lg_flux_mult / 1e6,
                syst * 10**-lg_flux_mult / 1e6,
            )
            for E_bin, (_, _, flux, stat, syst, _, _, _, lg_flux_mult) in zip(E, table)
        ],
        "LHAASO_QGSJET-II-04_H_energy.txt",
    )

    dump_table(
        [
            (
                E_bin,
                flux * 10**-lg_flux_mult / 1e6,
                stat * 10**-lg_flux_mult / 1e6,
                stat * 10**-lg_flux_mult / 1e6,
                syst * 10**-lg_flux_mult / 1e6,
                syst * 10**-lg_flux_mult / 1e6,
            )
            for E_bin, (_, _, _, _, _, flux, stat, syst, lg_flux_mult) in zip(E, table)
        ],
        "LHAASO_EPOS-LHC_H_energy.txt",
    )


def transform_KASCADE_reanalysis() -> None:
    output_prefix = "KASCADE_re_QGSJET-II-04"
    text = Path("lake/KASCADE_re_table_2.txt").read_text()

    paragraphs = text.split("\n\n")

    def apply_exp(f: float, fstat: float, fsyst: float, exp: float) -> tuple[float, float, float]:
        mult = 10 ** (-exp)
        return f * mult, fstat * mult, fsyst * mult

    def dump_table(table: list[tuple[float, float, float, float]], filename: str) -> None:
        table_arr = np.array(table).T
        E, flux, stat, syst = table_arr
        dump((E, flux, stat, stat, syst, syst), filename)

    H_table = []
    He_table = []
    C_table = []

    energy_regex = r"(\d+\.\d+)\s*−\s*(\d+\.\d+)"
    value_regex = r"\((\d+\.\d+)\s*±\s*(\d+\.\d+)\s*±\s*(\d+\.\d+)\)\s*×\s*10−(\d+)"
    delim = r"\s+"
    table_row_regex_1 = re.compile(
        energy_regex + delim + value_regex + delim + value_regex + delim + value_regex
    )
    for line in paragraphs[1].splitlines():
        m = table_row_regex_1.match(line)
        assert m is not None
        (
            lgE_min,
            lgE_max,
            p,
            pstat,
            psyst,
            pexp,
            He,
            Hestat,
            Hesyst,
            Heexp,
            C,
            Cstat,
            Csyst,
            Cexp,
        ) = (float(val_str) for val_str in m.groups())
        E = 10 ** ((lgE_min + lgE_max) / 2)
        H_table.append((E, *apply_exp(p, pstat, psyst, pexp)))
        He_table.append((E, *apply_exp(He, Hestat, Hesyst, Heexp)))
        C_table.append((E, *apply_exp(C, Cstat, Csyst, Cexp)))

    dump_table(H_table, f"{output_prefix}_H_energy.txt")
    dump_table(He_table, f"{output_prefix}_He_energy.txt")
    dump_table(C_table, f"{output_prefix}_C_energy.txt")

    table_row_regex_2 = re.compile(energy_regex + delim + value_regex + delim + value_regex)
    Si_table = []
    Fe_table = []
    for line in paragraphs[2].splitlines():
        m = table_row_regex_2.match(line)
        assert m is not None
        lgE_min, lgE_max, Si, Sistat, Sisyst, Siexp, Fe, Festat, Fesyst, Feexp = (
            float(val_str) for val_str in m.groups()
        )
        E = 10 ** ((lgE_min + lgE_max) / 2)
        Si_table.append((E, *apply_exp(Si, Sistat, Sisyst, Siexp)))
        Fe_table.append((E, *apply_exp(Fe, Festat, Fesyst, Feexp)))

    dump_table(Si_table, f"{output_prefix}_Si_energy.txt")
    dump_table(Fe_table, f"{output_prefix}_Fe_energy.txt")

    # lnA
    text = Path("lake/KASCADE_re_table_1.txt").read_text()
    lnA_table = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        line = line.replace("−", " ")
        lgE_min, lgE_max, lnA, stat, syst, _syst_th_lo, _syst_th_up = list(map(float, line.split()))
        E = 10 ** ((lgE_min + lgE_max) / 2)
        lnA_table.append((E, lnA, stat, syst))
    dump_table(lnA_table, f"{output_prefix}_lnA_energy.txt")

    # allparticle
    text = Path("lake/KASCADE_re_table_3.txt").read_text()
    all_table = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        line = line.replace("×10−", " ")
        line = line.replace("−", " ")
        lgE_min, lgE_max, flux, stat, syst, _syst_th_lo, _syst_th_up, exp = list(
            map(float, line.split())
        )
        E = 10 ** ((lgE_min + lgE_max) / 2)
        all_table.append((E, *apply_exp(flux, stat, syst, exp)))
    dump_table(all_table, f"{output_prefix}_all_energy.txt")


def transform_HAWC_2025() -> None:
    text = Path("lake/HAWC_2025_table_2.txt").read_text()

    val_re = r"(\d+\.\d+)"
    flux_value_regex = (
        r"\("
        + val_re
        + r"\s*±\s*"
        + val_re
        + r"\s*\+\s*"
        + val_re
        + r"\s*−\s*"
        + val_re
        + r"\)"
        + r"\s*×\s*10−(\d+)"
    )
    delim = r"\s+"
    table_row_regex = re.compile(
        val_re + delim + val_re + delim + val_re + delim + flux_value_regex
    )
    rows = []
    for line in text.splitlines():
        m = table_row_regex.match(line)
        assert m is not None
        (_, _, E_mean_TeV, F, F_stat, F_sys_up, F_sys_low, F_exp) = (
            float(val_str) for val_str in m.groups()
        )
        mult = 10**-F_exp
        rows.append((E_mean_TeV * 1e3, F * mult, F_stat * mult, F_sys_up * mult, F_sys_low * mult))

    table_arr = np.array(rows).T
    E, flux, stat, syst_up, syst_low = table_arr
    dump((E, flux, stat, stat, syst_low, syst_up), "HAWC_QGSJET-II-04_all_energy.txt")


if __name__ == "__main__":
    logging.info("Starting AMS02 transformation")
    transform_AMS02()

    logging.info("Starting CALET transformation")
    transform_CALET()

    logging.info("Starting DAMPE transformation")
    transform_DAMPE()

    logging.info("Starting CREAM transformation")
    transform_CREAM()

    logging.info("Starting ALLPARTICLES transformation")
    transform_allParticles()

    logging.info("Starting DAMPE ALL transformation")
    transform_Cagnoli2024()

    logging.info("Starting CREAM-light transformation")
    transform_CREAM_light()

    logging.info("Starting TIBET transformation")
    transform_TIBET_all()

    logging.info("Starting DAMPE-light transformation")
    transform_DAMPE_light()

    logging.info("Starting LHAASO transformation")
    transform_LHAASO()

    logging.info("Starting GRAPES transformation")
    transform_GRAPES()

    transform_crdb_generic("NUCLEON_p_He_ratio_rigidity.txt")
    transform_LHAASO_protons()

    transform_KASCADE_reanalysis()
    transform_HAWC_2025()

    logging.info("All transformations completed")
