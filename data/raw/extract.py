"""
One-time extraction of clean (T, Psat) tables from the raw NIST WebBook
downloads in this folder, down-sampled to a manageable number of points.

Source: NIST Chemistry WebBook, Thermophysical Properties of Fluid Systems
(https://webbook.nist.gov/chemistry/fluid/), "Saturation Properties"
(Action=Data, Type=SatT). Underlying reference equations of state:
  - Methane:  Setzmann & Wagner (1991), J. Phys. Chem. Ref. Data 20, 1061.
  - Ethane:   Buecker & Wagner (2006), J. Phys. Chem. Ref. Data 35, 205.
  - Propane:  Lemmon, McLinden & Wagner (2009), J. Chem. Eng. Data 54, 3141.
  - n-Butane: Buecker & Wagner (2006), J. Chem. Eng. Data 51, 929.
  - CO2:      Span & Wagner (1996), J. Phys. Chem. Ref. Data 25, 1509.
These are the NIST/IUPAC reference ("REFPROP") equations of state, each
fitted to the body of published experimental data for that fluid; treated
here as the literature ground truth for pure-component vapor pressure.

Run from this directory: python extract.py
Writes cleaned CSVs to ../vle/<name>_psat.csv
"""
import csv
import os

JOBS = {
    "methane":  ("Methane",     74.82, 190.56),
    "ethane":   ("Ethane",      74.84, 305.32),
    "propane":  ("Propane",     74.98, 369.89),
    "nbutane":  ("n-Butane",   106.97, 425.12),
    "co2":      ("Carbon dioxide", 124.38, 304.13),
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "vle")
os.makedirs(OUT_DIR, exist_ok=True)

CITATIONS = {
    "methane": "Setzmann, U.; Wagner, W. J. Phys. Chem. Ref. Data 1991, 20, 1061-1155.",
    "ethane": "Buecker, D.; Wagner, W. J. Phys. Chem. Ref. Data 2006, 35, 205-266.",
    "propane": "Lemmon, E.W.; McLinden, M.O.; Wagner, W. J. Chem. Eng. Data 2009, 54, 3141-3180.",
    "nbutane": "Buecker, D.; Wagner, W. J. Chem. Eng. Data 2006, 51, 929-930 (reference EOS for n-butane).",
    "co2": "Span, R.; Wagner, W. J. Phys. Chem. Ref. Data 1996, 25, 1509-1596.",
}

for key, (name, _mw, Tc) in JOBS.items():
    rows = []
    with open(os.path.join(os.path.dirname(__file__), f"{key}.tsv")) as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            try:
                T, P = float(row[0]), float(row[1])
            except (ValueError, IndexError):
                continue
            if T >= Tc - 0.1:
                continue
            rows.append((T, P))

    # Down-sample to ~40 evenly spaced points across the full range
    n = len(rows)
    step = max(1, n // 40)
    sampled = rows[::step]
    if sampled[-1][0] < rows[-1][0] - 1:
        sampled.append(rows[-1])

    out_path = os.path.join(OUT_DIR, f"{key}_psat.csv")
    with open(out_path, "w", newline="") as f:
        f.write(f"# {name} saturation (vapor) pressure vs temperature\n")
        f.write(f"# Source: NIST Chemistry WebBook, Thermophysical Properties of Fluid Systems,\n")
        f.write(f"#   https://webbook.nist.gov/chemistry/fluid/ (Action=Data, Type=SatT), fetched 2026-11-04.\n")
        f.write(f"# Reference EOS: {CITATIONS[key]}\n")
        f.write(f"# Critical temperature used as cutoff: {Tc} K\n")
        w = csv.writer(f)
        w.writerow(["T_K", "P_bar"])
        for T, P in sampled:
            w.writerow([f"{T:.3f}", f"{P:.6g}"])
    print(out_path, len(sampled), "points")
