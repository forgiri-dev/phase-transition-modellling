"""
Validate the PR EOS against the literature pure-fluid data in data/vle/
(NIST Chemistry WebBook reference equations of state; see
data/raw/extract.py for the exact source and citation per fluid).

For each component: split its saturation-pressure table into alternating
train/test points, MLE-refit omega on the training half (fitting.py),
then report AAD% of the EOS-predicted Psat against the *held-out* test
half, at both the literature omega and the refit omega. This is the
concrete, reproducible basis for the "<5% deviation from literature VLE
data" claim -- for binary/ternary *mixtures*, no independently-sourced
experimental VLE table was available to this project (see README.md for
why), so mixture diagrams are reported as EOS predictions, not validated
point-by-point.
"""
from __future__ import annotations

import csv
import os

import numpy as np

from .components import REGISTRY
from .fitting import aad_percent, fit_omega_mle

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vle")

FILES = {
    "methane": "methane_psat.csv",
    "ethane": "ethane_psat.csv",
    "propane": "propane_psat.csv",
    "n-butane": "nbutane_psat.csv",
    "carbon dioxide": "co2_psat.csv",
}


def _load(path):
    T, P = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("T_K"):
                continue
            t, p = line.strip().split(",")
            T.append(float(t))
            P.append(float(p) * 1e5)  # bar -> Pa
    return np.array(T), np.array(P)


def validate_all(margin_K=1.5, T_min=250.0, T_max=400.0):
    """Returns a list of per-component result dicts, and prints a summary
    table. Restricted to [T_min, T_max] -- the 250-400 K window this
    project targets -- intersected with each fluid's own subcritical
    range. `margin_K` additionally excludes points within this many K of
    Tc, where the bisection bracket in pure_psat_newton narrows faster
    than float precision can resolve.
    """
    results = []
    for name, fname in FILES.items():
        comp = REGISTRY[name]
        T, P = _load(os.path.join(DATA_DIR, fname))
        keep = (T < (comp.Tc - margin_K)) & (T >= T_min) & (T <= T_max)
        T, P = T[keep], P[keep]
        if len(T) < 6:
            # fluid's subcritical range doesn't reach this window (e.g.
            # methane, Tc=190.6 K, is entirely below 250 K): fall back to
            # its own full available range instead of skipping it.
            T_full, P_full = _load(os.path.join(DATA_DIR, fname))
            keep = T_full < (comp.Tc - margin_K)
            T, P = T_full[keep], P_full[keep]

        T_train, P_train = T[::2], P[::2]
        T_test, P_test = T[1::2], P[1::2]

        aad_lit = aad_percent(T_test, P_test, comp.Tc, comp.Pc_Pa, comp.omega)
        fit = fit_omega_mle(T_train, P_train, comp.Tc, comp.Pc_Pa, comp.omega)
        aad_fit = aad_percent(T_test, P_test, comp.Tc, comp.Pc_Pa, fit["omega_hat"])

        results.append({
            "component": name,
            "n_train": len(T_train),
            "n_test": len(T_test),
            "omega_lit": comp.omega,
            "omega_fit": fit["omega_hat"],
            "se_omega_fit": fit["se_omega"],
            "aad_pct_literature_omega": aad_lit,
            "aad_pct_fitted_omega": aad_fit,
        })
    return results


def write_table(results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "n_train", "n_test", "omega_lit", "omega_fit",
                    "se_omega_fit", "aad_pct_literature_omega", "aad_pct_fitted_omega"])
        for r in results:
            w.writerow([r["component"], r["n_train"], r["n_test"],
                        f"{r['omega_lit']:.4f}", f"{r['omega_fit']:.4f}",
                        f"{r['se_omega_fit']:.5f}",
                        f"{r['aad_pct_literature_omega']:.3f}",
                        f"{r['aad_pct_fitted_omega']:.3f}"])


def print_table(results):
    print(f"{'component':<16}{'n_tr':>5}{'n_te':>5}{'omega_lit':>11}{'omega_fit':>11}"
          f"{'AAD% lit':>11}{'AAD% fit':>11}")
    for r in results:
        print(f"{r['component']:<16}{r['n_train']:>5}{r['n_test']:>5}"
              f"{r['omega_lit']:>11.4f}{r['omega_fit']:>11.4f}"
              f"{r['aad_pct_literature_omega']:>11.3f}{r['aad_pct_fitted_omega']:>11.3f}")
