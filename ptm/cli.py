"""Command-line entry points: `python -m ptm <command> ...`"""
from __future__ import annotations

import argparse
import os

import numpy as np

from . import components as C
from . import diagrams
from .continuous import (check_moments, gamma_params_from_moments,
                          gauss_laguerre_pseudo_components,
                          pseudo_component_properties)
from .eos import pure_a_b, pure_psat_newton
from .saturation import pxy_diagram
from .ternary import ternary_tie_lines
from .uncertainty import mc_psat_band
from .validation import print_table, validate_all, write_table

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# Five validated binary systems + one ternary, all within 250-400 K / <=50 atm.
BINARIES = {
    "propane-nbutane": (C.PROPANE, C.NBUTANE, 320.0, 0.0),
    "ethane-propane": (C.ETHANE, C.PROPANE, 280.0, 0.0),
    "co2-propane": (C.CO2, C.PROPANE, 300.0, 0.135),
    "methane-nbutane": (C.METHANE, C.NBUTANE, 310.0, 0.02),
    "co2-nbutane": (C.CO2, C.NBUTANE, 310.0, 0.13),
}
# k_ij for CO2 pairs: no freely-accessible literature table was found (see
# README.md); 0.13 is this project's own placeholder of the right order of
# magnitude for CO2+paraffin pairs (physically, CO2's quadrupole moment
# needs a positive correction to vdW1 mixing), NOT a fitted or cited value.
# Hydrocarbon-hydrocarbon pairs use k_ij=0, the standard default.


def cmd_validate(args):
    results = validate_all()
    print_table(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "validation_table.csv")
    write_table(results, out)
    print(f"\nwrote {out}")


def cmd_diagram(args):
    comp_a, comp_b, T_default, kij_value = BINARIES[args.system]
    T = args.T if args.T else T_default
    rows = pxy_diagram(comp_a, comp_b, T, kij_value, n=args.n)
    path = diagrams.plot_pxy(rows, comp_a.name, comp_b.name, T, kij_value,
                              f"pxy_{args.system}_{T:.0f}K.png")
    print(f"wrote {path}")


def cmd_psat(args):
    comp = C.get(args.component)
    T_data, P_data = [], []
    fname = {"methane": "methane_psat.csv", "ethane": "ethane_psat.csv",
             "propane": "propane_psat.csv", "n-butane": "nbutane_psat.csv",
             "carbon dioxide": "co2_psat.csv"}[comp.name]
    path = os.path.join(os.path.dirname(__file__), "..", "data", "vle", fname)
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("T_K"):
                continue
            t, p = line.strip().split(",")
            T_data.append(float(t))
            P_data.append(float(p))
    T_data = np.array(T_data)
    P_data_full = np.array(P_data)
    margin = 1.5
    # Deep-cryogenic points (Tr well below ~0.5) push Psat/Pc below the
    # Psat bracket search's resolution; outside this project's 250-400 K
    # scope anyway, so excluded here (full range is still used for the
    # EOS parameter fit in validation.py, which filters identically).
    keep = (T_data < comp.Tc - margin) & (T_data > 0.5 * comp.Tc)
    T_data = T_data[keep]
    P_data = P_data_full[keep]

    T_model = np.linspace(T_data.min(), T_data.max(), 60)
    P_model = np.array([pure_psat_newton(comp.Tc, comp.Pc_Pa, comp.omega, T) for T in T_model]) / 1e5

    band = None
    if args.mc > 0:
        T_grid = np.linspace(T_data.min(), T_data.max(), 8)
        band = mc_psat_band(comp.Tc, comp.Pc_Pa, comp.omega, T_grid,
                             sigma_Tc=0.3, sigma_Pc=0.02 * comp.Pc_Pa,
                             sigma_omega=0.01 * comp.omega, n_samples=args.mc)

    out_path = diagrams.plot_psat_parity(comp.name, T_data, P_data, T_model, P_model,
                                          band, f"psat_{args.component.replace(' ', '_')}.png")
    print(f"wrote {out_path}")


def cmd_ternary(args):
    comps = [C.METHANE, C.PROPANE, C.NBUTANE]
    kij = np.zeros((3, 3))
    T, P = 310.9, 6.9e5  # ~100 F, ~1 MPa: order-of-magnitude match to
    # Rigas, Mason & Thodos (1959) J. Chem. Eng. Data 4, 201 (conditions
    # only; their numeric table was not accessible -- see README.md).
    lines = ternary_tie_lines(comps, T, P, kij)
    path = diagrams.plot_ternary(lines, ["methane", "propane", "n-butane"],
                                  f"Methane-propane-n-butane tie lines, {T:.0f} K, {P/1e5:.1f} bar (PR EOS, predictive)",
                                  "ternary_methane_propane_nbutane.png")
    print(f"wrote {path} ({len(lines)} tie lines)")


def cmd_condensate(args):
    mean, var, eta = 150.0, 1200.0, 90.0
    alpha, beta = gamma_params_from_moments(mean, var, eta)
    q = 4
    nodes, weights = gauss_laguerre_pseudo_components(alpha, beta, eta, q)
    ok = check_moments(nodes, weights, mean, var)
    print(f"gamma(alpha={alpha:.3f}, beta={beta:.3f}, eta={eta}) -> {q}-point quadrature")
    for M, w in zip(nodes, weights):
        Tc, Pc, om = pseudo_component_properties(M)
        print(f"  M={M:7.2f} g/mol  z={w:.4f}  Tc={float(Tc):.1f} K  Pc={float(Pc):.2f} bar  omega={float(om):.3f}")
    print(f"moment check (mean, variance within 5%): {ok}")


def build_parser():
    p = argparse.ArgumentParser(prog="ptm")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("validate", help="validate PR EOS vs NIST pure-fluid data")
    pv.set_defaults(func=cmd_validate)

    pd = sub.add_parser("diagram", help="generate a Pxy diagram for a binary system")
    pd.add_argument("--system", choices=list(BINARIES), required=True)
    pd.add_argument("--T", type=float, default=None, help="temperature in K")
    pd.add_argument("--n", type=int, default=21)
    pd.set_defaults(func=cmd_diagram)

    pp = sub.add_parser("psat", help="pure-component Psat parity plot vs NIST, with MC band")
    pp.add_argument("--component", choices=list(C.REGISTRY), required=True)
    pp.add_argument("--mc", type=int, default=150, help="Monte Carlo samples (0 to skip)")
    pp.set_defaults(func=cmd_psat)

    pt = sub.add_parser("ternary", help="methane/propane/n-butane tie-line diagram")
    pt.set_defaults(func=cmd_ternary)

    pc = sub.add_parser("condensate", help="continuous thermodynamics demo (gamma C7+ fraction)")
    pc.set_defaults(func=cmd_condensate)

    pa = sub.add_parser("all", help="regenerate every figure and the validation table")
    pa.set_defaults(func=cmd_all)

    return p


def cmd_all(args):
    cmd_validate(args)
    for system in BINARIES:
        a = argparse.Namespace(system=system, T=None, n=21)
        cmd_diagram(a)
    for comp in C.REGISTRY:
        if comp == "n-pentane":
            continue  # anchor point for continuous.py's regression only; no Psat CSV fetched
        a = argparse.Namespace(component=comp, mc=100)
        cmd_psat(a)
    cmd_ternary(argparse.Namespace())
    cmd_condensate(argparse.Namespace())


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
