"""
Ternary tie lines via repeated two-phase flashes: for a 3-component system
at fixed (T, P), sweep feed composition z along a line in the Gibbs
triangle and flash each one; feeds that land in the two-phase region
(0 < beta < 1) produce a liquid/vapor tie line.
"""
from __future__ import annotations

import numpy as np

from .eos import pure_a_b
from .flash import tp_flash


def ternary_tie_lines(components, T, P, kij, n_feeds=25, z3_values=(0.3, 0.5, 0.7)):
    """components: list of 3 Component objects, in the order the tie-line
    output (a,b,c) fractions are reported.
    z3_values: fixed mole fractions of the 3rd (heaviest) component to
    sweep across, with the first two components' ratio varied at each.
    Returns list of (liquid_xyz, vapor_xyz) tuples for feeds that flash
    two-phase.
    """
    a_i = np.array([pure_a_b(c.Tc, c.Pc_Pa, c.omega, T)[0] for c in components])
    b_i = np.array([pure_a_b(c.Tc, c.Pc_Pa, c.omega, T)[1] for c in components])
    Tc = np.array([c.Tc for c in components])
    Pc_Pa = np.array([c.Pc_Pa for c in components])
    omega = np.array([c.omega for c in components])

    lines = []
    for z3 in z3_values:
        for frac1 in np.linspace(0.02, 0.98, n_feeds):
            z1 = (1 - z3) * frac1
            z2 = 1 - z3 - z1
            if z2 <= 0:
                continue
            z = np.array([z1, z2, z3])
            try:
                res = tp_flash(z, a_i, b_i, kij, T, P, Tc, Pc_Pa, omega)
            except Exception:
                continue
            beta = res["beta"]
            # Require both a non-trivial vapor fraction AND genuinely
            # distinct phase compositions -- guards against the flash
            # converging to a spurious beta when the true answer is a
            # single stable phase (K_i -> 1 for all i, x ~= y).
            if 0.02 < beta < 0.98 and np.max(np.abs(res["x"] - res["y"])) > 0.03:
                lines.append((tuple(res["x"]), tuple(res["y"])))
    return lines
