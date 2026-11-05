"""
Bubble-point and dew-point pressure solvers, built on the same
successive-substitution-on-K pattern as the flash: these are exactly the
beta -> 0 (bubble) and beta -> 1 (dew) limits of a two-phase flash, where
one phase's composition is already fixed and known to sum to 1.

Bubble P at (T, x): find P, y such that sum_i K_i(T,P) x_i = 1.
Dew P at (T, y):    find P, x such that sum_i x_i = sum_i y_i / K_i(T,P) = 1.
"""
from __future__ import annotations

import numpy as np

from .eos import ln_phi_mix
from .stability import wilson_k


def bubble_P(x, a_i, b_i, kij, T, Tc, Pc_Pa, omega, P_guess=None,
             max_iter=200, tol=1e-9):
    x = np.asarray(x, dtype=float)
    K = wilson_k(Tc, Pc_Pa, omega, T, P_guess if P_guess else np.sum(x * Pc_Pa))
    P = P_guess if P_guess else 1.0 / np.sum(x / Pc_Pa * 1.0)  # crude start
    P = float(np.sum(x * Pc_Pa) * 0.3) if P_guess is None else P_guess

    y = K * x
    for _ in range(max_iter):
        y = y / y.sum()
        _, lnphi_l = ln_phi_mix(x, a_i, b_i, kij, T, P, "liquid")
        _, lnphi_v = ln_phi_mix(y, a_i, b_i, kij, T, P, "vapor")
        K = np.exp(lnphi_l - lnphi_v)
        S = np.sum(K * x)
        if abs(S - 1.0) < tol:
            break
        P = P * S
        y = K * x
    y = y / y.sum()
    return P, y


def dew_P(y, a_i, b_i, kij, T, Tc, Pc_Pa, omega, P_guess=None,
          max_iter=200, tol=1e-9):
    y = np.asarray(y, dtype=float)
    P = P_guess if P_guess else float(1.0 / np.sum(y / Pc_Pa) * 0.3)
    K = wilson_k(Tc, Pc_Pa, omega, T, P)
    x = y / K
    for _ in range(max_iter):
        x = x / x.sum()
        _, lnphi_l = ln_phi_mix(x, a_i, b_i, kij, T, P, "liquid")
        _, lnphi_v = ln_phi_mix(y, a_i, b_i, kij, T, P, "vapor")
        K = np.exp(lnphi_l - lnphi_v)
        S = np.sum(y / K)
        if abs(S - 1.0) < tol:
            break
        P = P / S
        x = y / K
    x = x / x.sum()
    return P, x


def pxy_diagram(comp_a, comp_b, T, kij_value, n=21):
    """Pxy at fixed T for a binary: for each liquid composition x1 in
    [0,1], returns bubble pressure and the equilibrium vapor composition
    y1 (so plotting (x1,P) and (y1,P) gives the classic loop diagram)."""
    from .eos import pure_a_b
    a1, b1 = pure_a_b(comp_a.Tc, comp_a.Pc_Pa, comp_a.omega, T)
    a2, b2 = pure_a_b(comp_b.Tc, comp_b.Pc_Pa, comp_b.omega, T)
    a_i = np.array([a1, a2])
    b_i = np.array([b1, b2])
    kij = np.array([[0.0, kij_value], [kij_value, 0.0]])
    Tc = np.array([comp_a.Tc, comp_b.Tc])
    Pc_Pa = np.array([comp_a.Pc_Pa, comp_b.Pc_Pa])
    omega = np.array([comp_a.omega, comp_b.omega])

    x1s = np.linspace(1e-4, 1 - 1e-4, n)
    P_guess = None
    rows = []
    for x1 in x1s:
        x = np.array([x1, 1 - x1])
        P, y = bubble_P(x, a_i, b_i, kij, T, Tc, Pc_Pa, omega, P_guess=P_guess)
        rows.append((x1, y[0], P))
        P_guess = P
    return rows
