"""
Isothermal two-phase (TP) flash: given feed z, T, P, find vapor fraction
beta and phase compositions x (liquid), y (vapor) such that

    z_i = (1-beta) x_i + beta y_i          (component balance)
    y_i = K_i x_i                          (equilibrium, K from fugacity match)
    sum x_i = sum y_i = 1

Rachford-Rice reduces the component balances + equilibrium relations (for
a *given* set of K-values) to the single scalar equation

    g(beta) = sum_i z_i (K_i - 1) / (1 + beta (K_i - 1)) = 0,  beta in (0, 1)

g is monotonically decreasing in beta on that interval wherever a physical
root exists, so bounded bisection/Newton always converges (see
docs/THEORY.md section 4 for the monotonicity proof). The outer loop
updates K_i = phi_i^liquid / phi_i^vapor from the EOS (successive
substitution); this is the standard two-level flash algorithm.
"""
from __future__ import annotations

import numpy as np

from .eos import ln_phi_mix
from .stability import wilson_k


def rachford_rice_beta(z, K, tol=1e-12, max_iter=200) -> float:
    """Solve g(beta)=0 for beta in (0,1) by Newton with bisection fallback."""
    z = np.asarray(z, dtype=float)
    K = np.asarray(K, dtype=float)

    def g(beta):
        return np.sum(z * (K - 1.0) / (1.0 + beta * (K - 1.0)))

    def gprime(beta):
        return -np.sum(z * (K - 1.0) ** 2 / (1.0 + beta * (K - 1.0)) ** 2)

    lo, hi = 0.0, 1.0
    eps = 1e-10
    lo, hi = lo + eps, hi - eps
    beta = 0.5
    for _ in range(max_iter):
        gb = g(beta)
        if abs(gb) < tol:
            return beta
        if gb > 0:
            lo = beta
        else:
            hi = beta
        step = gb / gprime(beta)
        beta_newton = beta - step
        beta = beta_newton if lo < beta_newton < hi else 0.5 * (lo + hi)
    return beta


def tp_flash(z, a_i, b_i, kij, T, P, Tc, Pc_Pa, omega,
             max_outer=200, tol=1e-10):
    """Two-phase isothermal flash.

    Returns dict with beta (vapor mole fraction), x (liquid comp), y (vapor
    comp), K (equilibrium ratios), and n_iter.
    """
    z = np.asarray(z, dtype=float)
    K = wilson_k(Tc, Pc_Pa, omega, T, P)

    for it in range(1, max_outer + 1):
        beta = rachford_rice_beta(z, K)
        x = z / (1.0 + beta * (K - 1.0))
        x = x / x.sum()
        y = K * x
        y = y / y.sum()

        _, lnphi_l = ln_phi_mix(x, a_i, b_i, kij, T, P, "liquid")
        _, lnphi_v = ln_phi_mix(y, a_i, b_i, kij, T, P, "vapor")
        K_new = np.exp(lnphi_l - lnphi_v)

        if np.max(np.abs(K_new / K - 1.0)) < tol:
            K = K_new
            break
        K = K_new

    beta = rachford_rice_beta(z, K)
    x = z / (1.0 + beta * (K - 1.0))
    x = x / x.sum()
    y = K * x
    y = y / y.sum()
    return {"beta": beta, "x": x, "y": y, "K": K, "n_iter": it}
