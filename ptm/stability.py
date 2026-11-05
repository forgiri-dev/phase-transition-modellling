"""
Michelsen tangent-plane-distance (TPD) stability test.

Given a feed composition z at (T, P), a single phase at z is stable only if
no trial composition y gives a tangent-plane distance below zero:

    tpd(y) = sum_i y_i [ln y_i + ln phi_i(y) - ln z_i - ln phi_i(z)]

Michelsen (1982) showed it is equivalent, and numerically much better
behaved, to work with unnormalized mole numbers Y and minimize

    tm(Y) = 1 + sum_i Y_i [ln Y_i + ln phi_i(Y/sum Y) - ln z_i - ln phi_i(z) - 1]

A stationary point with tm < 0 proves the feed is unstable (will split into
two phases); successive substitution on Y_i = exp(ln z_i + ln phi_i(z) - ln phi_i(y))
from several initial guesses (pure-component / Wilson-K-value guesses)
finds those stationary points.
"""
from __future__ import annotations

import numpy as np

from .eos import ln_phi_mix


def wilson_k(Tc, Pc_Pa, omega, T, P):
    """Wilson's approximate K-value, used only to seed stability/flash
    initial guesses -- not part of the EOS itself.

        K_i = (Pc_i/P) * exp(5.373 (1+omega_i)(1 - Tc_i/T))
    """
    Tc, Pc_Pa, omega = map(np.asarray, (Tc, Pc_Pa, omega))
    return (Pc_Pa / P) * np.exp(5.373 * (1 + omega) * (1 - Tc / T))


def _tm(Y, z, lnphi_z, a_i, b_i, kij, T, P, phase_guess):
    y = Y / Y.sum()
    _, lnphi_y = ln_phi_mix(y, a_i, b_i, kij, T, P, phase_guess)
    return 1.0 + np.sum(Y * (np.log(Y) + lnphi_y - np.log(z) - lnphi_z - 1.0))


def tpd_stationary(z, a_i, b_i, kij, T, P, Y0, phase_guess="stable", max_iter=200, tol=1e-12):
    """Successive substitution from one initial guess Y0 to a stationary
    point of tm(Y). Returns (Y, tm_value)."""
    _, lnphi_z = ln_phi_mix(z, a_i, b_i, kij, T, P, phase_guess)
    d = np.log(z) + lnphi_z          # constant target
    Y = np.array(Y0, dtype=float)
    for _ in range(max_iter):
        y = Y / Y.sum()
        _, lnphi_y = ln_phi_mix(y, a_i, b_i, kij, T, P, phase_guess)
        Y_new = np.exp(d - lnphi_y)
        if np.max(np.abs(Y_new - Y)) < tol:
            Y = Y_new
            break
        Y = Y_new
    return Y, _tm(Y, z, lnphi_z, a_i, b_i, kij, T, P, phase_guess)


def is_stable(z, a_i, b_i, kij, T, P, Tc, Pc_Pa, omega, tol=1e-9) -> bool:
    """True if the single phase at composition z is stable at (T, P).

    Tries two initial guesses per Michelsen's recipe: a vapor-like trial
    (Y0_i = z_i * K_i) and a liquid-like trial (Y0_i = z_i / K_i), with K
    from Wilson's correlation. If either stationary point has tm < -tol,
    the feed is unstable (a two-phase flash should be attempted instead).
    """
    z = np.asarray(z, dtype=float)
    K = wilson_k(Tc, Pc_Pa, omega, T, P)
    for Y0, guess in ((z * K, "vapor"), (z / K, "liquid")):
        _, tm = tpd_stationary(z, a_i, b_i, kij, T, P, Y0, phase_guess=guess)
        if tm < -tol:
            return False
    return True
