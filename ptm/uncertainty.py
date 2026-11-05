"""
Monte Carlo propagation of parameter uncertainty onto phase-diagram
predictions -- the second "continuous random variable" pillar of this
project (the first is continuous thermodynamics in continuous.py).

Tc, Pc, and omega are never known exactly; they come from a correlation
or a regression fit with a reported (or, here, fitted) standard error.
Treating them as independent normal random variables and sampling
through the full nonlinear EOS/flash model is the simplest consistent way
to turn that parameter uncertainty into a confidence band on a *derived*
quantity (Psat(T), a bubble curve, a phase envelope) that itself depends
on the parameters nonlinearly -- a first-order (delta-method) error
propagation would require differentiating the whole flash algorithm,
which Monte Carlo avoids entirely at the cost of sampling noise that
shrinks as 1/sqrt(n_samples).
"""
from __future__ import annotations

import numpy as np

from .eos import pure_psat_newton


def mc_psat_band(Tc, Pc_Pa, omega, T_grid, sigma_Tc=0.0, sigma_Pc=0.0,
                  sigma_omega=0.0, n_samples=500, seed=0, ci=0.95):
    """Monte Carlo 95% (or `ci`) confidence band for Psat(T) given
    Gaussian uncertainty in Tc, Pc, omega.

    Returns dict with T_grid, median, lower, upper (all same shape as
    T_grid), each a Psat curve in Pa.
    """
    rng = np.random.default_rng(seed)
    T_grid = np.asarray(T_grid, dtype=float)
    samples = np.empty((n_samples, len(T_grid)))
    alpha = (1 - ci) / 2

    Tc_s = rng.normal(Tc, sigma_Tc, n_samples) if sigma_Tc > 0 else np.full(n_samples, Tc)
    Pc_s = rng.normal(Pc_Pa, sigma_Pc, n_samples) if sigma_Pc > 0 else np.full(n_samples, Pc_Pa)
    om_s = rng.normal(omega, sigma_omega, n_samples) if sigma_omega > 0 else np.full(n_samples, omega)

    for k in range(n_samples):
        for j, T in enumerate(T_grid):
            if T >= Tc_s[k]:
                samples[k, j] = np.nan
                continue
            try:
                samples[k, j] = pure_psat_newton(Tc_s[k], Pc_s[k], om_s[k], T)
            except Exception:
                samples[k, j] = np.nan

    median = np.nanmedian(samples, axis=0)
    lower = np.nanpercentile(samples, 100 * alpha, axis=0)
    upper = np.nanpercentile(samples, 100 * (1 - alpha), axis=0)
    return {"T_grid": T_grid, "median": median, "lower": lower, "upper": upper,
            "n_samples": n_samples, "ci": ci}
