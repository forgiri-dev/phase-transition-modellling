"""
Continuous thermodynamics for a heavy ("C7+") fraction.

Real heavy ends (gas condensates, crude oil) are not a handful of discrete
species -- they are a continuum of isomers and homologues usually
characterized only by a molar-mass distribution. Continuous thermodynamics
(Cotterman & Prausnitz, 1985; Whitson, 1983) treats the mole fraction as a
probability density p(M) over molar mass M, most commonly a 3-parameter
gamma distribution:

    p(M) = (M - eta)^(alpha-1) * exp(-(M-eta)/beta) / (beta^alpha * Gamma(alpha))

with mean Mn = eta + alpha*beta and variance sigma^2 = alpha*beta^2. A cubic
EOS cannot integrate over a continuous distribution directly, so the
distribution is replaced by a small number of discrete pseudo-components
via Gauss-Laguerre quadrature: for a gamma density, the quadrature nodes
and weights are *exact* moment-matching nodes, i.e. a q-point rule
reproduces the distribution's first 2q-1 moments exactly. This is the
"continuous random variable" step this project's phase diagrams for the
condensate system are built on.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.special import roots_genlaguerre


def gamma_params_from_moments(mean: float, variance: float, eta: float = 90.0):
    """alpha, beta for a gamma distribution with given mean and variance,
    shifted to start at eta (the minimum molar mass in the fraction, e.g.
    ~90-100 g/mol for a C7+ cut)."""
    alpha = (mean - eta) ** 2 / variance
    beta = variance / (mean - eta)
    return alpha, beta


def gauss_laguerre_pseudo_components(alpha: float, beta: float, eta: float, q: int):
    """q quadrature nodes (molar masses) and weights (mole fractions) for
    the shifted gamma density p(M), via *generalized* Gauss-Laguerre
    quadrature.

    Substituting t = (M-eta)/beta turns the gamma density into the
    canonical generalized-Laguerre weight function:

        p(M) dM = [t^(alpha-1) e^-t / Gamma(alpha)] dt

    `scipy.special.roots_genlaguerre(q, alpha-1)` returns the q nodes t_i
    and weights w_i of the rule that integrates int_0^inf t^(alpha-1) e^-t
    f(t) dt exactly for any polynomial f of degree <= 2q-1 (sum w_i =
    Gamma(alpha) by construction). That is exactly the mean-molar-mass
    moment (f(t)=t, degree 1) and every higher moment up to degree 2q-1,
    so a q-point rule reproduces the distribution's mean and variance
    (degree-1 and degree-2 moments) exactly for any q >= 2 -- the property
    `check_moments` below verifies numerically.
    """
    t_nodes, t_weights = roots_genlaguerre(q, alpha - 1.0)
    nodes = eta + beta * t_nodes            # molar masses M_i
    weights = t_weights / gamma_fn(alpha)   # normalize to mole fractions, sum = 1
    return nodes, weights


def check_moments(nodes, weights, mean, variance, rtol=0.05):
    """Verify the discretization reproduces the distribution's mean and
    variance to within rtol (used by the test suite)."""
    m1 = np.sum(weights * nodes) / np.sum(weights)
    m2 = np.sum(weights * (nodes - m1) ** 2) / np.sum(weights)
    return (abs(m1 - mean) / mean < rtol) and (abs(m2 - variance) / variance < 2 * rtol)


def _fit_property_regression():
    """Least-squares log-log fit of Tc and Pc vs molar mass M, and a
    linear fit of omega vs M, over the five real n-alkanes this project
    already has literature-grade constants for (methane..n-pentane; see
    components.py). Used to *extrapolate* Tc/Pc/omega for the synthetic
    C7+ pseudo-components in the continuous-thermodynamics demo, instead
    of trusting an externally-quoted empirical correlation from memory.
    This is a real (if small, n=5) regression on real literature data --
    the same MLE/least-squares machinery as fitting.py, applied here to
    build inputs for the EOS rather than to validate it.
    """
    from .components import METHANE, ETHANE, PROPANE, NBUTANE, NPENTANE
    alkanes = [METHANE, ETHANE, PROPANE, NBUTANE, NPENTANE]
    M = np.array([c.Mw for c in alkanes])
    Tc = np.array([c.Tc for c in alkanes])
    Pc = np.array([c.Pc for c in alkanes])
    omega = np.array([c.omega for c in alkanes])

    b_Tc, ln_a_Tc = np.polyfit(np.log(M), np.log(Tc), 1)
    b_Pc, ln_a_Pc = np.polyfit(np.log(M), np.log(Pc), 1)
    m_om, c_om = np.polyfit(M, omega, 1)
    return dict(a_Tc=np.exp(ln_a_Tc), b_Tc=b_Tc, a_Pc=np.exp(ln_a_Pc), b_Pc=b_Pc,
                m_om=m_om, c_om=c_om)


_FIT = _fit_property_regression()


def pseudo_component_properties(M):
    """Tc (K), Pc (bar), omega for a paraffinic pseudo-component of molar
    mass M (g/mol), from the n=5 n-alkane regression in `_fit_property_regression`.

    This is an extrapolation (real C7+ cuts are naphthenic/aromatic blends,
    not pure n-alkanes, and M=150-300 lies outside the M=16-72 fit range),
    appropriate for a illustrative continuous-thermodynamics demo and
    explicitly *not* used for, or mixed with, the five validated systems.
    """
    f = _FIT
    Tc = f["a_Tc"] * M ** f["b_Tc"]
    Pc = f["a_Pc"] * M ** f["b_Pc"]
    omega = f["m_om"] * M + f["c_om"]
    return Tc, Pc, omega
