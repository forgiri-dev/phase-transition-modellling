"""
Maximum-likelihood parameter estimation, demonstrated on real data: refit
each pure component's PR acentric factor omega against its own NIST
saturation-pressure curve (data/vle/*_psat.csv), rather than trusting the
literature omega outright.

Model: ln P_sat(T; omega) from the EOS (ptm.eos.pure_psat_newton).
Likelihood: residuals r_i = ln P_i^obs - ln P_i^model(omega) are assumed
i.i.d. N(0, sigma^2) (homoscedastic in log-pressure -- reasonable since
vapor pressure spans orders of magnitude). Under that model the MLE of
omega is exactly the least-squares estimate, and the large-sample standard
error is

    se(omega_hat) = sigma_hat / sqrt(sum_i (d ln P_i / d omega)^2)

computed here from a finite-difference Jacobian and the Gauss-Newton
approximation to the Hessian of the negative log-likelihood (see
docs/THEORY.md section 6 for the derivation).

Train/test split: fit on every other data point, report AAD% on the
held-out half -- so the <5% deviation claim is checked on points the fit
never saw.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .eos import pure_psat_newton


def _model_lnP(T, Tc, Pc_Pa, omega):
    return np.log(pure_psat_newton(Tc, Pc_Pa, omega, T))


def fit_omega_mle(T_train, P_train_Pa, Tc, Pc_Pa, omega0, bounds=(-0.3, 1.0)):
    """MLE (= least squares on ln P) refit of omega against training data.

    Returns dict: omega_hat, sigma_hat (residual std in ln P), se_omega
    (standard error), and the per-point residuals.
    """
    T_train = np.asarray(T_train, dtype=float)
    lnP_obs = np.log(np.asarray(P_train_Pa, dtype=float))

    def sse(omega):
        lnP_model = np.array([_model_lnP(T, Tc, Pc_Pa, omega) for T in T_train])
        return np.sum((lnP_obs - lnP_model) ** 2)

    res = minimize_scalar_bounded(sse, bounds)
    omega_hat = res
    lnP_model = np.array([_model_lnP(T, Tc, Pc_Pa, omega_hat) for T in T_train])
    resid = lnP_obs - lnP_model
    n = len(T_train)
    sigma_hat = np.sqrt(np.sum(resid**2) / max(n - 1, 1))

    h = 1e-4
    lnP_plus = np.array([_model_lnP(T, Tc, Pc_Pa, omega_hat + h) for T in T_train])
    jac = (lnP_plus - lnP_model) / h
    fisher_info = np.sum(jac**2) / sigma_hat**2
    se_omega = 1.0 / np.sqrt(fisher_info) if fisher_info > 0 else np.nan

    return {"omega_hat": omega_hat, "sigma_hat": sigma_hat, "se_omega": se_omega,
            "residuals": resid, "n_train": n}


def minimize_scalar_bounded(f, bounds):
    res = minimize_scalar(f, bounds=bounds, method="bounded")
    return float(res.x)


def aad_percent(T_test, P_test_Pa, Tc, Pc_Pa, omega):
    """Average absolute deviation (%) of EOS-predicted vs. observed Psat
    on held-out points."""
    T_test = np.asarray(T_test, dtype=float)
    P_test_Pa = np.asarray(P_test_Pa, dtype=float)
    preds = np.array([pure_psat_newton(Tc, Pc_Pa, omega, T) for T in T_test])
    return float(np.mean(np.abs(preds - P_test_Pa) / P_test_Pa) * 100.0)
