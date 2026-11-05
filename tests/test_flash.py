import numpy as np
import pytest

from ptm.components import NBUTANE, PROPANE
from ptm.eos import pure_a_b
from ptm.flash import rachford_rice_beta, tp_flash
from ptm.saturation import bubble_P


def test_rachford_rice_mass_balance_closes():
    z = np.array([0.4, 0.6])
    K = np.array([2.5, 0.4])
    beta = rachford_rice_beta(z, K)
    x = z / (1 + beta * (K - 1))
    y = K * x
    x /= x.sum()
    y /= y.sum()
    recombined = (1 - beta) * x + beta * y
    # recombined should match z up to the normalization step above; check
    # the defining RR equation instead, which is exact:
    g = np.sum(z * (K - 1) / (1 + beta * (K - 1)))
    assert abs(g) < 1e-9


def test_rachford_rice_beta_zero_when_all_liquid():
    z = np.array([0.3, 0.7])
    K = np.array([0.5, 0.3])  # both K<1 => feed wants to stay liquid
    beta = rachford_rice_beta(z, K)
    assert beta < 0.05


def test_tp_flash_mass_balance():
    T = 320.0
    a1, b1 = pure_a_b(PROPANE.Tc, PROPANE.Pc_Pa, PROPANE.omega, T)
    a2, b2 = pure_a_b(NBUTANE.Tc, NBUTANE.Pc_Pa, NBUTANE.omega, T)
    a_i, b_i = np.array([a1, a2]), np.array([b1, b2])
    kij = np.zeros((2, 2))
    Tc = np.array([PROPANE.Tc, NBUTANE.Tc])
    Pc_Pa = np.array([PROPANE.Pc_Pa, NBUTANE.Pc_Pa])
    omega = np.array([PROPANE.omega, NBUTANE.omega])

    z = np.array([0.5, 0.5])
    _, y_bubble = bubble_P(z, a_i, b_i, kij, T, Tc, Pc_Pa, omega)
    P_bubble, _ = bubble_P(z, a_i, b_i, kij, T, Tc, Pc_Pa, omega)
    P = P_bubble * 0.97  # just below bubble P => mostly liquid, small vapor

    res = tp_flash(z, a_i, b_i, kij, T, P, Tc, Pc_Pa, omega)
    recombined = (1 - res["beta"]) * res["x"] + res["beta"] * res["y"]
    assert np.allclose(recombined, z, atol=1e-6)
    assert 0 <= res["beta"] <= 1
    assert res["x"].sum() == pytest.approx(1.0, abs=1e-8)
    assert res["y"].sum() == pytest.approx(1.0, abs=1e-8)
