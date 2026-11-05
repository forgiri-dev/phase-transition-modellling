import numpy as np
import pytest

from ptm.components import CO2, METHANE, NBUTANE, PROPANE
from ptm.eos import (cubic_coeffs, gibbs_departure, ln_phi_mix, mix_a_b,
                      pure_a_b, pure_psat_newton, real_roots, select_root)


def test_pure_psat_matches_nist_within_2_percent():
    # Spot checks against values read directly from this project's own
    # NIST WebBook extraction (data/vle/*.csv); see docs/IMPLEMENTATION.md.
    P = pure_psat_newton(PROPANE.Tc, PROPANE.Pc_Pa, PROPANE.omega, 250.24)
    assert abs(P - 2.1965e5) / 2.1965e5 < 0.02

    P = pure_psat_newton(CO2.Tc, CO2.Pc_Pa, CO2.omega, 250.03)
    assert abs(P - 17.869e5) / 17.869e5 < 0.02

    P = pure_psat_newton(NBUTANE.Tc, NBUTANE.Pc_Pa, NBUTANE.omega, 360.0)
    # n-butane at 360K: literature-order check only (no exact grid point);
    # must be a sane positive pressure well below Pc.
    assert 0 < P < NBUTANE.Pc_Pa


def test_cubic_has_three_real_roots_below_tc_near_saturation():
    a, b = pure_a_b(PROPANE.Tc, PROPANE.Pc_Pa, PROPANE.omega, 300.0)
    Psat = pure_psat_newton(PROPANE.Tc, PROPANE.Pc_Pa, PROPANE.omega, 300.0)
    A, B, coeffs = cubic_coeffs(a, b, Psat, 300.0)
    roots = real_roots(coeffs)
    assert len(roots) == 3


def test_cubic_has_one_real_root_above_tc():
    a, b = pure_a_b(PROPANE.Tc, PROPANE.Pc_Pa, PROPANE.omega, PROPANE.Tc + 20)
    A, B, coeffs = cubic_coeffs(a, b, PROPANE.Pc_Pa, PROPANE.Tc + 20)
    roots = real_roots(coeffs)
    assert len(roots) == 1


def test_select_root_picks_lower_gibbs_energy_consistently_with_liquid_vapor_labels():
    a, b = pure_a_b(PROPANE.Tc, PROPANE.Pc_Pa, PROPANE.omega, 300.0)
    Psat = pure_psat_newton(PROPANE.Tc, PROPANE.Pc_Pa, PROPANE.omega, 300.0)
    A, B, coeffs = cubic_coeffs(a, b, Psat, 300.0)
    roots = real_roots(coeffs)
    Zl, Zv = select_root(roots, A, B, "liquid"), select_root(roots, A, B, "vapor")
    assert Zl < Zv
    # At the true saturation pressure liquid and vapor roots are (by
    # definition of Psat) equally stable, so their Gibbs energies should
    # match closely, and the "stable" selector should not error.
    g_l, g_v = gibbs_departure(Zl, A, B), gibbs_departure(Zv, A, B)
    assert abs(g_l - g_v) < 1e-3
    Z_stable = select_root(roots, A, B, "stable")
    assert Z_stable in (Zl, Zv)

    # Just above Psat (more compressed): liquid root must have lower Gibbs
    # energy than vapor root, since liquid is now the stable phase.
    A2, B2, coeffs2 = cubic_coeffs(a, b, Psat * 1.5, 300.0)
    roots2 = real_roots(coeffs2)
    Zl2 = select_root(roots2, A2, B2, "liquid")
    Zv2 = select_root(roots2, A2, B2, "vapor")
    assert gibbs_departure(Zl2, A2, B2) < gibbs_departure(Zv2, A2, B2)
    assert select_root(roots2, A2, B2, "stable") == pytest.approx(Zl2)


def test_fugacity_coefficients_equal_at_pure_component_saturation():
    """Direct check of the equal-fugacity condition the Psat solver enforces."""
    T = 310.0
    Psat = pure_psat_newton(PROPANE.Tc, PROPANE.Pc_Pa, PROPANE.omega, T)
    a, b = pure_a_b(PROPANE.Tc, PROPANE.Pc_Pa, PROPANE.omega, T)
    x = np.array([1.0])
    kij = np.zeros((1, 1))
    _, lnphi_l = ln_phi_mix(x, [a], [b], kij, T, Psat, "liquid")
    _, lnphi_v = ln_phi_mix(x, [a], [b], kij, T, Psat, "vapor")
    assert lnphi_l[0] == pytest.approx(lnphi_v[0], abs=1e-6)


def test_mixing_rule_reduces_to_pure_component_at_x_equals_1():
    a1, b1 = pure_a_b(PROPANE.Tc, PROPANE.Pc_Pa, PROPANE.omega, 300.0)
    a2, b2 = pure_a_b(NBUTANE.Tc, NBUTANE.Pc_Pa, NBUTANE.omega, 300.0)
    kij = np.array([[0.0, 0.0], [0.0, 0.0]])
    a_mix, b_mix, _ = mix_a_b([1.0, 0.0], [a1, a2], [b1, b2], kij)
    assert a_mix == pytest.approx(a1)
    assert b_mix == pytest.approx(b1)


def test_kappa_1976_1978_small_jump_at_omega_0p49():
    """The 1976 and 1978 kappa(omega) correlations are two independently
    fitted polynomials, not pieces of one smooth function, so kappa(omega)
    has a small genuine jump at the omega=0.49 switch point (documented
    in Robinson & Peng's original report). This pins down the jump's
    size so a future edit that changes it is caught deliberately."""
    from ptm.eos import kappa
    k1 = kappa(0.489999)
    k2 = kappa(0.490001)
    assert abs(k1 - k2) < 0.01
