import numpy as np

from ptm.continuous import (check_moments, gamma_params_from_moments,
                             gauss_laguerre_pseudo_components,
                             pseudo_component_properties)


def test_quadrature_reproduces_mean_and_variance():
    mean, var, eta = 180.0, 2500.0, 90.0
    alpha, beta = gamma_params_from_moments(mean, var, eta)
    for q in (2, 3, 5, 8):
        nodes, weights = gauss_laguerre_pseudo_components(alpha, beta, eta, q)
        assert check_moments(nodes, weights, mean, var, rtol=0.02)
        assert abs(weights.sum() - 1.0) < 1e-6


def test_weights_are_nonnegative_and_nodes_above_eta():
    alpha, beta = gamma_params_from_moments(150.0, 1200.0, 90.0)
    nodes, weights = gauss_laguerre_pseudo_components(alpha, beta, 90.0, 4)
    assert np.all(weights >= 0)
    assert np.all(nodes > 90.0)


def test_pseudo_component_properties_decrease_pc_with_mass():
    Tc1, Pc1, om1 = pseudo_component_properties(100.0)
    Tc2, Pc2, om2 = pseudo_component_properties(300.0)
    assert Tc2 > Tc1       # heavier -> higher critical temperature
    assert Pc2 < Pc1       # heavier -> lower critical pressure
    assert om2 > om1       # heavier -> more acentric
