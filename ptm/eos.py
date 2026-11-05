"""
Peng-Robinson (1976/1978) cubic equation of state.

    P = RT/(v-b) - a*alpha(T) / (v^2 + 2*b*v - b^2)

Mixtures use the van der Waals one-fluid (vdW1) mixing rules with a single
symmetric binary interaction parameter k_ij per pair:

    a_mix = sum_i sum_j x_i x_j sqrt(a_i a_j) (1 - k_ij)
    b_mix = sum_i x_i b_i

All thermodynamic state functions here (Z, ln(phi_i), molar Gibbs energy)
follow directly from this one closure; see docs/THEORY.md section 3 for the
derivation of the fugacity-coefficient formula used in `ln_phi`.

Units: SI throughout this module (Pa, K, m^3/mol, J/mol). The CLI/diagrams
layer converts to/from bar for display.
"""
from __future__ import annotations

import numpy as np

R = 8.314462618  # J/(mol K)

SQRT2 = np.sqrt(2.0)


def kappa(omega: float) -> float:
    """PR alpha-function slope parameter.

    1976 correlation for omega <= 0.49; 1978 extension (Robinson & Peng,
    1978 report) for heavier/more acentric species, which keeps alpha(T)
    well-behaved up to omega ~ 1 instead of turning upward above Tr ~ 0.7.
    """
    if omega <= 0.49:
        return 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    return (0.379642 + 1.48503 * omega - 0.164423 * omega**2
            + 0.016666 * omega**3)


def pure_a_b(Tc: float, Pc_Pa: float, omega: float, T: float):
    """Pure-component a_i(T) and b_i (SI units, P in Pa)."""
    ac = 0.45724 * R**2 * Tc**2 / Pc_Pa
    b = 0.07780 * R * Tc / Pc_Pa
    k = kappa(omega)
    Tr = T / Tc
    alpha = (1.0 + k * (1.0 - np.sqrt(Tr))) ** 2
    return ac * alpha, b


def mix_a_b(x, a_i, b_i, kij):
    """vdW1 mixing rules.

    x, a_i, b_i: 1-D arrays of length n.
    kij: symmetric (n, n) array, zero diagonal.
    Returns a_mix, b_mix, and the n-vector a_i_bar_j := sum_j x_j sqrt(a_i a_j)(1-kij),
    which is the quantity each component's fugacity coefficient needs (it is the
    partial derivative d(n^2 a_mix)/d(n_i), divided by n).
    """
    x = np.asarray(x, dtype=float)
    a_i = np.asarray(a_i, dtype=float)
    b_i = np.asarray(b_i, dtype=float)
    sqrt_aa = np.sqrt(np.outer(a_i, a_i))
    A_ij = sqrt_aa * (1.0 - kij)
    a_cross = A_ij @ x          # length n: sum_j x_j sqrt(a_i a_j)(1-k_ij)
    a_mix = x @ a_cross
    b_mix = float(x @ b_i)
    return float(a_mix), b_mix, a_cross


def cubic_coeffs(a_mix: float, b_mix: float, P: float, T: float):
    """Return (A, B) dimensionless cubic parameters and the monic cubic
    coefficients [1, c2, c1, c0] for Z^3 + c2 Z^2 + c1 Z + c0 = 0."""
    A = a_mix * P / (R * T) ** 2
    B = b_mix * P / (R * T)
    c2 = -(1.0 - B)
    c1 = A - 3.0 * B**2 - 2.0 * B
    c0 = -(A * B - B**2 - B**3)
    return A, B, (1.0, c2, c1, c0)


def real_roots(coeffs) -> np.ndarray:
    """Real roots of the monic cubic, sorted ascending."""
    roots = np.roots(coeffs)
    real = np.array([r.real for r in roots if abs(r.imag) < 1e-9])
    return np.sort(real)


def gibbs_departure(Z: float, A: float, B: float) -> float:
    """Dimensionless residual molar Gibbs energy g^R/RT for a *pure* fluid
    root, used only to pick between liquid-like and vapor-like roots when
    the cubic has three real roots (see `select_root`). From the PR EOS:

        g^R/RT = Z - 1 - ln(Z - B) - A/(2*sqrt(2)*B) * ln[(Z+(1+sqrt2)B)/(Z+(1-sqrt2)B)]

    The physically stable root at given (T, P) is the one with the lower
    Gibbs energy; this is what `select_root` evaluates.
    """
    if Z <= B:
        return np.inf
    term = (Z + (1 + SQRT2) * B) / (Z + (1 - SQRT2) * B)
    return (Z - 1.0) - np.log(Z - B) - A / (2 * SQRT2 * B) * np.log(term)


def select_root(roots: np.ndarray, A: float, B: float, phase: str) -> float:
    """Pick the physically correct compressibility factor Z from the real
    roots of the PR cubic.

    One real root (single-phase region, T > Tc or far from saturation):
    that root is returned regardless of the requested `phase`.

    Three real roots (inside the two-root "dome", below Tc near the
    saturation curve): the smallest root is liquid-like, the largest is
    vapor-like, and the middle root is never physical (dP/dV > 0 there).
    When BOTH outer roots are mechanically stable, the thermodynamically
    *stable* one is the one with the lower residual Gibbs energy
    g^R/RT (`gibbs_departure`) -- that is the root nature actually selects,
    and it is what decides, e.g., exactly which side of a first-order
    phase transition a given (T, P) sits on when `phase` is ambiguous.

    `phase` is one of "liquid", "vapor", or "stable" (let Gibbs energy decide).
    """
    roots = roots[roots > B + 1e-12]
    if len(roots) == 0:
        raise ValueError("No mechanically stable root (Z > B) found.")
    if len(roots) == 1:
        return float(roots[0])

    Zl, Zv = float(roots[0]), float(roots[-1])
    if phase == "liquid":
        return Zl
    if phase == "vapor":
        return Zv
    if phase == "stable":
        gl = gibbs_departure(Zl, A, B)
        gv = gibbs_departure(Zv, A, B)
        return Zl if gl <= gv else Zv
    raise ValueError(f"phase must be 'liquid', 'vapor', or 'stable', got {phase!r}")


def ln_phi_mix(x, a_i, b_i, kij, T: float, P: float, phase: str):
    """Fugacity coefficients ln(phi_i) for every component in a mixture of
    composition x at (T, P), selecting the liquid- or vapor-like root.

    Returns (Z, ln_phi array).
    """
    a_mix, b_mix, a_cross = mix_a_b(x, a_i, b_i, kij)
    A, B, coeffs = cubic_coeffs(a_mix, b_mix, P, T)
    roots = real_roots(coeffs)
    Z = select_root(roots, A, B, phase)

    b_i = np.asarray(b_i, dtype=float)
    term = (Z + (1 + SQRT2) * B) / (Z + (1 - SQRT2) * B)
    ln_phi = (b_i / b_mix) * (Z - 1.0) - np.log(Z - B) \
        - A / (2 * SQRT2 * B) * (2.0 * a_cross / a_mix - b_i / b_mix) * np.log(term)
    return Z, ln_phi


def pure_psat_newton(Tc: float, Pc_Pa: float, omega: float, T: float,
                      tol: float = 1e-10, max_iter: int = 200) -> float:
    """Pure-component saturation pressure at T from equal fugacity
    (ln phi_liquid == ln phi_vapor), using the EOS itself -- no external
    vapor-pressure correlation.

    Bracketed bisection on ln(phi_l/phi_v), not successive substitution:
    close to Tc the naive P <- P*ratio update overshoots past the point
    where the cubic stops having three real roots and diverges. The log
    fugacity ratio is monotonic in P between a vanishingly small pressure
    (ratio >> 1, all-liquid side) and Pc (ratio < 1), so bisection on
    [P_lo, Pc) always converges.
    """
    a_mix, b_mix = a, b = pure_a_b(Tc, Pc_Pa, omega, T)
    x = np.array([1.0])
    kij = np.zeros((1, 1))

    def log_ratio(P: float):
        """Returns (ln phi_l - ln phi_v, has_two_roots); see class
        docstring note above on why single-root points are excluded from
        bracketing. Computes the cubic roots once per call (not per phase).
        """
        A, B, coeffs = cubic_coeffs(a_mix, b_mix, P, T)
        roots = real_roots(coeffs)
        valid = roots[roots > B + 1e-12]
        if len(valid) < 2:
            return 0.0, False
        Zl, Zv = float(valid[0]), float(valid[-1])
        lnphi_l = (b / b_mix) * (Zl - 1.0) - np.log(Zl - B) - A / (2 * SQRT2 * B) * np.log(
            (Zl + (1 + SQRT2) * B) / (Zl + (1 - SQRT2) * B))
        lnphi_v = (b / b_mix) * (Zv - 1.0) - np.log(Zv - B) - A / (2 * SQRT2 * B) * np.log(
            (Zv + (1 + SQRT2) * B) / (Zv + (1 - SQRT2) * B))
        return lnphi_l - lnphi_v, True

    # Log-spaced scan, refined near the top, to find a bracket [P_lo, P_hi]
    # where both ends genuinely have a liquid and a vapor root and the
    # sign of f differs (a single call to np.vectorize-free loop; kept
    # small since this runs inside optimizer/MC loops that call Psat many
    # times -- see fitting.py, uncertainty.py).
    # The two-real-root window can sit anywhere from near P=0 (low T, far
    # below Tc) to a narrow sliver just under Pc (T close to Tc), and its
    # width shrinks to zero as T -> Tc. One log-spaced grid resolves the
    # low-P case; a mirrored grid, log-spaced in (Pc - P), resolves the
    # high-P case. Their union covers both without needing thousands of
    # points everywhere.
    grid_lo = np.geomspace(1e-6 * Pc_Pa, 0.999999 * Pc_Pa, 250)
    grid_hi = Pc_Pa - np.geomspace(1e-6 * Pc_Pa, 0.999999 * Pc_Pa, 250)
    P_scan = np.sort(np.unique(np.concatenate([grid_lo, grid_hi])))
    P_scan = P_scan[P_scan > 0]
    P_lo = P_hi = f_lo = f_hi = None
    prev_P, prev_f = None, None
    for P in P_scan:
        f, two = log_ratio(P)
        if not two:
            prev_P, prev_f = None, None
            continue
        if prev_f is not None and prev_f * f <= 0:
            P_lo, P_hi, f_lo, f_hi = prev_P, P, prev_f, f
            break
        prev_P, prev_f = P, f
    if P_lo is None:
        raise RuntimeError(f"Could not bracket Psat at T={T} K: no sign change found in scan.")
    for _ in range(max_iter):
        P_mid = 0.5 * (P_lo + P_hi)
        f_mid, _ = log_ratio(P_mid)
        if abs(f_mid) < tol or (P_hi - P_lo) / P_mid < 1e-12:
            return P_mid
        if f_mid * f_lo > 0:
            P_lo, f_lo = P_mid, f_mid
        else:
            P_hi, f_hi = P_mid, f_mid
    return 0.5 * (P_lo + P_hi)
