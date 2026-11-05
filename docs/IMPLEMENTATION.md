# Implementation

How each piece of [THEORY.md](THEORY.md) is actually coded: file, function,
numerical method, convergence criteria, edge cases, and the test(s) that
verify it. Read alongside the theory doc -- section numbers below
correspond to the same sections there.

---

## 1-2. Peng-Robinson EOS -- `ptm/components.py`, `ptm/eos.py`

- **`components.py`**: `Component` dataclass holding Tc, Pc, omega, Mw for
  methane, ethane, propane, n-butane, CO2, and n-pentane (the last used
  only as a regression anchor in `continuous.py`, Section 6 below). Tc/Pc
  are taken from this project's own NIST WebBook extraction
  (`data/raw/extract.py`), so the EOS's critical point and the reference
  curve used for validation agree *by construction* for the pure fluids.
  omega values are the standard literature constants (Poling et al.).

- **`eos.py:kappa(omega)`** -- the two piecewise alpha-function
  correlations (1976/1978), switching at omega=0.49. `pure_a_b` computes
  pure a_i(T), b_i.

- **`eos.py:mix_a_b(x, a_i, b_i, kij)`** -- vdW1 mixing rule. Returns
  `a_cross`, the per-component sum sum_j x_j sqrt(a_i a_j)(1-k_ij), which
  is exactly the term the fugacity-coefficient formula needs (avoiding
  recomputing it inside `ln_phi_mix`).

- **`eos.py:cubic_coeffs` / `real_roots`** -- builds the monic cubic
  [1, c2, c1, c0] and solves it with `numpy.roots` (companion-matrix
  eigenvalues -- robust and vectorization-friendly for a degree-3
  polynomial; a closed-form Cardano solution would be marginally faster
  but adds branch-handling complexity for no real benefit at this scale).
  Roots with |imag| > 1e-9 are discarded as numerical noise around a
  genuinely complex pair.

- **`eos.py:select_root` / `gibbs_departure`** -- implements Section 2.6.
  One real root: returned regardless of the requested phase (no ambiguity
  possible). Three real roots: outer two are liquid-/vapor-like; the
  `"stable"` mode picks whichever has lower `gibbs_departure`. **Design
  note:** the rest of the codebase (flash, saturation, stability) always
  passes an explicit `"liquid"` or `"vapor"` request, not `"stable"` --
  because those algorithms already know, from the iteration structure,
  which branch they're evaluating (e.g. the flash's liquid-phase fugacity
  call should always read the liquid root, even transiently away from
  convergence). `"stable"` exists for `pure_psat_newton`'s bracket search,
  where the code must actively detect when only one phase exists.

- **`eos.py:ln_phi_mix`** -- the fugacity-coefficient formula (Section
  2.5), vectorized over all components at once.

- **`eos.py:pure_psat_newton`** -- pure-component Psat by equal fugacity.
  **Numerical method (the one real bug fixed during development):** an
  initial successive-substitution version (`P <- P * phi_l/phi_v`)
  diverged near Tc, where the cubic can briefly lose its second real root
  mid-iteration. Replaced with bracketed bisection on ln(phi_l/phi_v),
  which is well-behaved given a correct bracket. The bracket search itself
  needed two further fixes, both still visible in the function's comments
  as the reasoning for the final approach:
  1. A single log-spaced P-grid from near 0 to near Pc has *very* uneven
     resolution; for T close to Tc, the two-real-root window is a narrow
     sliver near Pc that a low-to-high log grid barely samples. Fixed by
     taking the **union** of two grids: one log-spaced from low P upward,
     one log-spaced in (Pc-P) from Pc downward -- between them, both the
     "window near 0" (low T) and "window near Pc" (T near Tc) cases get
     resolved.
  2. Points with only one real root carry no equal-fugacity information
     (both phase calls degenerate to the same Z) and must be explicitly
     excluded from the bracket search, not just treated as "f=0" --
     otherwise the scan wrongly reports convergence at a P with no real
     phase split.
  - **Edge case:** within ~1-2 K of Tc, the two-root window becomes
    narrower than the scan can resolve, and the function raises
    `RuntimeError` rather than silently returning nonsense --
    `validation.py` and `uncertainty.py` both exclude a ~1.5 K margin
    around each fluid's Tc for exactly this reason.
  - **Tests:** `test_pure_psat_matches_nist_within_2_percent`,
    `test_cubic_has_three_real_roots_below_tc_near_saturation` /
    `..._one_real_root_above_tc`, `test_fugacity_coefficients_equal_at_...`.

---

## 3. Stability -- `ptm/stability.py`

`wilson_k` (Section 4.1). `tpd_stationary` does successive substitution
Y_i <- exp(ln z_i + ln phi_i(z) - ln phi_i(y)) to a stationary point of the
Michelsen tm(Y) functional; `is_stable` tries both a vapor-like and a
liquid-like initial guess and declares instability if either stationary
point's tm < -tol. **Not currently wired into the ternary tie-line
generator's accept/reject logic** (`ternary.py` instead filters by
converged flash beta and composition spread -- see below); a natural
extension would be to call `is_stable` before attempting a flash at all,
which is the textbook-correct order of operations and is noted as a
possible follow-on.

---

## 4. Flash and saturation -- `ptm/flash.py`, `ptm/saturation.py`

- **`flash.py:rachford_rice_beta`** -- safeguarded Newton-within-bisection
  on g(beta); always stays inside the open interval (lo, hi), so it can't
  diverge even if a Newton step overshoots. **Test:**
  `test_rachford_rice_mass_balance_closes` checks g(beta_solution) ~ 0
  directly; `test_rachford_rice_beta_zero_when_all_liquid` checks the
  boundary case where every K<1.

- **`flash.py:tp_flash`** -- two-level loop: inner Rachford-Rice for beta
  at fixed K, outer successive substitution updating K from the EOS.
  **Test:** `test_tp_flash_mass_balance` checks the recombined
  (1-beta)x + beta*y reproduces the feed z to 1e-6, and that x, y each
  sum to 1.

- **`saturation.py:bubble_P` / `dew_P`** -- same successive-substitution
  pattern specialized to beta=0/beta=1. `pxy_diagram` sweeps x1 and calls
  `bubble_P` at each point, reusing the previous point's converged P as
  the next initial guess (continuation, not a fresh guess every time --
  substantially fewer iterations near the middle of the diagram where the
  curve is smooth).

---

## 5. Ternary tie lines -- `ptm/ternary.py`

`ternary_tie_lines` sweeps feed composition along lines of constant
(heaviest-component) z3 and flashes each feed with `tp_flash`. A feed is
accepted as a genuine two-phase tie line only if **both** 0.02 < beta <
0.98 **and** the liquid/vapor compositions differ by more than 0.03 in at
least one component -- the second check guards against `tp_flash`
converging to a spurious beta when the true physical answer is a single
stable phase (K_i -> 1 for every i, so Rachford-Rice's denominator becomes
near-degenerate). This is a pragmatic filter, not a stability-tested
guarantee; see the stability.py note above.

---

## 6. Continuous thermodynamics -- `ptm/continuous.py`

- **`gamma_params_from_moments`** -- closed-form alpha, beta from a target
  mean and variance (Section 5).
- **`gauss_laguerre_pseudo_components`** -- `scipy.special.roots_genlaguerre
  (q, alpha-1)` gives the generalized-Laguerre nodes/weights directly;
  nodes are rescaled `eta + beta*t_i`, weights normalized by
  `1/Gamma(alpha)`. **Verified, not just asserted:** `check_moments`
  recomputes the weighted mean and variance from the discrete (nodes,
  weights) and compares to the target distribution's mean/variance;
  `test_quadrature_reproduces_mean_and_variance` checks this holds for
  q = 2, 3, 5, 8 (Section 5's exactness claim is for *any* q>=2, and the
  test actually exercises that range rather than a single q).
- **`pseudo_component_properties(M)`** -- **deliberately not** an
  externally-quoted correlation (an earlier draft used a half-remembered
  Riazi-Daubert-style formula that produced a critical temperature of
  ~1600 K for a C7-like pseudo-component -- obviously wrong, caught by a
  manual sanity check, and replaced). The current version fits its own
  log-log regression of Tc, Pc vs. molar mass (and a linear fit of omega
  vs. mass) on the five real n-alkanes this project already has
  literature constants for (methane through n-pentane), then extrapolates
  to the C7+ pseudo-component range. This is a real, if small (n=5),
  regression on real data -- openly extrapolated well outside its fit
  range, which is disclosed in the function's docstring rather than
  presented as a validated correlation.

---

## 7. Monte Carlo uncertainty -- `ptm/uncertainty.py`

`mc_psat_band` draws `n_samples` independent (Tc, Pc, omega) triples from
independent normals, calls `pure_psat_newton` for every sample at every
requested temperature, and reports the median plus the requested
percentile band (95% by default) at each T. Samples where the drawn Tc
happens to fall below the requested T, or where the Psat solver fails to
bracket, are set to NaN and excluded via `np.nanpercentile`/`nanmedian`
rather than crashing the whole batch. **Performance note:** this calls the
(now-fast, ~2-5 ms) `pure_psat_newton` up to `n_samples * len(T_grid)`
times, which is why the CLI defaults to a modest 100-150 samples rather
than the thousands a production uncertainty study might use.

---

## 8. Fitting and validation -- `ptm/fitting.py`, `ptm/validation.py`

- **`fitting.py:fit_omega_mle`** -- `scipy.optimize.minimize_scalar`
  (bounded, omega in [-0.3, 1.0]) minimizing sum of squared log-residuals
  (Section 7.1); the 1/sqrt(Fisher information) standard error (Section
  7.2) uses a forward-difference Jacobian with step h=1e-4.
  **Test:** `test_mle_recovers_known_omega_from_synthetic_noiseless_data`
  generates "data" from the model itself at a known omega and checks the
  fit recovers it to 1e-3 -- this isolates the optimizer/likelihood
  machinery from any question about whether the EOS fits reality, which
  the *other* validation tests cover.

- **`validation.py:validate_all`** -- loads each fluid's
  `data/vle/*_psat.csv`, restricts to [250, 400] K intersected with that
  fluid's own subcritical range (methane's Tc=190.6 K never reaches 250 K,
  so it falls back to its full available range below Tc), splits
  alternating points into train/test, and reports AAD% at both the
  literature omega and the MLE-refit omega. **Reproduce the headline
  numbers:**

  ```
  python -m ptm validate
  ```

  gives (as of this run; also written to `results/validation_table.csv`):

  | component | n_train | n_test | omega (lit) | omega (fit) | AAD% (lit omega) | AAD% (fit omega) |
  |---|---|---|---|---|---|---|
  | methane | 21 | 20 | 0.0115 | 0.0142 | 0.692 | 0.469 |
  | ethane | 15 | 15 | 0.0995 | 0.1062 | 0.480 | 0.283 |
  | propane | 20 | 19 | 0.1523 | 0.1527 | 0.358 | 0.357 |
  | n-butane | 14 | 14 | 0.2002 | 0.2013 | 0.456 | 0.457 |
  | carbon dioxide | 17 | 17 | 0.2236 | 0.2188 | 0.265 | 0.207 |

  Every fluid is under 0.7% AAD on held-out points even with the
  off-the-shelf literature omega -- the MLE refit (correctly) buys only a
  small further improvement, which is itself informative: it says the
  literature omega values are already close to locally optimal for
  reproducing *this* EOS's Psat curve, exactly what you'd hope from
  well-characterized constants for common fluids.

---

## 9. Diagrams and CLI -- `ptm/diagrams.py`, `ptm/cli.py`

All plotting is isolated in `diagrams.py` (matplotlib, `Agg` backend so it
runs headless) so none of the physics modules import matplotlib. `cli.py`
defines the five validated binary systems (`BINARIES` dict, with each
pair's T and k_ij) and wires `argparse` subcommands to the corresponding
function in each physics module; `all` regenerates every figure and the
validation table in one call. Every mixture-diagram command's plot title
includes the k_ij value used, and the methane-propane-n-butane ternary
title is explicitly labeled "(PR EOS, predictive)" -- the honesty
disclosure from README.md is repeated in the artifacts themselves, not
just the documentation.

---

## Reproducing everything end to end

```
pip install -r requirements.txt
pytest tests/ -q                 # 15 tests, all physics/statistics claims above
python -m ptm all                # regenerates every figure + results/validation_table.csv
```
