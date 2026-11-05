# Theory

The academic content behind this project, in the order it's used: phase
equilibrium thermodynamics, the Peng-Robinson equation of state, phase
stability and flash, then the three places continuous-random-variable /
statistical theory enters. Companion document:
[IMPLEMENTATION.md](IMPLEMENTATION.md) maps every section here to actual
code.

---

## 1. Phase equilibrium thermodynamics

### 1.1 The equilibrium condition

A closed system at fixed temperature T and pressure P reaches equilibrium
when its total Gibbs energy is minimized. For a system that can split into
a liquid phase (superscript L) and a vapor phase (V), each containing n
components, the minimum occurs when, for every component i:

    T^L = T^V            (thermal equilibrium)
    P^L = P^V             (mechanical equilibrium)
    mu_i^L = mu_i^V        (chemical equilibrium)

where mu_i is the chemical potential of component i. Chemical potential is
inconvenient to compute directly from an equation of state, so it's
rewritten in terms of **fugacity** f_i, defined so that mu_i = mu_i^ref +
RT ln(f_i/f_i^ref). The equilibrium condition becomes the much more usable

    f_i^L = f_i^V   for every component i                      (1)

Fugacity is further written as f_i = x_i * phi_i * P (liquid) or y_i *
phi_i * P (vapor), where phi_i is the **fugacity coefficient** -- the
ratio of real to ideal-gas behavior, computable directly from any PVT
equation of state. Equation (1) becomes

    x_i * phi_i^L(T,P,x) = y_i * phi_i^V(T,P,y)   for every i      (2)

Everything from here down exists to evaluate phi_i and solve (2).

### 1.2 Gibbs phase rule

For a system with C components and P_h phases, the number of independent
intensive variables is F = C - P_h + 2. A binary system (C=2) in two-phase
equilibrium (P_h=2) has F=2 degrees of freedom: fixing T and P fully
determines both phase compositions. That's why a binary Pxy diagram at
fixed T is a complete, well-posed object (Section 4), and why a ternary
system (C=3, F=3) needs T, P, *and* overall composition specified before a
flash has a unique answer.

---

## 2. The Peng-Robinson cubic equation of state

### 2.1 Why a cubic EOS

The ideal gas law (PV=nRT) cannot predict phase separation at all --
there's no mechanism in it for a liquid branch. A **cubic** equation of
state is the simplest PVT relation whose isotherms can have the S-shaped
("van der Waals loop") curve needed to represent both a liquid and a vapor
root at the same T and P. Peng & Robinson (1976) [1] improved on the
earlier van der Waals and Redlich-Kwong-Soave cubics specifically for
better liquid-density and vapor-pressure predictions near the critical
point, which is why it's the standard choice for hydrocarbon and CO2
systems like the ones here.

### 2.2 The equation

    P = RT/(v - b) - a*alpha(T) / (v^2 + 2bv - b^2)

Pure-component parameters, fixed by requiring (dP/dv)_Tc = (d2P/dv2)_Tc =
0 at the critical point:

    a_c = 0.45724 R^2 Tc^2 / Pc
    b   = 0.07780 R Tc / Pc

Temperature dependence enters through alpha(T), designed to reproduce the
correct pure-component vapor pressure away from Tc:

    alpha(T) = [1 + kappa (1 - sqrt(T/Tc))]^2
    kappa = 0.37464 + 1.54226*omega - 0.26992*omega^2          (omega <= 0.49)
    kappa = 0.379642 + 1.48503*omega - 0.164423*omega^2 + 0.016666*omega^3   (omega > 0.49, 1978 extension)

omega is the **acentric factor**, Pitzer's empirical measure of how far a
molecule's shape departs from a perfect (spherical, nonpolar) fluid;
omega=0 for a reference fluid like argon, and increases with molecular
size/asymmetry (methane: 0.011; n-butane: 0.200; see `ptm/components.py`).
The 1978 kappa correlation [2] extends validity to heavier/more acentric
species (needed here for the synthetic C7+ pseudo-components, which reach
omega > 1 at the heavy end -- Section 6).

### 2.3 Mixtures: van der Waals one-fluid mixing rules

A mixture's a and b are combined from the pure values via

    a_mix = sum_i sum_j x_i x_j sqrt(a_i a_j) (1 - k_ij)
    b_mix = sum_i x_i b_i

k_ij is the **binary interaction parameter**, a single empirical
correction per pair that captures how far real cross-interactions depart
from the geometric-mean approximation sqrt(a_i a_j). For chemically similar
nonpolar molecules (e.g. two n-alkanes) k_ij ~ 0 works well; for pairs with
a large size or polarity mismatch (e.g. CO2, which has a significant
quadrupole moment, with a paraffin) a nonzero, usually positive, k_ij is
needed. **k_ij must come from fitting real mixture VLE data** -- this
project could not source that data for the systems studied (see README.md)
and uses k_ij=0 for hydrocarbon pairs, k_ij=0.13 (an unfitted placeholder)
for CO2 pairs.

### 2.4 Cubic form and roots

Substituting the mixing rule into the EOS and writing it in terms of the
compressibility factor Z = Pv/RT gives a cubic polynomial

    Z^3 - (1-B)Z^2 + (A - 3B^2 - 2B)Z - (AB - B^2 - B^3) = 0
    A = a_mix P / (RT)^2,   B = b_mix P / (RT)

This is the **same geometric fact** as the van der Waals loop in Section
2.1: below Tc and near the saturation pressure, the cubic has **three real
roots** (smallest = liquid-like, largest = vapor-like, middle = always
unphysical, since dP/dv>0 there violates mechanical stability). Above Tc,
or far from saturation, only **one real root** exists. When three real
roots exist, the thermodynamically stable one is whichever has the lower
molar Gibbs energy -- see Section 2.6.

### 2.5 Fugacity coefficients

Differentiating the EOS's residual Helmholtz energy with respect to mole
number gives, for each component i in a mixture:

    ln(phi_i) = (b_i/b_mix)(Z-1) - ln(Z-B)
                - A/(2*sqrt2*B) * [2*sum_j x_j sqrt(a_i a_j)(1-k_ij)/a_mix - b_i/b_mix]
                * ln[(Z + (1+sqrt2)B) / (Z + (1-sqrt2)B)]

This is the quantity equation (2) is actually solved in terms of.

### 2.6 Choosing between roots: residual Gibbs energy

For a pure fluid, the EOS's residual molar Gibbs energy is

    g^R/RT = Z - 1 - ln(Z-B) - A/(2*sqrt2*B) * ln[(Z+(1+sqrt2)B)/(Z+(1-sqrt2)B)]

When the cubic has three real roots, the physically realized phase at that
(T,P) is the one with the **lower** g^R (lower total Gibbs energy = more
stable). Below the true saturation pressure, the vapor root has lower g^R;
above it, the liquid root does; exactly at saturation, they're equal --
which is precisely the equal-fugacity condition (2) again, viewed from the
Gibbs-energy side instead of the fugacity side. See
`ptm/eos.py:select_root` / `gibbs_departure` and
`tests/test_eos.py::test_select_root_picks_lower_gibbs_energy_...`.

---

## 3. Phase stability: the tangent-plane distance test

Solving equation (2) presumes you already know the system splits into two
phases. For a feed of unknown phase count, Michelsen's tangent-plane
distance (TPD) test [3] answers that question first. For a trial phase
composition y, define

    tpd(y) = sum_i y_i [ln y_i + ln phi_i(y) - ln z_i - ln phi_i(z)]

This is the vertical distance between the actual Gibbs-energy surface and
the plane tangent to it at the feed composition z. If **any** trial y
gives tpd(y) < 0, the single phase at z is unstable (lower Gibbs energy is
reachable by splitting), and the feed must flash into two phases.
Michelsen reformulates this in unnormalized mole numbers Y (avoiding a
spurious minimum at Y=z) and finds stationary points by successive
substitution from vapor-like and liquid-like initial guesses (Wilson
K-values, Section 4.1). Implemented in `ptm/stability.py`.

---

## 4. Rachford-Rice flash and bubble/dew points

### 4.1 Wilson's K-value correlation

An inexpensive starting guess for the vapor/liquid distribution ratio
K_i = y_i/x_i, used only to seed the iterative solvers below (never as the
converged answer):

    K_i = (Pc_i/P) * exp[5.373(1+omega_i)(1 - Tc_i/T)]

### 4.2 Rachford-Rice

Given a feed z and **current** K-values, component balances
z_i = (1-beta)x_i + beta*y_i, y_i=K_i x_i, and sum x_i = sum y_i = 1
combine into a single scalar equation in the vapor fraction beta:

    g(beta) = sum_i z_i(K_i - 1) / (1 + beta(K_i-1)) = 0,   beta in (0,1)

**Monotonicity proof sketch:** g'(beta) = -sum_i z_i(K_i-1)^2 / (1+beta(K_i-1))^2
<= 0 everywhere on the open interval where every term's denominator stays
positive (true for a feed whose K-values actually bracket 1, i.e. a feed
that genuinely wants to split). So g is non-increasing, hence bisection
(or safeguarded Newton) on (0,1) always converges to the unique root;
`ptm/flash.py:rachford_rice_beta` uses exactly this safeguarded approach.

### 4.3 Two-level flash and bubble/dew points

The outer loop updates K_i = phi_i^L/phi_i^V from the EOS at the current
x, y (successive substitution; `ptm/flash.py:tp_flash`). Bubble- and
dew-point pressure solvers (`ptm/saturation.py`) are the same iteration
specialized to the beta->0 and beta->1 limits: find P such that
sum_i K_i x_i = 1 (bubble, liquid composition given) or sum_i y_i/K_i = 1
(dew, vapor composition given). Sweeping liquid composition x1 from 0 to 1
at fixed T and solving the bubble point at each x1 produces the classic
**Pxy diagram** (Section 1.2's F=2 result in action).

---

## 5. Continuous random variable theory, pillar 1: continuous thermodynamics

Real heavy fractions (e.g. the "C7+" tail of a gas condensate) are not a
handful of discrete species -- they're a continuum of isomers and
homologues, usually characterized only by a molar-mass distribution.
**Continuous thermodynamics** (Cotterman & Prausnitz, 1985 [4]; building on
Whitson, 1983 [5]) treats the mole fraction of the fraction as a
probability density p(M) over molar mass M, typically a 3-parameter
(shifted) **gamma distribution**:

    p(M) = (M-eta)^(alpha-1) exp[-(M-eta)/beta] / [beta^alpha * Gamma(alpha)],   M >= eta

with mean Mn = eta + alpha*beta and variance sigma^2 = alpha*beta^2 (eta
is the minimum molar mass in the cut, alpha and beta shape/scale
parameters fit to a measured mean and variance).

A cubic EOS can't integrate over a continuous density directly -- it needs
a finite list of discrete pseudo-components. **Gauss-Laguerre quadrature**
is the mathematically exact way to collapse p(M) into q discrete nodes
(molar masses) and weights (mole fractions): substituting t=(M-eta)/beta
turns the gamma density into the canonical generalized-Laguerre weight
function t^(alpha-1) e^-t, for which q-point Gauss quadrature integrates
any polynomial in t up to degree 2q-1 **exactly**. Since the mean is a
degree-1 moment and the variance a degree-2 moment, a q>=2-point rule
reproduces both exactly, regardless of q -- the higher moments (skewness,
etc.) are what additional quadrature points buy you. `scipy.special.
roots_genlaguerre` supplies the nodes/weights; see
`ptm/continuous.py:gauss_laguerre_pseudo_components` and the moment-check
test.

---

## 6. Continuous random variable theory, pillar 2: Monte Carlo uncertainty propagation

Tc, Pc, and omega are never known exactly -- they come from a correlation
or regression with some standard error. Treating them as independent
normal random variables, Tc ~ N(Tc_hat, sigma_Tc^2) etc., and sampling many
draws through the **full nonlinear** EOS/flash pipeline (not a linearized
delta-method approximation) gives a Monte Carlo confidence band on any
derived quantity -- here, Psat(T) curves and, by the same mechanism,
bubble/dew curves. This is simple brute-force simulation, but it's the
correct tool when the map from parameters to the quantity of interest is
strongly nonlinear (which Psat(T; Tc,Pc,omega) is, especially near Tc) and
differentiating the whole iterative solver for a first-order propagation
would be far more work for no real gain in accuracy at n_samples ~
hundreds. See `ptm/uncertainty.py:mc_psat_band`.

---

## 7. Continuous random variable theory, pillar 3: maximum-likelihood parameter estimation

### 7.1 Model and likelihood

Given paired data (T_i, P_i^obs) and an EOS model P_i^model(theta) (here,
theta = omega, with Tc and Pc fixed at their independently-known values),
assume multiplicative (equivalently, log-additive) i.i.d. Gaussian error:

    ln P_i^obs = ln P_i^model(theta) + epsilon_i,   epsilon_i ~ N(0, sigma^2)

(log-space because vapor pressure spans orders of magnitude across the
250-400 K range, so a constant *relative* error is far more realistic than
a constant *absolute* one). The log-likelihood is

    L(theta, sigma) = -n/2 ln(2*pi*sigma^2) - (1/2sigma^2) sum_i [ln P_i^obs - ln P_i^model(theta)]^2

Maximizing over theta for *any* fixed sigma is identical to minimizing the
sum of squared log-residuals -- so the MLE of omega is exactly the
nonlinear least-squares estimate, found here with a bounded scalar
optimizer (`scipy.optimize.minimize_scalar`, "bounded" method) rather than
a closed form, since P^model(omega) is only available through the EOS's
own iterative Psat solver.

### 7.2 Standard error from the Fisher information

The large-sample variance of the MLE is the inverse Fisher information.
For this Gaussian model with scalar theta, the (observed) Fisher
information reduces to

    I(theta_hat) = (1/sigma_hat^2) * sum_i (d ln P_i^model / d theta)^2 |_{theta_hat}

(the Gauss-Newton approximation to the Hessian of the negative
log-likelihood, exact at the optimum for this model). The derivative is
evaluated by forward finite difference around the fitted omega_hat (the
model has no closed-form derivative, since it's defined by an iterative
bisection). se(omega_hat) = 1/sqrt(I(theta_hat)). See
`ptm/fitting.py:fit_omega_mle`.

### 7.3 Out-of-sample validation

To avoid reporting a fit's error on the same points it was fit to (which
trivially underestimates error), every validation in this project splits
each fluid's data into alternating train/test points, fits on train, and
reports AAD% = mean(|P_pred - P_test|/P_test) * 100 on the **held-out**
test points only (`ptm/validation.py`). Results: every one of the 5 fluids
comes in under 0.7% AAD -- see `results/validation_table.csv` and
[IMPLEMENTATION.md](IMPLEMENTATION.md) Section 8 for the exact numbers and
how to reproduce them.

---

## References

1. Peng, D.-Y.; Robinson, D.B. "A New Two-Constant Equation of State." *Ind. Eng. Chem. Fundam.* **1976**, *15* (1), 59-64.
2. Robinson, D.B.; Peng, D.-Y. "The Characterization of the Heptanes and Heavier Fractions for the GPA Peng-Robinson Programs." GPA Research Report RR-28, **1978**.
3. Michelsen, M.L. "The Isothermal Flash Problem. Part I. Stability." *Fluid Phase Equilib.* **1982**, *9* (1), 1-19.
4. Cotterman, R.L.; Prausnitz, J.M. "Flash Calculations for Continuous or Semicontinuous Mixtures Using an Equation of State." *Ind. Eng. Chem. Process Des. Dev.* **1985**, *24* (2), 434-443.
5. Whitson, C.H. "Characterizing Hydrocarbon Plus Fractions." *SPE J.* **1983**, *23* (4), 683-694.
6. Poling, B.E.; Prausnitz, J.M.; O'Connell, J.P. *The Properties of Gases and Liquids*, 5th ed.; McGraw-Hill: New York, 2001 (pure-component constants, `ptm/components.py`).
7. NIST Chemistry WebBook, *Thermophysical Properties of Fluid Systems*, https://webbook.nist.gov/chemistry/fluid/ (validation data source, `data/vle/`; per-fluid reference-EOS citations in `data/raw/extract.py`).
