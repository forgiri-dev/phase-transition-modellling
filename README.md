# Phase Transition Modeling

Statistical/EOS-based prediction of phase-separation (vapor-liquid
equilibrium) behavior for multi-component hydrocarbon + CO2 systems across
250-400 K and up to 50 atm, with uncertainty quantified via Monte Carlo
sampling and continuous random-variable theory used for a heavy-fraction
("C7+") pseudo-component model.

Built from scratch in Python (numpy/scipy/matplotlib only): a Peng-Robinson
cubic equation of state, Michelsen's tangent-plane stability test,
Rachford-Rice two-phase flash, bubble/dew-point solvers, maximum-likelihood
parameter fitting, Monte Carlo uncertainty propagation, and Gauss-Laguerre
quadrature for continuous thermodynamics.

See **[docs/THEORY.md](docs/THEORY.md)** for the physics and statistics,
and **[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)** for how each piece
of theory maps to code, file by file.

## What's actually validated, and what's predictive

This matters, so it's stated up front rather than buried:

- **Pure-component vapor pressure** (5 fluids: methane, ethane, propane,
  n-butane, CO2) is validated against real literature data -- the NIST
  Chemistry WebBook reference equations of state (`data/vle/*_psat.csv`,
  fetched directly from `webbook.nist.gov`, source cited in each file's
  header; see `data/raw/extract.py`). Held-out test-point deviation is
  **under 0.7% AAD** for every fluid (`python -m ptm validate`,
  `results/validation_table.csv`) -- comfortably inside the 5% target.
- **Binary and ternary mixture diagrams** (5 binaries + 1 ternary, all
  within 250-400 K and <=50 atm) are **PR-EOS predictions**, not
  point-by-point validated against independent experimental mixture data.
  I tried: the classic data sources for these systems (Kay 1938; Reamer,
  Sage & Lacey 1951; Rigas, Mason & Thodos 1959; and several more recent
  papers) are paywalled, and the open-access candidates I found (OSTI
  records, a Caltech thesis, a PMC article, a NIST mixture tool) each
  failed on fetch -- paywall, bot-check, or dead host. Rather than
  fabricate or misremember numeric VLE tables, hydrocarbon-hydrocarbon
  binaries use k_ij=0 (the standard van der Waals one-fluid default,
  itself validated as reasonable for similar-sized nonpolar hydrocarbons
  in Peng & Robinson's original 1976 paper), and CO2 pairs use k_ij=0.13,
  this project's own placeholder of the right order of magnitude for
  CO2+paraffin systems -- **not a fitted or literature value**. This is
  flagged in the plot titles ("predictive") and in `ptm/cli.py`.
- **Continuous thermodynamics** (gamma-distributed C7+ fraction) and the
  **Monte Carlo uncertainty bands** are illustrative/methodological demos,
  not tied to a specific real reservoir fluid.

## Install

```
pip install -r requirements.txt
```

## Usage

```
python -m ptm validate                        # pure-fluid EOS validation table
python -m ptm diagram --system propane-nbutane --T 320
python -m ptm diagram --system co2-propane
python -m ptm psat --component propane --mc 150   # Psat parity + 95% MC band
python -m ptm ternary                          # methane/propane/n-butane tie lines
python -m ptm condensate                        # gamma C7+ -> pseudo-components
python -m ptm all                               # regenerate everything
```

Figures land in `figures/`, tables in `results/`.

## Tests

```
pytest tests/ -q
```

15 tests: EOS root selection and fugacity-coefficient consistency,
Rachford-Rice mass balance, flash mass balance, Gauss-Laguerre moment
recovery, and MLE recovery of a known parameter from synthetic data.

## Layout

```
ptm/            Python package (EOS, flash, stability, fitting, diagrams, CLI)
data/raw/       NIST WebBook downloads + extraction script
data/vle/       Cleaned, cited Psat tables used for validation
tests/          pytest suite
figures/        Generated plots
results/        Generated tables
docs/THEORY.md          Academic background report
docs/IMPLEMENTATION.md  Implementation report (theory -> code mapping)
```
