# ICHNOS — Fisher information / Cramér–Rao analysis

How precisely can the ICHNOS tandem-timer decoder recover **(dose, time since
onset)** from the two fluorescence channels? This repo holds the Fisher
information implementation, the reconciliation work against the ablation
study's independent implementation, and the measurement protocol that the
absolute numbers are waiting on.

> **Read this first.** Every *absolute* σ_t and σ_dose in this repo is
> conditional on one unmeasured constant (`PHOTONS_PER_UNIT`). Orderings,
> correlations and relative comparisons are trustworthy; absolute minutes are
> not yet. See [Status](#status).

## What is here

| Path | What it is |
|---|---|
| `fisher_info_v3_1.py` | The analysis. Exact Poisson 2×2 FIM for θ = (dose, t) on the merged SBML model, with explicit run provenance. |
| `scripts/step1_factorial.py` | One-variable-at-a-time over pre-equilibration × clearance, so each protocol change's contribution is separable. |
| `scripts/step1_diagnose.py` | Isolates how much of a change is the photon-scale re-anchoring artifact vs physics. |
| `scripts/step1_match_ablation.py` | Runs this code on the ablation study's exact protocol, to compare like with like. |
| `scripts/crlb_attainment.py` | **Validation**: is the bound actually reachable, and at how many pooled cells? Maximum-likelihood recovery from synthetic Poisson measurements vs `F⁻¹`. |
| `scripts/plot_attainment.py` | The figure for that test. |
| `scripts/ridge_stability.py` | Moves along the unmeasurable `d_x`/`k_off` ridge, with a negative control, to test which conclusions depend on it. |
| `scripts/unify_crlb.py` | Same trajectory, same parameterization, noise model swapped (Poisson ↔ lognormal CV). |
| `docs/findings.md` | What the reconciliation found, with the receipts. |
| `docs/attainment.md` | The validation result and what it licenses us to claim. |
| `docs/ridge.md` | Whether the conclusions survive the non-identifiable kinetics. |
| **`docs/claims.md`** | **The claims ledger — what may be said, what supports it, how to present it.** Start here. |

## Setup

This analysis does **not** reimplement the circuit. It imports the merged SBML
model from the main modelling repo, so that the Fisher analysis and the
sensitivity/ablation analyses can never silently diverge.

```bash
git clone https://github.com/ioanna888/Ichnos_PULSE.git
pip install -r requirements.txt           # numpy, python-libsbml, tellurium
export PYTHONPATH=$PWD/Ichnos_PULSE/python:$PWD/Ichnos_PULSE/python/sensitivity_analysis/sensitivity
python fisher_info_v3_1.py
```

`run_sensitivity_v4.py` must be importable from `PYTHONPATH` — it provides the
`Variant` class and `resolve()`. On the current main branch it sits under
`python/sensitivity_analysis/sensitivity/`, which is why two paths are needed.
On Windows PowerShell:
`$env:PYTHONPATH = "$PWD\Ichnos_PULSE\python;$PWD\Ichnos_PULSE\python\sensitivity_analysis\sensitivity"`. The ablation study's independent CRLB
implementation lives separately, at
`https://github.com/marifylli/ICHNOS-ablation` (`ichnos_ablation.py`,
`decoding_crlb()`); this repo does not vendor either codebase.

Every run prints a provenance header naming the parameter package, the
pre-equilibration horizon, the photon scale and the detection mode. **No number
from this repo should be quoted without it.**

## Status

### Validated

- **The FIM implementation.** Two independent differentiation routes agree to
  0.64%, and the empirical dose/time correlation of recovered estimates
  converges to the FIM's own (−0.923 against −0.929).
- **The bound is attainable — but only as a population measurement.** It is
  missed by 4.6× at one cell, reached from about 10 pooled cells for timing
  and about 300–1000 for dose. See `docs/attainment.md`.

### Trustworthy

- Which readout time is better than which (orderings).
- `corr(dose, t)` — invariant under rescaling, and it agrees with the ablation
  study's independent implementation to 2% (−0.844 vs −0.859) once both run the
  same protocol.
- The shape of CRLB(t), and whether an interior minimum exists at all.
- Relative comparisons between lesions and between branches.
- That the channels are not separable: corr moves only 6% across a 370×
  change in the non-identifiable kinetics.

### Not yet anchored

- **Absolute σ_t, for a second and independent reason.** It moves 2.34×
  along the `d_x_ox`/`k_off_ox` ridge the data cannot constrain, so fixing
  the photon budget alone will not anchor it (`docs/ridge.md`).
- **Any single-cell σ_t.** The bound does not describe a one-cell decoder at
  all: the estimator is biased +30 min on a 60 min truth at N = 1.
- **Every absolute σ_t and σ_dose.** Fisher information is linear in λ, so
  σ_t ∝ 1/√`PHOTONS_PER_UNIT`, and that constant is a placeholder. The Photon
  Transfer Curve measurement replaces it.
- **Whether the er branch is detectable at all.** Its red channel rises 9.76
  photons above baseline against a threshold of 10 — a 2.4% margin, decided
  entirely by `PHOTONS_PER_UNIT` and `MIN_PHOTONS`, both unmeasured.
- **The detection floor** (`MIN_PHOTONS` here, θ in the ablation, `--fold` in
  the Kd analysis — one physical quantity under three names). Awaits a
  negative-control measurement.

### Known limitations, stated rather than omitted

1. Perfect crosstalk correction is assumed. Blind per-image crosstalk
   estimation is unidentifiable for a tandem-fusion reporter without a
   single-colour control per session, so **the real bound is ≥ what this
   reports**. Treat the output as a lower bound.
2. Shot-noise only: no camera read noise or gain. The camera is a **cooled
   QImaging MicroPublisher 3.3 RTV** (10-bit sensor, 8-bit saved, Image-Pro
   Plus) — not the SC30 that earlier versions of this file named.
3. No autofluorescence baseline, no proteasomal fragment-escape state.
4. `d_x_ox` is not individually identifiable — only the combination
   r = k_on·d_x/(k_off·k_x) is determined, and k_off and d_x move ~400×
   between gel quantifications.

## Parameter packages, not parameters

Clearance cannot be set on its own in this code, by design. `k_clear = 1.969`
belongs to a fit that also sets `k_on = 349.4, d_x = 11.11, K_act = 158.1,
n = 4.801` (`profile_reporter_ox.py`). Applying it on top of the SBML's
`K_act_ox = 208, n_ox = 2.75, d_x_ox = 0.5` mixes four parameters across two
fits — and the main repo already warns that doing so "is not neutral … they
pull in opposite directions" (`run_kd_extended.py`).

So `PARAM_PACKAGES` holds named packages that are applied **whole**, and
`ACTIVE_PACKAGE = None` runs the model as committed. A package that moves
`K_act` makes the code warn that the reference dose is no longer the
half-activation point.

Once the fitted values live in the SBML itself, `None` becomes the only correct
setting and that whole block should be deleted.

## Reproducing the earlier numbers

v2's published figures are reproducible exactly — set `PRE_T_HOURS = 0`,
`PHOTON_SCALE_MODE = "legacy_peak"`, `DETECT_MODE = "absolute"` and
`ACTIVE_PACKAGE = "v2_legacy_1777"`:

| | t\* | SD(t) | corr |
|---|---|---|---|
| ox | 74.20 min | 1150.62 min | −0.563 |
| er | 81.40 min | 1569.36 min | −0.772 |

Those modes exist for regression only. `legacy_peak` makes the noise floor a
function of the protocol, which is the defect v3.1 fixes.

## License / provenance

Parameter values, calibration decisions and the equipment record come from the
ICHNOS dry-lab documents and the two modelling repos linked above. Numbers
carried from those sources are cited in the code comments at the point of use.
