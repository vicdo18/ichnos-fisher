# Reconciliation findings

Two independent implementations compute the Cramér–Rao bound on a joint
(dose, t) estimate from the same merged SBML model: `fisher_info_*.py` here
(exact Poisson photon noise) and `decoding_crlb()` in the ablation study
(multiplicative lognormal, CV = 0.25). They were never compared. This is what
the comparison found.

## 1. They agree on the geometry, to 2%

| Source | Branch | corr(dose, t) |
|---|---|---|
| Ablation, intact arm | ox | −0.859 |
| This code, ablation's protocol | ox | −0.844 |
| This code | er | −0.871 |

The CRLB correlation is invariant under separate rescaling of each parameter,
so it is the right quantity to compare across implementations that
parameterize dose differently (raw dose here, ln dose there). Two independent
codebases, two noise models, the same degeneracy geometry. That is the
strongest independent validation the project has.

**It needs the protocol stated.** The sign of corr at 3 h flips with
clearance: +0.867 with k_clear = 1.777, −0.844 with k_clear = 0. The
ablation's −0.859 is only reproducible with constant S.

## 2. The ~5× disagreement in magnitude was never one cause

The headline gap — 182 min (ablation) vs 929 min (this code) — compared two
numbers evaluated at different times under different protocols. Nothing was
held constant. Ruled out and found, in order:

**Not a gradient scaling bug.** `np.gradient(green, t_min)` passes the full
minute-valued coordinate array, not a step, so d λ/dt is correctly per-minute.
Verified by reading, then by running.

**Not the noise model.** With the trajectory and evaluation time held fixed and
only the noise model swapped, σ_t differs by **8% (ox) and 13% (er)** — not 5×.
At the photon levels in play (~20 ph/channel at 3 h) the Poisson CV is
1/√20 ≈ 0.22, nearly identical to the assumed 0.25. The two noise models
almost coincide there.

**Pre-equilibration.** The ablation integrates 50 h at zero stress before
dosing; this code started every trace from an empty vessel, where the first
hours are dominated by reporter-pool filling rather than by the stress
response — same timescale, not separable. The ablation measured the cost:
*"without pre-eq the dose information is underestimated ~5x"* (`protocol.py`
line 39).

**Clearance.** The ablation has none (S constant forever); this code applied
1.777/h to both branches, a ~23-minute input half-life. By 3 h the stimulus is
gone, ∂λ/∂t → 0 and the FIM is near-singular — which is why at a matched
evaluation time the gap widens to ~65×, not 5×.

Order of operations matters here and is a trap: once S is governed by
dS/dt = −k_clear·S, pre-equilibrating with the dose applied decays it away
before the run starts (50 h at 1.777/h ≈ 128 half-lives). Pre-eq must run at
S = 0 **and** k_clear = 0, with both set afterwards.

## 3. What was wrong in this code, found by fixing it

**The photon scale was a function of the protocol.** `conv = 25 / max(trace)`
anchored the absolute noise level to the maximum of whatever trace was being
simulated. Enabling pre-equilibration raised the maximum (10.70 → 13.96),
dropping the conversion to 0.766× and handing every time point fewer photons.
Measured: **~40% of the apparent degradation from pre-equilibration was this
re-anchoring, not physics.**

**And it was per branch**, which is physically impossible — same reporter, same
camera, same microscope cannot yield different photons per reporter molecule.
Normalizing each branch to its own peak gave the dimmer branch a *larger*
conversion factor, erasing the difference. With one shared scale:

| Branch | Peak green (model units) | Peak (photons) | Green rise above baseline | Red rise |
|---|---|---|---|---|
| ox | 27.85 | 65.06 | 50.02 | 41.16 |
| er | 7.66 | 17.89 | 12.39 | **9.76** |

The er branch is **3.6× dimmer**, and its red channel rises 9.76 photons
against a threshold of 10. v2 hid this entirely.

The right reading is not "er does not work". A 2.4% margin against a
placeholder threshold means: **er sits exactly on the detection floor, and
which side it falls on is decided by two unmeasured constants.** That is a more
useful statement than either verdict.

**The detection threshold changed meaning under pre-equilibration.** An
absolute photon floor applied to a trace with a non-zero baseline answers "is
baseline + response detectable?", not "is the response detectable?". Enabling
pre-eq moved the first valid point 74.2 → 28.8 min, and that earlier crossing
was the red *baseline* reaching 10 photons. Fixed by measuring the signal
relative to each channel's t = 0 value, which reduces exactly to the old
behaviour when there is no pre-equilibration.

**The 50 h horizon was copied, not checked.** The ablation justifies 50 h with
"already converged by 20 h" — for its own model. For the merged model with the
clearance rate rule that does not hold:

| Horizon | Max relative difference from 100 h | |
|---|---|---|
| 20 h | 5.931 × 10⁻⁴ | not converged |
| 50 h | 1.243 × 10⁻⁸ | converged |

50 h was right; 20 h would not have been.

## 4. What this leaves open

The remaining disagreement, with protocol and geometry held fixed, is a
**scale** problem: the absolute noise magnitude. On this side that is
`TARGET_SNR_AT_MAX = 5` (25 photons at the trace peak); on the ablation's side
it is `NOISE_CV = 0.25`. Both are self-declared placeholders. Going from
σ_t ≈ 9900 min at 3 h to the ablation's 182 min needs ≈ 2960× more photons, or
~74,000 at peak — an entirely plausible real count for a fluorescence image,
which is exactly why it has to be measured rather than assumed.

That measurement is the Photon Transfer Curve plus a negative control, and it
has no dependency on any of the modelling work. It should start in parallel
with it, not after.
