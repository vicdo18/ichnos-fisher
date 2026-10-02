# Claims ledger — what we may say, and what supports it

Every statement the Fisher analysis produces is one of two kinds, and
confusing them is the single most likely way to lose a judging conversation:

- **Shape** — orderings, correlations, which readout time beats which, whether
  an optimum exists. Invariant under rescaling, so independent of the photon
  budget. These are defensible today.
- **Scale** — any absolute number of minutes or percent. These scale with
  `PHOTONS_PER_UNIT`, which has not been measured, and (for σ_t) with
  `d_x_ox`, which cannot be measured from existing data. These are not
  defensible today and should be stated as formulas, not values.

| # | Claim | Kind | Evidence | What would falsify it | Status |
|---|---|---|---|---|---|
| 1 | The two channels are not separable: dose and time are confounded, corr ≈ −0.9 | shape | Two independent implementations agree to 2% (−0.844 vs ablation's −0.859); survives a 370× change in the non-identifiable kinetics (6% move); the empirical correlation of ML-recovered estimates converges to the predicted one (−0.923 vs −0.929) | A measurement with genuinely separated channels (real single-colour control) showing \|corr\| well below 0.9 | **Defensible** |
| 2 | The informative readout window is early — roughly 1–1.5 h — not 3 h | shape | Fisher interior minimum at 46–100 min depending on threshold and kinetics; ablation age-resolution best at 1 h; physics-informed analysis independently at 1 h | A measured detection floor sitting above that window | **Defensible qualitatively.** Not as a single minute — t* moves 1.32× along the parameter ridge |
| 3 | The CRLB is not attainable from one cell. It describes a pooled measurement: ~10 cells for timing, ~300–1000 for dose | structure | `docs/attainment.md`: ratio 4.58 at N=1 with a +29.8 min bias on a 60 min truth, converging to 1.00; four independent convergence signatures agree | A different estimator (e.g. Bayesian with a real dose prior) reaching the bound at lower N — the MLE is one choice, not the only one | **Defensible** |
| 4 | The detection threshold decides whether the optimum is reachable at all: at MIN_PHOTONS = 10 the first accessible point (74 min) has already passed the interior optimum (46 min) | structure | Direct computation, both thresholds reported | A measured LOD low enough that the optimum is inside the accessible range | **Defensible** |
| 5 | Timing precision is σ_t = *X* minutes | scale | Insufficient. Depends on `PHOTONS_PER_UNIT` (unmeasured) **and** on `d_x_ox` (2.34× spread along an unmeasurable ridge, `docs/ridge.md`) | — | **Do not state** |
| 6 | "Events 0.5–3 h old dated to better than 30 min at N = 100" | scale | Same two unmeasured dependencies | — | **Do not state as is** |
| 7 | The er branch sits on the detection floor: its red channel rises 9.76 photons against a threshold of 10 | scale-dependent structure | `docs/findings.md`; a 2.4% margin | A photon-budget measurement moving it clearly to one side | **State conditionally** — "within 3% of the floor; which side is decided by two unmeasured constants" |

## How to present it

**Lead with the decision, not the mathematics.** The analysis moved the
planned readout from 3 h to ~1 h, and the wet lab's **H₂O₂** time points were
chosen from it. That is the Model criterion — a model that changed what the
team did — and it is worth more than a tighter bound.

**Be precise about which stressor.** H₂O₂ is the `ox` branch, and that
recommendation is model-derived: the CRLB window plus the ablation study's
age-resolution table. **CuSO₄ has no model at all** — `VARIANTS` in
`ichnos_config.py` is `{ox, er}`, and copper appears once in the whole
codebase as a note that TIP is shared "in ox/er/copper". The CuSO₄ time-point
advice (bracket the peak with 0.5 / 1 / 2 h) came from identifiability
reasoning applied to the wet lab's own observation that the signal peaks near
1 h and falls at high dose — a peak sitting at the edge of the sampling window
makes peak position and amplitude mutually unidentifiable — not from any
computation in this repo.

Do not blur the two. Presented correctly it is a better story anyway: we told
the wet lab where we could compute and where we could not, and the
recommendation was *different* in the two cases — more time points precisely
where no model exists to interpolate between them.

**Then the validation, which almost nobody does.** We did not only compute a
Cramér–Rao bound; we tested whether any estimator reaches it, and found it
does not at a single cell. One figure carries this
(`results/crlb_attainment.png`).

**Then the limits, stated as formulas rather than hidden.** Write

> σ_t = 929 min × √(25 / λ_peak), with λ_peak = 25 photons **provisional**

instead of "σ_t = 929 min". When a judge asks how you know the number, the
answer "we don't — we know how it scales, and here is the measurement that
fixes it" is a stronger answer than a number, because it demonstrates you know
which of your quantities are measured.

**Expected questions, and the honest answers**

- *"Your d_x_ox could be 400× different. Why believe any of this?"* — We
  checked (`docs/ridge.md`). The correlation claim moves 6%; the readout-window
  claim stays qualitatively the same; σ_t moves 2.34×, which is why we do not
  quote an absolute σ_t.
- *"How do you know your Fisher matrix is right?"* — Two independent
  differentiation routes agree to 0.64%, and the empirical correlation of
  recovered estimates converges to the matrix's own prediction.
- *"Is 30-minute dating real?"* — Not yet. It rests on an unmeasured photon
  budget and an unmeasured kinetic parameter. Here is the measurement plan.
- *"Does your model cover all three stressors?"* — No. There are two
  branches, ox (H₂O₂) and er (DTT). There is no copper model. We said so to
  the wet lab when they asked, and designed the CuSO₄ time points to carry
  the shape in the data instead.
- *"Why did you change the model mid-project?"* — Because we found and
  documented three defects in our own code (`docs/findings.md`), including one
  that hid a 3.6× brightness difference between branches.

**What we would need in order to claim more**

1. Photon Transfer Curve + negative control → fixes `PHOTONS_PER_UNIT` and the
   detection floor, and converts claims 5–7 from formulas into values.
2. An experiment that pins `d_x_ox` → removes the remaining 2.34× on σ_t.
3. A single-fluorophore control strain → removes the perfect-crosstalk
   assumption, under which everything above is a lower bound on a lower bound.
