# Do the conclusions depend on the non-identifiable parameters?

`circuit_impact.py` states the problem: the ox kinetics are not reliable
individually — `k_off` and `d_x` move by ~400× between densitometry
quantifications, and only the combination `r = k_on·d_x/(k_off·k_x)` is
determined. So `d_x_ox` does not exist as a measured quantity. Any value for
it is a convention, and a judge is entitled to ask why anything computed with
it should be believed.

`scripts/ridge_stability.py` answers by moving **along the ridge**: the
one-parameter family `d_x → a·d_x, k_off → a·k_off` that leaves `r` unchanged
and is therefore invisible to the quantifications that produced `r`. A
negative control sweeps the same `d_x` range with `k_off` fixed, which changes
`r` and moves in a direction the data *do* constrain — so that a null result
would mean the test is insensitive rather than that the model is robust.

Sanity check on the ridge definition: scaling the SBML pair (d_x = 0.5,
k_off = 160) to d_x = 0.03 gives k_off = 9.6, and to d_x = 11.11 gives
k_off = 3555. The team's quoted values at those quantifications are ~5.3 and
~1900 — same order at both ends, so the family reproduces the observed
candidate pairs rather than being invented here.

## Result

Scanning d_x_ox over 0.03 → 11.11 (370×):

**Along the ridge** (r fixed; the data cannot see this move)

| d_x_ox | k_off_ox | t* (min) | SD(t) min | SD(dose) % | corr | optimum |
|---|---|---|---|---|---|---|
| 0.03 | 9.6 | 95.1 | 1446.7 | 191.7 | −0.910 | interior |
| 0.1 | 32.0 | 82.1 | 1418.5 | 128.6 | −0.895 | interior |
| 0.5 | 160.0 | 75.6 | 1790.8 | 102.3 | −0.885 | boundary |
| 1.5 | 480.0 | 86.4 | 2467.0 | 98.7 | −0.897 | boundary |
| 4 | 1280.0 | 95.8 | 3123.4 | 112.9 | −0.927 | boundary |
| 11.1 | 3555.2 | 100.1 | 3322.3 | 118.5 | −0.935 | boundary |

**Across the ridge** (negative control; r changes)

| d_x_ox | k_off_ox | t* (min) | SD(t) min | SD(dose) % | corr | optimum |
|---|---|---|---|---|---|---|
| 0.03 | 160.0 | 76.4 | 1782.0 | 92.5 | −0.861 | boundary |
| 0.1 | 160.0 | 76.4 | 1783.7 | 93.7 | −0.865 | boundary |
| 0.5 | 160.0 | 75.6 | 1790.8 | 102.3 | −0.885 | boundary |
| 1.5 | 160.0 | 74.2 | 1803.7 | 122.7 | −0.916 | interior |
| 4 | 160.0 | 77.1 | 1800.8 | 147.8 | −0.939 | interior |
| 11.1 | 160.0 | 82.1 | 1780.1 | 193.6 | −0.951 | interior |

**Spread, max/min over each sweep**

| quantity | along ridge | across ridge |
|---|---|---|
| readout point t* | 1.32× | 1.11× |
| corr(dose, t) | **1.06×** | 1.11× |
| SD(t) | **2.34×** | 1.01× |
| SD(dose) % | 1.94× | 2.09× |

## What it says — and it is not the comfortable answer

**The correlation survives.** `corr(dose, t)` moves by 6% across a 370×
change in a parameter nobody can measure, and by 11% across the ridge. The
claim that the two channels are not separable — the one independently
reproduced by the ablation study's implementation to 2% — does not depend on
the non-identifiable kinetics. This is the strongest single claim the Fisher
analysis has, and the ridge test strengthens it further.

**The readout window survives qualitatively, not precisely.** t* moves from
75.6 to 100.1 min along the ridge. The recommendation "read early, around
1–1.5 h, not at 3 h" holds everywhere on the ridge. The claim "the optimum is
at 46 min" or at any single minute does not.

**σ_t does not survive.** It moves by **2.34× along the direction the data
cannot constrain**, while moving by only 1.01× across the direction the data
*do* constrain. That is the opposite of the reassuring pattern, and it is the
finding: timing precision depends almost entirely on the combination nobody
has pinned down.

The negative control did its job — quantities move in both sweeps, so the
test is sensitive — but what it revealed is that the sensitivity of σ_t is
aligned with the unmeasured direction.

**Whether a genuine interior optimum exists is itself parameter-dependent.**
The "optimum" column flips between interior and boundary in both sweeps. The
existence of an interior CRLB(t) minimum is not a robust structural property
of the circuit.

## Consequence

σ_t is **doubly unanchored**: by the photon scale (a placeholder awaiting the
Photon Transfer Curve) and, independently, by d_x_ox / k_off_ox (a 2.34×
spread along an unmeasurable ridge). Fixing the photon budget alone will not
produce a defensible absolute σ_t. If an absolute timing-precision number is
needed, `d_x_ox` has to be pinned experimentally too — which makes it the
second measurement on the list, after the PTC.

What can be stated without either measurement, because it survives both: the
channels are not separable (corr ≈ −0.9), and the informative readout window
is early rather than at 3 h.
