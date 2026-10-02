"""
ICHNOS -- do our conclusions survive the non-identifiable parameters?

THE VULNERABILITY THIS CLOSES
======================================================================
circuit_impact.py states the problem plainly:

    "The ox kinetics are NOT reliable individually -- k_off and d_x move
     by ~400x between densitometry quantifications, and only the
     combination r = k_on*d_x/(k_off*k_x) is determined."

So d_x_ox does not exist as a measured quantity. Any single value for it
is a convention. A judge is entitled to ask: if the parameter could be
400x different, why should we believe anything you computed with it?

There are two possible answers and only one of them is defensible:

  (a) "We picked the best value."     -- not an answer; it is still a
                                         point on a ridge the data
                                         cannot locate.
  (b) "We checked. The conclusions we
       defend do not depend on it."    -- an answer, if it is true.

This script tests (b), by moving ALONG the ridge: the direction in
parameter space the data cannot see.

HOW THE RIDGE IS DEFINED
======================================================================
r = k_on * d_x / (k_off * k_x) is what the data determine. Holding k_on
and k_x fixed, r is invariant under d_x -> a*d_x, k_off -> a*k_off. That
one-parameter family IS the ridge, and moving along it is by construction
invisible to the quantifications that produced r.

Sanity check on that definition: the SBML pair is (d_x = 0.5,
k_off = 160). Scaling to d_x = 0.03 gives k_off = 9.6, and to d_x = 11.11
gives k_off = 3555. The values actually quoted by the team for those
quantifications are ~5.3 and ~1900 -- same order of magnitude at both
ends, so the proportional family does reproduce the observed candidate
pairs rather than being an invention of this script.

A NEGATIVE CONTROL, SO A NULL RESULT MEANS SOMETHING
======================================================================
If nothing moved no matter what we did, the test would be insensitive
rather than reassuring. So the same d_x sweep is also run with k_off held
FIXED -- which moves ACROSS the ridge, changing r, a direction the data
DO constrain. The contrast between the two sweeps is the result:

  along the ridge, conclusions stable + across it, they move
      -> the test can detect a change, and the non-identifiability
         genuinely does not threaten our claims.

  both stable -> the test is insensitive; conclude nothing.
  along the ridge unstable -> our claims do depend on an unmeasured
         parameter, and must be stated conditionally.

Usage:
    export PYTHONPATH=.../Ichnos_PULSE/python:.../sensitivity_analysis/sensitivity
    python scripts/ridge_stability.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fisher_info_v3_1 as fi  # noqa: E402

VARIANT = "ox"

# The SBML's own pair, used as the anchor of the ridge.
D_X_REF = 0.5
K_OFF_REF = 160.0

# Span the full quantification spread: 0.03 to 11.11 is the 370x range the
# team's own notes quote for d_x_ox.
D_X_SCAN = np.array([0.03, 0.1, 0.5, 1.5, 4.0, 11.11])


def metrics(d_x, k_off, label):
    """CRLB summary for one (d_x_ox, k_off_ox) pair.

    Routed through the package machinery rather than by poking parameters
    directly, so that this sweep obeys the same whole-package discipline
    as every other run in this repo.
    """
    fi.PARAM_PACKAGES["_ridge_probe"] = {
        VARIANT: {"d_x_ox": float(d_x), "k_off_ox": float(k_off)}}
    saved = fi.ACTIVE_PACKAGE
    fi.ACTIVE_PACKAGE = "_ridge_probe"
    try:
        t_h, FIM, valid, dose = fi.fisher_matrix(VARIANT)
        crlb_dose, crlb_t, corr, crlb_t_valid, det = fi.crlb_from_fim(
            t_h, FIM, valid)
    finally:
        fi.ACTIVE_PACKAGE = saved

    if np.all(np.isnan(crlb_t_valid)):
        return dict(label=label, d_x=d_x, k_off=k_off, ok=False)

    i = int(np.nanargmin(crlb_t_valid))
    first_valid = int(np.argmax(valid))
    return dict(label=label, d_x=d_x, k_off=k_off, ok=True,
                t_star=t_h[i] * 60.0,
                sd_t=np.sqrt(crlb_t[i]) * 60.0,
                sd_dose_pct=np.sqrt(crlb_dose[i]) / dose * 100.0,
                corr=corr[i],
                interior=(i != first_valid),
                t_first=t_h[first_valid] * 60.0)


def sweep(along_ridge):
    """One sweep. along_ridge=True keeps r fixed (k_off scales with d_x)."""
    rows = []
    for d_x in D_X_SCAN:
        k_off = K_OFF_REF * (d_x / D_X_REF) if along_ridge else K_OFF_REF
        rows.append(metrics(d_x, k_off,
                            "along" if along_ridge else "across"))
    return rows


def show(rows, title):
    print(f"\n{title}")
    hdr = (f"  {'d_x_ox':>8} {'k_off_ox':>9} {'t* (min)':>9} {'SD(t) min':>10} "
           f"{'SD(d) %':>8} {'corr':>8} {'kind':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        if not r["ok"]:
            print(f"  {r['d_x']:>8.3g} {r['k_off']:>9.1f}  "
                  f"no detectable point within the window")
            continue
        print(f"  {r['d_x']:>8.3g} {r['k_off']:>9.1f} {r['t_star']:>9.1f} "
              f"{r['sd_t']:>10.1f} {r['sd_dose_pct']:>8.1f} {r['corr']:>+8.3f} "
              f"{('interior' if r['interior'] else 'boundary'):>9}")
    return rows


def spread(rows, key):
    """max/min of a quantity over a sweep; the number a judge asks for."""
    vals = [abs(r[key]) for r in rows if r.get("ok")]
    if len(vals) < 2 or min(vals) == 0:
        return float("nan")
    return max(vals) / min(vals)


def main():
    print("=" * 74)
    print("RIDGE STABILITY -- do the conclusions depend on d_x_ox?")
    print("=" * 74)
    print(f"  variant           : {VARIANT}")
    print(f"  d_x_ox scanned    : {D_X_SCAN.min():g} to {D_X_SCAN.max():g} "
          f"({D_X_SCAN.max() / D_X_SCAN.min():.0f}x)")
    print(f"  ridge definition  : r = k_on*d_x/(k_off*k_x) held fixed by "
          f"scaling k_off with d_x")
    print(f"  anchor            : SBML pair d_x = {D_X_REF:g}, "
          f"k_off = {K_OFF_REF:g}")
    print(f"  package           : {fi.ACTIVE_PACKAGE!r}, pre-eq "
          f"{fi.PRE_T_HOURS:g} h, photon scale {fi.PHOTONS_PER_UNIT:g} "
          f"ph/unit (PROVISIONAL)")

    along = show(sweep(True),
                 "ALONG the ridge (r fixed -- the data cannot see this move):")
    across = show(sweep(False),
                  "ACROSS the ridge (k_off fixed, r changes -- NEGATIVE "
                  "CONTROL,\nthe data DO constrain this direction):")

    print("\n" + "=" * 74)
    print("SPREAD (max/min over each sweep)")
    print("=" * 74)
    print(f"  {'quantity':<22} {'along ridge':>13} {'across ridge':>14}")
    for key, name in (("t_star", "readout point t*"),
                      ("corr", "corr(dose, t)"),
                      ("sd_t", "SD(t)"),
                      ("sd_dose_pct", "SD(dose) %")):
        print(f"  {name:<22} {spread(along, key):>13.2f}x "
              f"{spread(across, key):>13.2f}x")

    print("\nHOW TO READ THIS")
    print("  A spread near 1.00x means the quantity does not move over that")
    print("  sweep. The claim 'our conclusions do not depend on the")
    print("  non-identifiable parameter' is earned only if the ALONG column")
    print("  is near 1 AND the ACROSS column is not -- otherwise the test")
    print("  simply cannot detect anything and proves nothing.")


if __name__ == "__main__":
    main()
