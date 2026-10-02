"""
ICHNOS -- fisher_info_v2.py

Fisher Information / Cramer-Rao analysis for the ICHNOS tandem-timer
biosensor decoder, reworked around three changes from fisher_info.py (v1).

WHAT CHANGED FROM v1, AND WHY
------------------------------------------------------------------------
1. SHARED MODEL, NOT A SEPARATE REIMPLEMENTATION.
   v1 hand-wrote its own 5-state ODE system (scipy.integrate.odeint) with
   a hardcoded constant production rate (prod = 1.0), completely
   disconnected from the actual merged SBML circuit model used elsewhere
   in the project. Here, Observed_Green and Measured_Ratio_RG are pulled
   directly from the SAME merged model already used by
   run_sensitivity_v4.py and run_control_circuit_v2.py, via the existing
   `Variant` class. This mirrors the discipline already applied in
   run_control_circuit_v2.py's ablate_feedback() -- "the control is not
   hand-rewritten" -- so that the Fisher analysis and the
   sensitivity/control-circuit analysis can never silently diverge.
   As a side effect, this automatically gives the real, time-dependent
   production rate
       prod(t) = beta_basal + (beta_max - beta_basal) * A(t)
   instead of a constant, which resolves Gap #4 of the original
   evaluation report ("Introduce dynamic / pulsatile production").

2. A 2x2 FISHER INFORMATION MATRIX FOR (dose, t), NOT A SCALAR FOR t ALONE.
   The decoder reports (dose, t) jointly (a 2D posterior -- already
   documented in the image-processing pipeline notes). The
   sensitivity_v4 report shows explicitly that Measured_Ratio_RG is NOT
   perfectly dose-invariant. CORRECTED (post advisor review): the
   originally-cited ratio_spread = 8.2% for the ER branch came from the
   OLD static-dose scenario (k_clear = 0), before the decaying-dose input
   used below (K_CLEAR_PER_HOUR = 1.777) was adopted. Under the scenario
   actually used here, ratio_spread for the ER branch is 9.47% at 6h and
   21.4% at 3h -- LARGER, not smaller, than the figure this comment used
   to cite (see the Fisher v5.2 report, Section 2, for the full
   correction history). Either figure is above the project's own stated
   5% threshold. This means dose and
   t are partially confounded: a scalar CRLB for t alone, computed as if
   dose were already known exactly, would overstate the achievable
   precision. We therefore compute the full 2x2 matrix and report the
   dose/time correlation explicitly.

3. EXACT POISSON FISHER INFORMATION, NOT A DELTA-METHOD APPROXIMATION ON
   THE RATIO. For independent Poisson-distributed channel counts with
   means lambda_k(theta), the exact Fisher information is

       FIM(theta) = sum_k  grad(lambda_k) * grad(lambda_k)^T / lambda_k

   This is computed directly on the sufficient statistics (red_counts,
   green_counts) rather than propagating an approximate first-order
   (delta-method) variance through the derived ratio mature_red /
   Observed_Green, as v1 did. One fewer layer of approximation, and it
   extends naturally to a joint (dose, t) parameter vector.

4. THE ALREADY-FIXED DETECTION-THRESHOLD BUG IS PRESERVED: the validity
   mask requires BOTH channels to exceed MIN_PHOTONS
   (green_counts >= MIN_PHOTONS) AND (red_counts >= MIN_PHOTONS),
   not only the green channel as the original v1 implementation did.

5. SATURATION USES THE REAL CAMERA LIMIT. SATURATION = 255 (SC30,
   8-bit/channel, confirmed against the equipment inventory), not the
   old, incorrect 65535 (16-bit, which matched the public reference
   images the model was originally sanity-checked against, not our own
   camera).

WHAT THIS MODEL DOES NOT DO -- EXPLICITLY STATED LIMITATIONS, NOT SILENT
OMISSIONS
------------------------------------------------------------------------
(a) It does NOT include crosstalk-correction / spectral-unmixing
    uncertainty. The image-processing pipeline (Stage 4, status
    "review") has already shown that blind, per-image crosstalk
    estimation is mathematically unidentifiable for a tandem-fusion
    reporter without a real single-color GFP-only control per imaging
    session (the true red signal is, by construction, proportional to
    the true green signal within every cell). The CRLB computed here
    assumes perfect signal separation. The real CRLB is therefore >=
    what is reported here. TREAT THIS AS A LOWER BOUND, NOT A FINAL
    NUMBER.
(b) It does NOT include camera read noise / gain (SC30, no active
    cooling per the equipment log). A Photon Transfer Curve measurement
    is still outstanding. This is a shot-noise-only model, stated
    explicitly.
(c) It does NOT include an autofluorescence baseline (FAD/NADH) or a
    proteasomal fragment-escape state (incomplete degradation of the
    sfGFP beta-barrel) -- both are older, still-open gaps from the
    original evaluation report, left out of this revision because of
    time constraints. They remain on the list of known limitations and
    should not be forgotten in the next iteration.
(d) The clearance-rate scenario (k_clear) and the reference dose are
    EXPLICIT choices set below, not silent defaults. Change them openly
    if you change the scenario -- do not bury a different assumption
    elsewhere in the code.

USAGE
------------------------------------------------------------------------
Place this file next to run_sensitivity_v4.py (it imports from it) and
run:

    python fisher_info_v2.py
"""

import numpy as np

from run_sensitivity_v4 import Variant, resolve

# ---------------------------------------------------------------------------
# EXPLICIT ASSUMPTIONS -- change them here, openly, not somewhere else in
# the code.
# ---------------------------------------------------------------------------

# ===========================================================================
# PARAMETER PACKAGES -- NOT individual constants.
# ===========================================================================
# WHY THIS REPLACED THE OLD SCALAR `K_CLEAR_PER_HOUR`.
#
# v3 of this file set K_CLEAR_PER_HOUR = {"ox": 1.969, "er": 0.5032},
# reading those values off a summary table that called them "the fitted
# values". That was wrong, and the repo says so in two places:
#
#   run_kd_extended.py, beside K_CLEAR = {"er": 0.5032, "ox": 1.969}:
#       "The clearance fits ALSO move d_x (er 0 -> 1.755, ox 0.5 -> 11.11)
#        ... so this sweep applies the decaying input WITHOUT the fitted
#        d_x. This is not neutral. circuit_impact.py separated the two
#        effects and found they pull in OPPOSITE directions."
#
#   profile_reporter_ox.py:103 -- the package 1.969 actually belongs to:
#       REF = dict(r=1.948, k_on=349.4, d_x=11.11, K_act=158.1,
#                  n=4.801, k_clear=1.969)
#
# Applying k_clear = 1.969 on top of the SBML's K_act_ox = 208,
# n_ox = 2.75, d_x_ox = 0.5 therefore mixes FOUR parameters across two
# different fits -- exactly the error the master plan forbids ("as a
# package; they are not mixed from different fits").
#
# So clearance is no longer an independent knob here. It is only reachable
# as a member of a named package that moves every parameter that fit
# moved. Pick one by name, or pick None and inherit whatever the SBML
# itself says. There is deliberately no way to set k_clear alone.
#
# ACTIVE_PACKAGE = None is the honest default: it runs the model as
# committed. Once the BINN plan's Step 1 lands the fitted values IN the
# SBML, None becomes the only setting anyone should ever use and this
# whole block can be deleted.
PARAM_PACKAGES = {
    # The ox clearance/window fit, applied whole. Note it also moves the
    # dose-response calibration (K_act_ox 208 -> 158.1, n_ox 2.75 ->
    # 4.801), which is why it cannot be mixed with DOSE_REF["ox"] = 208
    # without the code saying so (see the warning in report()).
    "ox_window_fit": {
        "ox": {"k_clear": 1.969, "d_x_ox": 11.11, "K_act_ox": 158.1,
               "n_ox": 4.801, "k_on_ox": 349.4},
    },
    # The er M2 (n=4) fit, applied whole. K_act_er and n_er are a package
    # on their own: section 2.6 of the ablation docx shows K_act moving
    # 3.6x (344-1236) depending on n.
    "er_M2_n4": {
        "er": {"k_clear": 0.5032, "d_x_er": 1.755, "K_act_er": 930.0,
               "n_er": 4.0},
    },
    # The scenario v2 actually ran, kept ONLY so its numbers remain
    # reproducible. 1.777 came from the sensitivity_v4 joint fit to
    # Delaunay 2B+2C, whose own calibration used k_off_ox = 150 and
    # d_x_ox = 0.6 -- neither of which is in the SBML today either. It is
    # listed as a package with that provenance attached rather than as a
    # defensible physical value.
    "v2_legacy_1777": {
        "ox": {"k_clear": 1.777},
        "er": {"k_clear": 1.777},
    },
}
ACTIVE_PACKAGE = None     # None = inherit the SBML's own values

# Reference stress dose per branch.
#   ox : K_act_ox = 208, used here as a WORKING REFERENCE VALUE ONLY
#        (Table A of the sensitivity report tags it "identifiable", i.e.
#        estimable in principle -- that is not the same as "confirmed").
#        CORRECTED (post advisor review): this value is NOT independently
#        confirmed. At least four different K_act_ox/n_ox values currently
#        circulate on the team (208/2.75, 217.2/2.51, window-method
#        141-178/3.80-6.02, unmix-method 159-2005/1.87-2.96 -- see the
#        ICHNOS Dry Lab reconciliation notes, point A4), and the
#        calibration methodology behind 208 itself (t_50 matching on a
#        coarse grid, vs. full least-squares; with or without input
#        clearance) has not been confirmed by whoever ran that
#        calibration. Update this constant once that is resolved -- see
#        the Fisher v5.2 report, Section 4.2 and Section 8.
#   er : 2200, one of the ACTUAL Pincus calibration doses -- not
#        K_act_er, which is non-identifiable (a one-parameter family of
#        equivalent solutions, Table B of the same report).
DOSE_REF = {"ox": 208.0, "er": 2200.0}

DOSE_EPS = 0.02          # +/-2% centered finite-difference step for d/d(dose)
T_END_HOURS = 24.0       #6.0        # same simulation window as run_sensitivity_v4 (see its Section 2.3)
N_POINTS = 2000          # time resolution of the simulated trace

MIN_PHOTONS = 10.0        # detection threshold; enforced on BOTH channels (bug already fixed)
SATURATION = 255.0       # SC30 camera, 8-bit per channel -- NOT 65535

# ===========================================================================
# PRE-EQUILIBRATION
# ===========================================================================
# All species start at 0 in the merged SBML. That is not a cell, it is an
# empty vessel. Without pre-equilibration the first hours of the trace are
# dominated by the FILLING of the reporter pools rather than by the stress
# response, and the two have the same timescale, so they are not
# separable. The ablation study measured the cost of omitting it
# (protocol.py line 39: "without pre-eq the dose information is
# underestimated ~5x") and pre-equilibrates 50 h at zero stress.
#
# ORDER MATTERS, AND IT IS A TRAP. Once k_clear is active, S is governed by
# a rate rule (dS/dt = -k_clear*S). Pre-equilibrating with the dose
# already applied would decay it away before the run even starts: 50 h at
# 1.777/h is ~128 half-lives. So pre-eq runs at S = 0 AND k_clear = 0, and
# both are set afterwards. check_pre_eq_convergence() verifies the horizon.
PRE_T_HOURS = 50.0
PRE_T_POINTS = 200

# ===========================================================================
# PHOTON SCALE -- decoupled from the trace, on purpose
# ===========================================================================
# v2 set the photon scale as conv = TARGET_SNR_AT_MAX**2 / max(trace), i.e.
# it anchored the absolute noise level to the MAXIMUM of whatever trace was
# being simulated. That makes the noise level a function of the protocol:
# switching pre-equilibration on raises the trace maximum (ox: 10.70 ->
# 13.96), which drops the conversion to 0.766x and hands every time point
# FEWER photons. Measured consequence: ~40% of the apparent degradation
# from enabling pre-eq was this re-anchoring, not physics. A protocol flag
# must not silently move the noise floor.
#
# So the scale is now an explicit constant in photons per model
# concentration unit. BE CLEAR ABOUT WHAT THIS DOES AND DOES NOT FIX: it
# removes the artifact. It calibrates nothing. 2.33642 is simply the value
# v2's rule happened to produce for the ox reference run, carried over so
# the numbers stay comparable -- it is a PLACEHOLDER with no physical
# basis, and it is the single number that scales every absolute sigma_t
# this script reports (Fisher information is linear in lambda, so
# sigma_t ~ 1/sqrt(PHOTONS_PER_UNIT)). The Photon Transfer Curve
# measurement replaces it.
PHOTON_SCALE_MODE = "fixed"      # "fixed" | "legacy_peak"
PHOTONS_PER_UNIT = 2.33642       # PROVISIONAL -- awaiting the PTC
TARGET_SNR_AT_MAX = 5.0          # only used by "legacy_peak" (v2 regression)

# ===========================================================================
# DETECTION THRESHOLD -- now measured above the baseline
# ===========================================================================
# MIN_PHOTONS was written to answer "is the RESPONSE detectable?". Applied
# as an absolute floor to a pre-equilibrated trace it answers a different
# question -- "is baseline + response detectable?" -- because the
# pre-existing reporter pool already carries photons. Measured: enabling
# pre-eq moved the first "valid" point from 74.2 min to 28.8 min, and that
# earlier crossing is the red BASELINE reaching 10 photons, not the stress
# response becoming visible.
#
# "above_baseline" subtracts each channel's t=0 value before testing. With
# PRE_T_HOURS = 0 the baseline is 0 and this reduces exactly to v2's
# behaviour, so the two modes agree wherever v2 was meaningful.
DETECT_MODE = "above_baseline"   # "above_baseline" | "absolute"


# ---------------------------------------------------------------------------
def run_trace(V, dose, k_clear_per_hour, t_end_hours, n_points, variant_name=None):
    """Return the full simulated time trace (t_hours, Observed_Green,
    mature_red) for one dose, from the already-loaded merged model `V`.

    mature_red is recovered algebraically from the model's own ratio
    output, since Measured_Ratio_RG = mature_red / (Observed_Green + eps):
        mature_red = Measured_Ratio_RG * (Observed_Green + eps)

    The column order [time, Observed_Green, Measured_Ratio_RG, A, TIP] is
    exactly the `selections` list already set in Variant.__init__ -- this
    function does not reimplement or reorder anything, it only reads the
    existing simulation output.

    Parameters
    ----------
    V : Variant
        An already-constructed Variant instance (from run_sensitivity_v4),
        reused across multiple doses for efficiency.
    dose : float
        Stress input level (same units as the STRESS grids in
        run_sensitivity_v4).
    k_clear_per_hour : float
        Clearance rate fed into the model's own `k_clear` parameter
        (added by run_sensitivity_v4.add_clearance at model-build time).
    t_end_hours, n_points : simulation horizon and resolution.
    variant_name : str, optional
        Only used to label a failure message if the simulation does not
        converge (see below) -- has no effect on the simulation itself.
    """
    V.r.resetToOrigin()

    # Apply the active parameter package BEFORE pre-equilibration, so the
    # altered circuit also holds at baseline. (Same discipline as
    # ichnos_ablation.run(): "Overrides are applied BEFORE
    # pre-equilibration so that the lesion also holds at baseline --
    # otherwise you would be comparing circuits that start from different
    # states.") k_clear is excluded here and applied after pre-eq, for the
    # reason given beside PRE_T_HOURS.
    pkg = package_for(variant_name)
    for pname, value in pkg.items():
        if pname == "k_clear":
            continue
        V.r[resolve(V.sbml, pname)] = float(value)

    if PRE_T_HOURS > 0:
        V.r["k_clear"] = 0.0
        V.r[V.S_id] = 0.0
        V.r.simulate(0, PRE_T_HOURS, PRE_T_POINTS)

    V.r["k_clear"] = float(k_clear_per_hour)
    V.r[V.S_id] = float(dose)
    # ADDED (post advisor review, action item 6b): a non-converging
    # integrator run (e.g. the T_END_HOURS=48 CV_CONV_FAILURE case in the
    # Fisher v5.2 report, Section 6.4) used to fail with no record of
    # which branch/dose/window was actually being simulated. Log that
    # before re-raising, so a future 48h (or other) run can actually
    # answer "which branch failed" from its own console output -- do not
    # swallow the exception, only label it.
    try:
        res = np.asarray(V.r.simulate(0, t_end_hours, n_points))
    except Exception as exc:
        label = variant_name if variant_name is not None else "<unknown variant>"
        print(f"  [SIMULATION FAILED] variant={label} dose={dose:g} "
              f"k_clear={k_clear_per_hour:g}/h t_end_hours={t_end_hours:g} "
              f"n_points={n_points} -- {type(exc).__name__}: {exc}")
        raise
    t_h, og, ratio = res[:, 0], res[:, 1], res[:, 2]
    mature_red = ratio * (og + 1e-9)
    return t_h, og, mature_red


def package_for(variant_name):
    """The parameter overrides the active package applies to this branch.

    Empty dict when ACTIVE_PACKAGE is None (inherit the SBML) or when the
    named package says nothing about this branch. Raises on an unknown
    package name rather than silently inheriting -- a typo in a scenario
    name must not look like a clean run.
    """
    if ACTIVE_PACKAGE is None:
        return {}
    if ACTIVE_PACKAGE not in PARAM_PACKAGES:
        raise KeyError(
            f"ACTIVE_PACKAGE={ACTIVE_PACKAGE!r} is not in PARAM_PACKAGES "
            f"({sorted(PARAM_PACKAGES)}). Refusing to run: an unrecognised "
            f"scenario name must not fall back to the SBML defaults "
            f"silently.")
    return dict(PARAM_PACKAGES[ACTIVE_PACKAGE].get(variant_name, {}))


def k_clear_for(variant_name):
    """The clearance rate this run uses, in 1/hour.

    Only reachable through a package. With no package, the SBML's own
    k_clear is used -- which is 0.0 as committed today (add_clearance
    creates the parameter with value 0), i.e. constant S. That is the
    honest reading of the model as it stands, not a chosen scenario.
    """
    return float(package_for(variant_name).get("k_clear", 0.0))


def photon_scale(*trace_maxima):
    """Photons per model concentration unit.

    "fixed" (default): the explicit PHOTONS_PER_UNIT constant, independent
    of the trace, the protocol, and the branch. A placeholder awaiting the
    PTC, but a STABLE one -- changing pre-equilibration or k_clear no
    longer moves the noise floor underneath the comparison.

    "legacy_peak": v2's rule (peak of the reference trace =
    TARGET_SNR_AT_MAX**2 photons). Kept only so v2's published numbers
    stay reproducible for regression; it makes the noise level a function
    of the protocol and should not be used for new results.

    Both channels share one factor either way: the model still has no
    basis for different quantum yields for sfGFP and mCherry (Gap #2,
    still open).
    """
    if PHOTON_SCALE_MODE == "fixed":
        return PHOTONS_PER_UNIT
    if PHOTON_SCALE_MODE == "legacy_peak":
        peak = max(trace_maxima)
        return (TARGET_SNR_AT_MAX ** 2) / max(peak, 1e-12)
    raise ValueError(f"PHOTON_SCALE_MODE={PHOTON_SCALE_MODE!r}")


def detection_mask(green_c, red_c):
    """Which time points count as detectable, on BOTH channels.

    "above_baseline": the signal is measured relative to each channel's
    t=0 value, so the threshold tests the RESPONSE rather than the
    pre-existing pool. Reduces exactly to "absolute" when PRE_T_HOURS = 0,
    because then the baseline is 0.

    "absolute": v2's behaviour, kept for regression only.
    """
    if DETECT_MODE == "above_baseline":
        g = green_c - green_c[0]
        r = red_c - red_c[0]
    elif DETECT_MODE == "absolute":
        g, r = green_c, red_c
    else:
        raise ValueError(f"DETECT_MODE={DETECT_MODE!r}")
    return (g >= MIN_PHOTONS) & (r >= MIN_PHOTONS)


def fisher_matrix(variant_name):
    """Compute the 2x2 Fisher Information Matrix(t) for theta = (dose, t)
    at every simulated time point, at the branch's reference dose.

    Returns
    -------
    t_h : ndarray, shape (n_points,)
        Time axis in hours.
    FIM : ndarray, shape (n_points, 2, 2)
        Fisher Information Matrix at each time point, parameter order
        (dose, t).
    valid : ndarray of bool, shape (n_points,)
        True where both channels exceed MIN_PHOTONS (detection-threshold
        mask, both channels required).
    dose : float
        The reference dose used for this branch.
    """
    V = Variant(variant_name)
    dose = DOSE_REF[variant_name]
    doses = (dose * (1 - DOSE_EPS), dose, dose * (1 + DOSE_EPS))

    # Three runs at neighboring doses, used for a centered finite
    # difference d(lambda)/d(dose). The center run is the one actually
    # reported (t*, SD(t), etc.); the outer two exist only to estimate the
    # dose derivative.
    k_clear = k_clear_for(variant_name)
    traces = [run_trace(V, d, k_clear, T_END_HOURS, N_POINTS,
                         variant_name=variant_name) for d in doses]
    t_h = traces[1][0]

    conv = photon_scale(traces[1][1].max(), traces[1][2].max())
    green = np.array([tr[1] for tr in traces]) * conv   # shape (3, n_points)
    red = np.array([tr[2] for tr in traces]) * conv

    d_dose = doses[2] - doses[0]
    dgreen_ddose = (green[2] - green[0]) / d_dose
    dred_ddose = (red[2] - red[0]) / d_dose

    t_min = t_h * 60.0
    dgreen_dt = np.gradient(green[1], t_min)   # per minute, consistent with v1's time units
    dred_dt = np.gradient(red[1], t_min)

    green_c, red_c = green[1], red[1]

    valid = detection_mask(green_c, red_c)

    n_sat = int(np.sum((green_c > SATURATION) | (red_c > SATURATION)))
    if n_sat:
        print(f"  [!] {variant_name}: {n_sat} time points exceed the SC30 "
              f"saturation limit ({SATURATION:.0f}) -- reconsider the photon "
              f"budget calibration (TARGET_SNR_AT_MAX).")

    n = len(t_h)
    FIM = np.zeros((n, 2, 2))
    # Exact Poisson Fisher information, summed over the two independent
    # channels: FIM = sum_k grad(lambda_k) grad(lambda_k)^T / lambda_k
    for lam, dlam_ddose, dlam_dt in (
            (green_c, dgreen_ddose, dgreen_dt),
            (red_c, dred_ddose, dred_dt)):
        safe_lam = np.maximum(lam, 1e-12)
        FIM[:, 0, 0] += dlam_ddose * dlam_ddose / safe_lam
        FIM[:, 0, 1] += dlam_ddose * dlam_dt / safe_lam
        FIM[:, 1, 0] += dlam_ddose * dlam_dt / safe_lam
        FIM[:, 1, 1] += dlam_dt * dlam_dt / safe_lam

    return t_h, FIM, valid, dose


def crlb_from_fim(t_h, FIM, valid):
    """Invert the Fisher Information Matrix at each time point to obtain
    the Cramer-Rao Lower Bound on Var(dose), Var(t), and their
    correlation.

    Points where FIM is (near-)singular (det <= 1e-30) are left as NaN
    rather than inverted, to avoid reporting a spuriously small variance
    from a numerically unstable inversion.

    ADDED (post advisor review, action item 5g): `det` (the raw FIM
    determinant at every time point, before any masking) is now returned
    too, so the caller can print it around specific points of interest --
    e.g. to confirm that an isolated CRLB(t) spike really is a near-
    singular-FIM artifact, and that a reported "optimal" t* is NOT itself
    sitting on a near-singular point.
    """
    det = np.linalg.det(FIM)
    n = len(t_h)
    crlb_dose = np.full(n, np.nan)
    crlb_t = np.full(n, np.nan)
    corr = np.full(n, np.nan)

    ok = det > 1e-30
    if np.any(ok):
        inv = np.linalg.inv(FIM[ok])
        crlb_dose[ok] = inv[:, 0, 0]
        crlb_t[ok] = inv[:, 1, 1]
        cov = inv[:, 0, 1]
        denom = np.sqrt(np.maximum(crlb_dose[ok] * crlb_t[ok], 1e-30))
        corr[ok] = cov / denom

    # Only time points that pass the detection-threshold mask are eligible
    # to be reported as the "optimal" readout window -- a theoretically
    # sharp CRLB minimum at a point where the signal is not yet detectable
    # is not an achievable operating point.
    crlb_t_valid = np.where(valid, crlb_t, np.nan)
    return crlb_dose, crlb_t, corr, crlb_t_valid, det


def print_det_fim_window(t_h, det, center_idx, label, half_window_min=10.0):
    """ADDED (post advisor review, action item 5g). Print det(FIM) for
    every simulated time point within `half_window_min` minutes of
    `center_idx`, labelled `label`.

    Deliberately generic (works for either branch, either simulation
    window, either detection threshold) rather than hardcoding a specific
    time like "~300 min": the exact location of a near-singular point can
    shift with the configuration, and the whole point of this printout is
    to let the actual numbers -- not an assumption of where they'll land
    -- confirm or refute the near-singularity read of Section 6.5.
    """
    t_min = t_h * 60.0
    center_t = t_min[center_idx]
    in_window = np.abs(t_min - center_t) <= half_window_min
    print(f"  det(FIM) around {label} (t = {center_t:.1f} min):")
    for i in np.where(in_window)[0]:
        flag = "  <-- center" if i == center_idx else ""
        print(f"    t = {t_min[i]:7.2f} min   det(FIM) = {det[i]:.6e}{flag}")


def report(variant_name):
    """Run the full analysis for one branch and print a human-readable
    summary. Returns a dict of the underlying arrays for further use
    (e.g. plotting), or None if no valid readout point was found within
    the simulated window.
    """
    t_h, FIM, valid, dose = fisher_matrix(variant_name)
    crlb_dose, crlb_t, corr, crlb_t_valid, det = crlb_from_fim(t_h, FIM, valid)

    pkg = package_for(variant_name)
    print(f"\n{'=' * 70}\n{variant_name.upper()}  (reference dose = {dose:g}, "
          f"k_clear = {k_clear_for(variant_name):g} /h, "
          f"pre_eq = {PRE_T_HOURS:g} h)\n{'=' * 70}")
    if pkg:
        print(f"  package '{ACTIVE_PACKAGE}' applied in full: "
              + ", ".join(f"{k}={v:g}" for k, v in sorted(pkg.items())))
        # A package that moves the dose-response calibration also moves
        # what DOSE_REF means. Say so rather than letting a reference dose
        # that is no longer near K_act pass unremarked.
        kact_key = f"K_act_{variant_name}"
        if kact_key in pkg:
            print(f"  [!] this package moves {kact_key} to {pkg[kact_key]:g}, "
                  f"so DOSE_REF['{variant_name}'] = {dose:g} is now "
                  f"{dose / pkg[kact_key]:.2f}x K_act, not 1.00x -- the "
                  f"reference dose is no longer the half-activation point "
                  f"and the two must be decided together.")
    else:
        print(f"  no parameter package applied -- running the SBML as "
              f"committed (k_clear = 0, i.e. constant S)")

    if np.all(np.isnan(crlb_t_valid)):
        print(f"  NO time point passes the both-channel detection threshold "
              f"within {T_END_HOURS:g} h. Increase T_END_HOURS or "
              f"TARGET_SNR_AT_MAX and rerun.")
        return None

    i_opt = int(np.nanargmin(crlb_t_valid))
    sd_t_min = np.sqrt(crlb_t[i_opt]) * 60.0
    sd_dose = np.sqrt(crlb_dose[i_opt])

    first_valid = np.argmax(valid) if np.any(valid) else None

    # FRAMING FIX (post advisor review, following the Fisher v5 report's
    # Sections 6.6-6.7): do not call this point "optimal t*" as if it were
    # a chosen, sufficient readout time. Two things must be said alongside
    # the number, every time it is printed, not just in the surrounding
    # report text:
    #   (1) whether this is a genuine interior CRLB(t) minimum, or just the
    #       first point past the detection threshold (i.e. CRLB(t) is
    #       monotonic here and there is no interior minimum at all -- this
    #       is exactly what happens on the ER branch, per the v5 report's
    #       Section 6.5/6.6);
    #   (2) how big SD(t) and SD(dose) actually are relative to the window
    #       and the reference dose, so a "best available" point is never
    #       silently read as "accurate enough" (Section 6.7's point).
    is_boundary_driven = (first_valid is not None) and (i_opt == first_valid)

    print(f"  best-available readout point within this window = "
          f"{t_h[i_opt] * 60:.1f} min")
    if is_boundary_driven:
        print(f"    -> BOUNDARY-DRIVEN: CRLB(t) has no interior minimum here; "
              f"this is simply the first point that clears the "
              f"MIN_PHOTONS={MIN_PHOTONS:g} detection threshold. Do not call "
              f"this an 'optimal' readout time -- earlier detection (lower "
              f"threshold / better SNR) would move it earlier still.")
    else:
        print(f"    -> genuine interior CRLB(t) minimum (not set by the "
              f"detection threshold)")
    print(f"    SD(t)         = {sd_t_min:.2f} min  "
          f"({sd_t_min / (T_END_HOURS * 60.0) * 100:.0f}% of the "
          f"{T_END_HOURS:g}h simulated window)")
    print(f"    SD(dose)      = {sd_dose:.2f}  "
          f"({sd_dose / dose * 100:.0f}% of the reference dose {dose:g})")
    if sd_t_min > T_END_HOURS * 60.0 or sd_dose / dose > 1.0:
        print(f"    -> at this operating point, the estimate is NOT usable: "
              f"'best available' here does not mean 'accurate enough'.")
    print(f"    corr(dose, t) = {corr[i_opt]:.3f}  "
          f"({'substantial dose/time confounding' if abs(corr[i_opt]) > 0.3 else 'mild'})")

    if first_valid is not None:
        print(f"  first valid detection point: {t_h[first_valid] * 60:.1f} min "
              f"(both channels >= {MIN_PHOTONS:g} photon)")

    # ADDED (post advisor review, action item 5g): print det(FIM) around
    # (1) the reported t* itself -- confirms t* is NOT sitting on a near-
    # singular point -- and (2) wherever det(FIM) is smallest WITHIN THE
    # ALREADY-DETECTABLE RANGE (valid==True) -- the generic stand-in for
    # the "isolated ~300 min spike" described in Section 6.5, wherever it
    # actually falls for this configuration.
    #
    # BUG FIXED after the first run of this diagnostic: searching argmin(det)
    # over ALL time points (including before the signal is even detectable)
    # always finds t=0, where det(FIM)=0 trivially because both channels'
    # photon counts (and their gradients) are ~0 before any signal has been
    # produced -- that is a "no signal yet" artifact, not the kind of
    # near-singular-FIM artifact Section 6.5 describes (parallel gradients
    # AFTER the signal is well underway). Restricting the search to the
    # valid/detectable range excludes that trivial pre-signal zero.
    print_det_fim_window(t_h, det, i_opt, "best-available readout point")
    det_in_valid_range = np.where(valid, det, np.inf)
    if np.any(np.isfinite(det_in_valid_range)):
        i_min_det = int(np.argmin(det_in_valid_range))
        if i_min_det != i_opt:
            print_det_fim_window(
                t_h, det, i_min_det,
                "det(FIM) minimum within the detectable range (candidate singularity)")

    return dict(t_h=t_h, crlb_dose=crlb_dose, crlb_t=crlb_t, corr=corr,
                valid=valid, i_opt=i_opt, dose=dose, det=det)


def check_pre_eq_convergence(variant_name="ox", horizons=(20.0, 50.0, 100.0)):
    """Is PRE_T_HOURS long enough, for THIS model?

    v3 copied 50 h from the ablation's protocol.py, whose justification
    ("already converged by 20 h; 50 h is a safe margin") was measured on
    the ablation's own model, not on this merged one with the clearance
    rate rule added. That is an assumption, not a result, until checked.

    Reports the largest relative difference between the traces produced at
    each horizon. If the longer horizons agree to ~1e-6 the baseline has
    converged and 50 h is safe; if they drift, the baseline is still
    moving when the dose lands and every number downstream inherits that.
    """
    global PRE_T_HOURS
    saved = PRE_T_HOURS
    V = Variant(variant_name)
    dose = DOSE_REF[variant_name]
    k_clear = k_clear_for(variant_name)
    out = {}
    try:
        for h in horizons:
            PRE_T_HOURS = h
            t, og, red = run_trace(V, dose, k_clear, T_END_HOURS, N_POINTS,
                                   variant_name=variant_name)
            out[h] = (og.copy(), red.copy())
    finally:
        PRE_T_HOURS = saved

    ref_h = max(horizons)
    print(f"  pre-eq convergence, {variant_name} "
          f"(reference horizon = {ref_h:g} h):")
    for h in horizons:
        if h == ref_h:
            continue
        worst = 0.0
        for a, b in zip(out[h], out[ref_h]):
            denom = np.maximum(np.abs(b), 1e-12)
            worst = max(worst, float(np.max(np.abs(a - b) / denom)))
        verdict = "converged" if worst < 1e-4 else "NOT converged"
        print(f"    {h:5.0f} h vs {ref_h:g} h: max rel. diff = "
              f"{worst:.3e}  -> {verdict}")


if __name__ == "__main__":
    print("=" * 70)
    print("RUN PROVENANCE -- every number below is conditional on these")
    print("=" * 70)
    print(f"  parameter package : {ACTIVE_PACKAGE!r}"
          + ("  (SBML as committed)" if ACTIVE_PACKAGE is None else ""))
    print(f"  pre-equilibration : {PRE_T_HOURS:g} h at zero stress")
    print(f"  photon scale      : {PHOTON_SCALE_MODE} "
          + (f"= {PHOTONS_PER_UNIT:g} photons/unit  <-- PROVISIONAL, "
             f"scales every absolute sigma_t (sigma_t ~ 1/sqrt(scale))"
             if PHOTON_SCALE_MODE == "fixed"
             else f"(peak = {TARGET_SNR_AT_MAX ** 2:g} photons; "
                  f"protocol-dependent, regression only)"))
    print(f"  detection         : {DETECT_MODE}, MIN_PHOTONS = {MIN_PHOTONS:g}")
    print(f"  window            : {T_END_HOURS:g} h, {N_POINTS} points")
    print()
    print("Explicit limitations active in this run:")
    print("  (a) perfect crosstalk correction assumed -- lower bound, not a final number")
    print("  (b) shot-noise-only; no camera read noise / gain. The PTC is still")
    print("      outstanding, and the camera is a COOLED QImaging MicroPublisher")
    print("      3.3 RTV (10-bit sensor, 8-bit saved), not the SC30 this file's")
    print("      header used to name -- see the equipment record, Sept 2026.")
    print("  (c) no autofluorescence baseline / no fragment-escape state")
    print("  (d) PHOTONS_PER_UNIT is a placeholder; absolute sigma_t is NOT yet")
    print("      an anchored quantity. Orderings and correlations are.\n")

    check_pre_eq_convergence("ox")
    print()

    results = {}
    for vn in ("ox", "er"):
        results[vn] = report(vn)

    print(f"\n{'=' * 70}\nSUMMARY FOR THE REPORT\n{'=' * 70}")
    for vn, r in results.items():
        if r is None:
            continue
        print(f"  {vn}: best-available readout point = {r['t_h'][r['i_opt']] * 60:.1f} min, "
              f"corr(dose, t) = {r['corr'][r['i_opt']]:.3f}")