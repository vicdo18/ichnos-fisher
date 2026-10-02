"""
ICHNOS -- is the Cramer-Rao bound ATTAINABLE, and at how many cells?

WHY THIS SCRIPT EXISTS
======================================================================
fisher_info_v3_1.py computes a Cramer-Rao lower bound on a joint estimate
of theta = (dose, time-since-onset). A CRLB is a *theorem about
estimators*, not a measurement: it says no unbiased estimator can do
better. It does NOT say any estimator does that well. Two things can go
wrong, and both are invisible if you only ever compute F^-1:

  (1) THE BOUND CAN BE LOOSE. The CRLB is attained asymptotically, by the
      maximum-likelihood estimator, as information accumulates. Our
      decoder reads TWO Poisson counts (green, red) from one cell to
      estimate TWO parameters. There is no large-sample limit there at
      all, so the MLE may be badly biased and its variance may exceed
      F^-1 by a lot. Reporting F^-1 as "the decoder's precision" would
      then overstate the device.

  (2) THE IMPLEMENTATION CAN BE WRONG. A sign error or a mis-scaled
      derivative in the FIM produces a confident, plausible, wrong
      number. Nothing in fisher_info_v3_1.py can catch that on its own.

This script tests both at once, by the standard route: simulate synthetic
measurements from the model at a KNOWN (dose, t), estimate (dose, t) back
from them by maximum likelihood, repeat, and compare the empirical
covariance of the estimates against F^-1.

  empirical / CRLB ~ 1   -> the bound is tight AND the FIM is right
  empirical / CRLB >> 1  -> the bound is not attainable at this N

The second outcome is not a failure of this analysis. It is the
decision-relevant result: it says how many cells must be pooled before
the bound becomes an honest description of the decoder, which is exactly
what the imaging protocol needs to specify.

TWO DESIGN DECISIONS THAT MAKE THIS A VALID TEST
======================================================================
A. THE BOUND AND THE ESTIMATOR SHARE ONE lambda(dose, t).
   The estimator needs lambda at arbitrary (dose, t), which means an
   interpolant over a grid of simulations. If the FIM were computed from
   the ODE solver's own gradients while the estimator used the
   interpolant, any interpolation error would show up as apparent
   non-attainment -- we would be measuring our own grid spacing and
   calling it a property of the decoder. So BOTH are computed from the
   same interpolant, and the interpolant-derived FIM is cross-checked
   against fisher_info_v3_1's independently computed FIM before anything
   else runs (see check_fim_consistency). If those disagree, the grid is
   too coarse and the run aborts rather than reporting a ratio.

B. THE PARAMETERIZATION IS (ln dose, t), NOT (dose, t).
   Dose error is naturally relative for a dose-response device, and the
   ablation study's independent implementation uses ln d as well, so
   results are directly comparable. The FIM transforms exactly:
   d/d(ln d) = d * d/d(dose), i.e. F_ln = D F D with D = diag(dose, 1).
   Computing the derivative with respect to ln dose directly, as done
   here, avoids carrying that transformation by hand.

WHAT THIS TEST DOES AND DOES NOT VALIDATE
======================================================================
VALIDATES: the shot-noise (Poisson) bound, the FIM implementation, and
    the attainability of the bound as a function of pooled cell count.
DOES NOT VALIDATE: the forward model itself (that is the job of the
    dose-response fits against Delaunay/Pincus data), the absolute photon
    scale (PHOTONS_PER_UNIT is still a placeholder awaiting the Photon
    Transfer Curve), or anything about cell-to-cell BIOLOGICAL
    variability. Pooling N cells here is pure Poisson pooling: N cells
    contribute Poisson(N*lambda) counts. Real cells also differ from each
    other, which adds a variance floor that does not average away -- that
    is the ablation study's shared-error term, and it can only make the
    true bound worse than what this script reports.

Usage:
    export PYTHONPATH=/path/to/Ichnos_PULSE/python
    python scripts/crlb_attainment.py
"""

import os
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fisher_info_v3_1 as fi  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration -- every choice here is explicit and reported at run time.
# ---------------------------------------------------------------------------

VARIANT = "ox"

# The operating point we are defending. 60 min is the readout time this
# project actually recommends to the wet lab (the CRLB(t) interior minimum
# sits near 46-75 min depending on the detection threshold, and the
# ablation study's age-resolution table is best at 1 h), so the decoder's
# real-world precision at 60 min is the number that matters. Testing at
# the CRLB's own optimum instead would flatter the result.
T0_MIN = 60.0

# Dose grid for the interpolant, as multiplicative factors on DOSE_REF.
# Wide enough that the estimator is not pinned by its own bounds at the
# noise levels tested; fine enough that interpolation error stays well
# below the Poisson noise (verified by check_fim_consistency).
DOSE_FACTORS = np.geomspace(0.25, 4.0, 33)

# Cells pooled per measurement. Poisson pooling only (see header).
N_CELLS = (1, 3, 10, 30, 100, 300, 1000)

N_REPLICATES = 600       # synthetic measurements per N
MULTI_START = 3          # restarts per fit; the (dose,t) likelihood has a
                         # near-degenerate ridge (corr ~ -0.88), so a
                         # single start can stop anywhere along it
SEED = 20261002          # fixed for reproducibility

# Relative tolerance for the FIM cross-check in (A) above.
FIM_CHECK_TOL = 0.05


# ---------------------------------------------------------------------------
# The shared lambda(ln dose, t) surface
# ---------------------------------------------------------------------------

def build_lambda_surface(variant_name):
    """Simulate once per dose and return interpolators for both channels.

    Returns (interp_green, interp_red, ln_dose_grid, t_min_grid, dose_ref)
    where each interpolator maps (ln dose, t in minutes) -> photon count.

    The photon scale is applied here, once, from fisher_info_v3_1 -- so
    this surface is in the same units as the FIM that script reports.
    """
    dose_ref = fi.DOSE_REF[variant_name]
    k_clear = fi.k_clear_for(variant_name)
    V = fi.Variant(variant_name)
    scale = fi.photon_scale()

    doses = dose_ref * DOSE_FACTORS
    green_rows, red_rows, t_min = [], [], None
    for d in doses:
        t_h, og, red = fi.run_trace(V, d, k_clear, fi.T_END_HOURS,
                                    fi.N_POINTS, variant_name=variant_name)
        if t_min is None:
            t_min = t_h * 60.0
        green_rows.append(og * scale)
        red_rows.append(red * scale)

    g = np.asarray(green_rows)
    r = np.asarray(red_rows)
    ln_d = np.log(doses)

    mk = lambda arr: RegularGridInterpolator(  # noqa: E731
        (ln_d, t_min), arr, method="cubic", bounds_error=False, fill_value=None)
    return mk(g), mk(r), ln_d, t_min, dose_ref


def lam1(interp, ln_dose, t_min):
    """One channel's mean at one (ln dose, t), clipped to be positive.

    A Poisson mean must be > 0 for the log-likelihood to exist. Cubic
    interpolation can undershoot slightly below zero where the true trace
    is ~0, so clip rather than let a NaN propagate silently.
    """
    return float(np.maximum(interp(np.array([[ln_dose, t_min]]))[0], 1e-12))


def lam_of(ig, ir, ln_dose, t_min):
    """Both channel means at one (ln dose, t)."""
    return lam1(ig, ln_dose, t_min), lam1(ir, ln_dose, t_min)


def fim_from_surface(ig, ir, ln_dose, t_min, h_ln=1e-3, h_t=0.05):
    """The 2x2 Poisson FIM for (ln dose, t) at one point, in photons.

    Derivatives are central differences ON THE INTERPOLANT, so that the
    bound and the estimator are consistent by construction (header, A).
    Units: ln dose is dimensionless, t is minutes, so Var(t) comes out in
    min^2 and Var(ln d) is a squared relative error.
    """
    F = np.zeros((2, 2))
    for interp in (ig, ir):
        lam = lam1(interp, ln_dose, t_min)
        dl_dlnd = (lam1(interp, ln_dose + h_ln, t_min)
                   - lam1(interp, ln_dose - h_ln, t_min)) / (2 * h_ln)
        dl_dt = (lam1(interp, ln_dose, t_min + h_t)
                 - lam1(interp, ln_dose, t_min - h_t)) / (2 * h_t)
        grad = np.array([dl_dlnd, dl_dt])
        F += np.outer(grad, grad) / lam
    return F


def check_fim_consistency(ig, ir, ln_dose, t_min, variant_name):
    """Cross-check the interpolant FIM against fisher_info_v3_1's own.

    The module computes the FIM in (dose, t) from the ODE output directly.
    Transforming it to (ln dose, t) is exact: F_ln = D F D with
    D = diag(dose, 1). If the two agree, the interpolant resolves the
    model well enough that any non-attainment this script reports is a
    property of the estimator and not of the grid. If they do not, the
    grid is too coarse and we must not report a ratio at all.
    """
    t_h, FIM, valid, dose = fi.fisher_matrix(variant_name)
    i = int(np.argmin(np.abs(t_h * 60.0 - t_min)))
    D = np.diag([dose, 1.0])
    F_module = D @ FIM[i] @ D
    F_surface = fim_from_surface(ig, ir, ln_dose, t_min)

    print("  FIM cross-check at the operating point "
          f"(t = {t_h[i] * 60:.2f} min), (ln dose, t) parameterization:")
    worst = 0.0
    for (a, b), label in (((0, 0), "F[lnd,lnd]"), ((0, 1), "F[lnd,t]"),
                          ((1, 1), "F[t,t]")):
        m, s = F_module[a, b], F_surface[a, b]
        rel = abs(s - m) / max(abs(m), 1e-30)
        worst = max(worst, rel)
        print(f"    {label:11} module = {m:12.6e}   surface = {s:12.6e}"
              f"   rel. diff = {rel:.2e}")
    if worst > FIM_CHECK_TOL:
        raise RuntimeError(
            f"Interpolant FIM disagrees with the module FIM by {worst:.1%} "
            f"(tolerance {FIM_CHECK_TOL:.0%}). The dose/time grid is too "
            f"coarse to compare an estimator against this bound -- refine "
            f"DOSE_FACTORS or N_POINTS rather than reporting a ratio.")
    print(f"    -> consistent to {worst:.2%}; the interpolant resolves the "
          f"model adequately.\n")
    return F_surface


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------

def neg_loglik(params, yg, yr, n_cells, ig, ir):
    """Negative Poisson log-likelihood of pooled counts from n_cells.

    N cells contribute Poisson(N * lambda) counts, so the mean scales with
    N while the parameters do not. Terms in log(y!) are dropped: they do
    not depend on theta and so do not affect the argmax.
    """
    ln_dose, t_min = params
    lg, lr = lam_of(ig, ir, ln_dose, t_min)
    mg, mr = n_cells * lg, n_cells * lr
    return -(yg * np.log(mg) - mg + yr * np.log(mr) - mr)


def fit_once(yg, yr, n_cells, ig, ir, bounds, starts):
    """Maximum-likelihood (ln dose, t) from one synthetic measurement.

    Multi-start because the likelihood has a near-degenerate ridge. Returns
    (ln_dose_hat, t_hat, hit_boundary).
    """
    best, best_val = None, np.inf
    for s in starts:
        res = minimize(neg_loglik, s, args=(yg, yr, n_cells, ig, ir),
                       method="L-BFGS-B", bounds=bounds)
        if res.fun < best_val:
            best_val, best = res.fun, res.x
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    on_edge = bool(np.any(np.isclose(best, lo, rtol=0, atol=1e-6))
                   or np.any(np.isclose(best, hi, rtol=0, atol=1e-6)))
    return best[0], best[1], on_edge


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------

def run():
    print("=" * 78)
    print("CRLB ATTAINMENT TEST -- is the bound reachable, and at what N?")
    print("=" * 78)
    print(f"  variant            : {VARIANT}")
    print(f"  operating point    : t = {T0_MIN:g} min, "
          f"dose = DOSE_REF = {fi.DOSE_REF[VARIANT]:g}")
    print(f"  parameterization   : (ln dose, t in minutes)")
    print(f"  parameter package  : {fi.ACTIVE_PACKAGE!r}")
    print(f"  pre-equilibration  : {fi.PRE_T_HOURS:g} h")
    print(f"  photon scale       : {fi.PHOTON_SCALE_MODE} "
          f"= {fi.PHOTONS_PER_UNIT:g} photons/unit (PROVISIONAL)")
    print(f"  replicates per N   : {N_REPLICATES}, seed {SEED}")
    print(f"  noise model        : Poisson shot noise only; no cell-to-cell")
    print(f"                       biological variability (see header)")
    print()

    ig, ir, ln_d_grid, t_grid, dose_ref = build_lambda_surface(VARIANT)
    ln_d0 = float(np.log(dose_ref))

    F = check_fim_consistency(ig, ir, ln_d0, T0_MIN, VARIANT)

    lg0, lr0 = lam_of(ig, ir, ln_d0, T0_MIN)
    print(f"  channel means at the operating point: green = {lg0:.2f} ph, "
          f"red = {lr0:.2f} ph  (per cell)")

    det = np.linalg.det(F)
    if det <= 0:
        print("  FIM is singular at this point -- the bound is infinite and "
              "there is nothing to attain. Stopping.")
        return
    C1 = np.linalg.inv(F)          # CRLB covariance at N = 1
    crlb_corr = C1[0, 1] / np.sqrt(C1[0, 0] * C1[1, 1])
    print(f"  CRLB at N=1: SD(ln dose) = {np.sqrt(C1[0, 0]):.3f}  "
          f"({np.sqrt(C1[0, 0]) * 100:.0f}% relative), "
          f"SD(t) = {np.sqrt(C1[1, 1]):.1f} min, corr = {crlb_corr:+.3f}\n")

    bounds = [(ln_d_grid[0], ln_d_grid[-1]), (float(t_grid[0]), float(t_grid[-1]))]
    # Starts: the truth, plus points spread along the degenerate ridge, so
    # that a success is not an artifact of starting at the answer.
    starts = [np.array([ln_d0, T0_MIN]),
              np.array([ln_d0 - 0.5, T0_MIN * 0.6]),
              np.array([ln_d0 + 0.5, T0_MIN * 1.6]),
              np.array([ln_d0 - 0.2, T0_MIN * 1.8]),
              np.array([ln_d0 + 0.2, T0_MIN * 0.4])][:MULTI_START]

    rng = np.random.default_rng(SEED)
    hdr = (f"{'N':>5} {'SD(t) emp':>10} {'SD(t) CRLB':>11} {'ratio':>7} "
           f"{'SD(lnd) emp':>12} {'SD(lnd) CRLB':>13} {'ratio':>7} "
           f"{'bias t':>8} {'corr emp':>9} {'edge':>6}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for n in N_CELLS:
        yg = rng.poisson(n * lg0, N_REPLICATES)
        yr = rng.poisson(n * lr0, N_REPLICATES)

        est = np.empty((N_REPLICATES, 2))
        edge = 0
        for k in range(N_REPLICATES):
            # A channel that reads zero carries no information about
            # theta through log(lambda); it is a real measurement outcome,
            # so it is kept, and the optimizer handles it via the -lambda
            # term alone.
            a, b, on_edge = fit_once(float(yg[k]), float(yr[k]), n,
                                     ig, ir, bounds, starts)
            est[k] = (a, b)
            edge += int(on_edge)

        Cn = C1 / n                      # CRLB scales as F^-1 / N
        sd_t_crlb = np.sqrt(Cn[1, 1])
        sd_d_crlb = np.sqrt(Cn[0, 0])
        sd_t_emp = float(np.std(est[:, 1], ddof=1))
        sd_d_emp = float(np.std(est[:, 0], ddof=1))
        bias_t = float(np.mean(est[:, 1]) - T0_MIN)
        corr_emp = float(np.corrcoef(est[:, 0], est[:, 1])[0, 1])

        print(f"{n:>5} {sd_t_emp:>10.2f} {sd_t_crlb:>11.2f} "
              f"{sd_t_emp / sd_t_crlb:>7.2f} {sd_d_emp:>12.3f} "
              f"{sd_d_crlb:>13.3f} {sd_d_emp / sd_d_crlb:>7.2f} "
              f"{bias_t:>8.2f} {corr_emp:>+9.3f} "
              f"{100.0 * edge / N_REPLICATES:>5.0f}%")

        rows.append(dict(n=n, sd_t_emp=sd_t_emp, sd_t_crlb=sd_t_crlb,
                         sd_d_emp=sd_d_emp, sd_d_crlb=sd_d_crlb,
                         bias_t=bias_t, corr_emp=corr_emp,
                         edge_frac=edge / N_REPLICATES,
                         est=est.copy()))

    print()
    print("READING THIS TABLE")
    print("  ratio ~ 1      : the bound is attained -- F^-1 is an honest")
    print("                   description of the decoder at this N, and the")
    print("                   FIM implementation is corroborated.")
    print("  ratio >> 1     : the bound is not attainable at this N. The")
    print("                   quoted sigma_t understates the real error.")
    print("  'edge'         : fraction of fits stopped at a parameter bound,")
    print("                   i.e. measurements carrying so little")
    print("                   information that the estimate ran to the edge")
    print("                   of the searched range. These are not failures")
    print("                   of the optimizer; they are what a decoder does")
    print("                   with an uninformative observation.")
    print("  'bias t'       : mean(t_hat) - t_true, in minutes. The CRLB")
    print("                   assumes an unbiased estimator; a large bias")
    print("                   here means the bound does not apply as stated.")
    # Persist everything the figure needs, so plotting does not have to
    # re-run the experiment (and cannot silently plot a different one).
    out = dict(rows=rows, crlb_cov_n1=C1, ln_dose_true=ln_d0,
               t0_min=T0_MIN, variant=VARIANT, lam_green=lg0, lam_red=lr0,
               photons_per_unit=fi.PHOTONS_PER_UNIT,
               active_package=fi.ACTIVE_PACKAGE, seed=SEED,
               n_replicates=N_REPLICATES)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "results", "crlb_attainment.npy")
    np.save(path, np.array([out], dtype=object), allow_pickle=True)
    print(f"\n  saved -> {os.path.normpath(path)}")
    return out


if __name__ == "__main__":
    run()
