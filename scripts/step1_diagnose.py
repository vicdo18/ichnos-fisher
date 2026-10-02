"""
Why did pre-equilibration make the Poisson bound WORSE?

Two candidate causes, and they have different consequences:

  (1) ARTIFACT. photon_conversion_factor() anchors the photon scale to the
      MAXIMUM of the trace (TARGET_SNR_AT_MAX**2 photons at the peak). With
      pre-eq the pools start full, so the max is larger, so every point is
      assigned FEWER photons -> more shot noise everywhere. That is a
      re-normalisation of an arbitrary placeholder, not a statement about
      the device, and it would be fixed by a physically calibrated photon
      scale (the PTC measurement).

  (2) PHYSICS. A pre-filled reporter pool is a constant background: it
      contributes photons (hence Poisson shot noise, in the denominator of
      (dlam/dt)^2 / lam) but no time derivative. Under a Poisson model,
      background with no signal content strictly destroys information.
      Under the ablation's constant-CV lognormal model the trade-off is
      different, because there the noise scales with the signal itself.

Distinguish them: hold the photon scale FIXED between the two runs and see
whether the degradation survives.
"""
import numpy as np
import fisher_info_v3_1 as fi

VARIANT = "ox"
TARGET_MIN = 180.0


def trace_at(pre_t, k_clear, conv_override=None):
    fi.PRE_T_HOURS = pre_t
    V = fi.Variant(VARIANT)
    dose = fi.DOSE_REF[VARIANT]
    doses = (dose * (1 - fi.DOSE_EPS), dose, dose * (1 + fi.DOSE_EPS))
    traces = [fi.run_trace(V, d, k_clear, fi.T_END_HOURS, fi.N_POINTS,
                           variant_name=VARIANT) for d in doses]
    t_h = traces[1][0]
    conv_native = fi.photon_conversion_factor(traces[1][1].max(),
                                              traces[1][2].max())
    conv = conv_native if conv_override is None else conv_override
    green = np.array([tr[1] for tr in traces]) * conv
    red = np.array([tr[2] for tr in traces]) * conv
    d_dose = doses[2] - doses[0]
    dg_dd = (green[2] - green[0]) / d_dose
    dr_dd = (red[2] - red[0]) / d_dose
    t_min = t_h * 60.0
    dg_dt = np.gradient(green[1], t_min)
    dr_dt = np.gradient(red[1], t_min)
    g, r = green[1], red[1]

    n = len(t_h)
    FIM = np.zeros((n, 2, 2))
    for lam, dl_dd, dl_dt in ((g, dg_dd, dg_dt), (r, dr_dd, dr_dt)):
        sl = np.maximum(lam, 1e-12)
        FIM[:, 0, 0] += dl_dd * dl_dd / sl
        FIM[:, 0, 1] += dl_dd * dl_dt / sl
        FIM[:, 1, 0] += dl_dd * dl_dt / sl
        FIM[:, 1, 1] += dl_dt * dl_dt / sl
    return dict(t_min=t_min, conv_native=conv_native, conv_used=conv,
                g=g, r=r, dg_dt=dg_dt, FIM=FIM,
                raw_max_g=traces[1][1].max(), raw_max_r=traces[1][2].max())


def sd_t_at(d, t_target):
    j = int(np.argmin(np.abs(d["t_min"] - t_target)))
    det = np.linalg.det(d["FIM"][j])
    if det <= 1e-30:
        return float("inf"), j
    return np.sqrt(np.linalg.inv(d["FIM"][j])[1, 1]) * 60.0, j


KC = 1.777
A = trace_at(0.0, KC)
B = trace_at(50.0, KC)
# B, but forced onto A's photon scale: isolates cause (2) from cause (1)
B_fixed = trace_at(50.0, KC, conv_override=A["conv_native"])

print("PHOTON SCALE")
print(f"  A (no pre-eq)  raw max green = {A['raw_max_g']:.6g}   "
      f"conversion = {A['conv_native']:.6g} photons/unit")
print(f"  B (pre-eq 50h) raw max green = {B['raw_max_g']:.6g}   "
      f"conversion = {B['conv_native']:.6g} photons/unit")
print(f"  -> the pre-eq run's photon scale is "
      f"{B['conv_native']/A['conv_native']:.3f}x A's "
      f"(same model, same dose, only the anchor moved)")

print("\nSIGNAL AND BACKGROUND, ox, at t = 180 min")
for name, d in (("A  no pre-eq", A), ("B  pre-eq", B),
                ("B' pre-eq, A's photon scale", B_fixed)):
    j = int(np.argmin(np.abs(d["t_min"] - TARGET_MIN)))
    g0 = d["g"][0]
    print(f"  {name:28} green(t=0)={g0:10.3f} ph   "
          f"green(180)={d['g'][j]:10.3f} ph   "
          f"dgreen/dt={d['dg_dt'][j]:+.4g} ph/min")

print("\nSIGMA_T (ox)")
for name, d in (("A  no pre-eq", A), ("B  pre-eq", B),
                ("B' pre-eq, A's photon scale", B_fixed)):
    s180, _ = sd_t_at(d, TARGET_MIN)
    # also the free interior minimum, no detection mask
    det = np.linalg.det(d["FIM"])
    ok = det > 1e-30
    var_t = np.full(len(det), np.inf)
    inv = np.linalg.inv(d["FIM"][ok])
    var_t[ok] = inv[:, 1, 1]
    i = int(np.argmin(var_t))
    print(f"  {name:28} SD(t)@180min = {s180:10.1f} min   "
          f"free min: t*={d['t_min'][i]:7.2f} min  "
          f"SD(t)={np.sqrt(var_t[i])*60:9.2f} min")

print("\nVERDICT")
r_native = sd_t_at(B, TARGET_MIN)[0] / sd_t_at(A, TARGET_MIN)[0]
r_fixed = sd_t_at(B_fixed, TARGET_MIN)[0] / sd_t_at(A, TARGET_MIN)[0]
print(f"  degradation from pre-eq, native photon scale : {r_native:.3f}x")
print(f"  degradation from pre-eq, photon scale held   : {r_fixed:.3f}x")
print("  If the second number is ~1, the whole effect was the photon-scale")
print("  re-anchoring (artifact). If it stays >1, a pre-filled pool really")
print("  does destroy Poisson information (physics).")
