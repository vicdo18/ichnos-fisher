"""
Level 2.1 of the master plan: same trajectory, same (dose, t) parameterization,
switchable noise model (poisson vs lognormal CV=c), to settle the ~5x gap
between fisher_info_v2.py (Poisson) and the ablation's decoding_crlb()
(lognormal, CV=0.25) with one shared code path instead of two independent
implementations.
"""
import numpy as np
from fisher_info_v3_1 import (
    Variant, DOSE_REF, DOSE_EPS, K_CLEAR_PER_HOUR, T_END_HOURS, N_POINTS,
    run_trace, photon_conversion_factor, MIN_PHOTONS,
)


def fisher_matrix_noise(variant_name, noise="poisson", cv=0.25):
    V = Variant(variant_name)
    dose = DOSE_REF[variant_name]
    doses = (dose * (1 - DOSE_EPS), dose, dose * (1 + DOSE_EPS))
    traces = [run_trace(V, d, K_CLEAR_PER_HOUR, T_END_HOURS, N_POINTS,
                         variant_name=variant_name) for d in doses]
    t_h = traces[1][0]
    conv = photon_conversion_factor(traces[1][1].max(), traces[1][2].max())
    green = np.array([tr[1] for tr in traces]) * conv
    red = np.array([tr[2] for tr in traces]) * conv
    d_dose = doses[2] - doses[0]
    dgreen_ddose = (green[2] - green[0]) / d_dose
    dred_ddose = (red[2] - red[0]) / d_dose
    t_min = t_h * 60.0
    dgreen_dt = np.gradient(green[1], t_min)
    dred_dt = np.gradient(red[1], t_min)
    green_c, red_c = green[1], red[1]
    valid_phot = (green_c >= MIN_PHOTONS) & (red_c >= MIN_PHOTONS)

    n = len(t_h)
    FIM = np.zeros((n, 2, 2))
    for lam, dlam_ddose, dlam_dt in (
            (green_c, dgreen_ddose, dgreen_dt),
            (red_c, dred_ddose, dred_dt)):
        safe_lam = np.maximum(lam, 1e-12)
        if noise == "poisson":
            w = 1.0 / safe_lam
        elif noise == "lognormal":
            # multiplicative lognormal noise, CV=c on the raw channel ==>
            # additive gaussian noise of std ~ c on ln(lambda); Fisher info
            # per channel = (d ln(lambda)/dtheta)^2 / c^2
            #            = (dlambda/dtheta)^2 / (lambda^2 * c^2)
            w = 1.0 / (safe_lam ** 2 * cv ** 2)
        else:
            raise ValueError(noise)
        FIM[:, 0, 0] += dlam_ddose * dlam_ddose * w
        FIM[:, 0, 1] += dlam_ddose * dlam_dt * w
        FIM[:, 1, 0] += dlam_ddose * dlam_dt * w
        FIM[:, 1, 1] += dlam_dt * dlam_dt * w
    return t_h, FIM, valid_phot, dose, green_c, red_c


def crlb(t_h, FIM, mask):
    det = np.linalg.det(FIM)
    n = len(t_h)
    crlb_dose = np.full(n, np.nan)
    crlb_t = np.full(n, np.nan)
    corr = np.full(n, np.nan)
    ok = mask & (det > 1e-30)
    if not np.any(ok):
        return crlb_dose, crlb_t, corr
    inv = np.linalg.inv(FIM[ok])
    crlb_dose[ok] = inv[:, 0, 0]
    crlb_t[ok] = inv[:, 1, 1]
    corr[ok] = inv[:, 0, 1] / np.sqrt(inv[:, 0, 0] * inv[:, 1, 1])
    return crlb_dose, crlb_t, corr


def report(variant, noise, cv=0.25, use_photon_mask=True, label=""):
    t_h, FIM, valid_phot, dose, green_c, red_c = fisher_matrix_noise(variant, noise, cv)
    mask = valid_phot if use_photon_mask else np.ones_like(valid_phot, dtype=bool)
    crlb_dose, crlb_t, corr = crlb(t_h, FIM, mask)
    if np.all(np.isnan(crlb_t)):
        print(f"{variant:3} {noise:9} {label:22} -> no valid point")
        return
    i = int(np.nanargmin(crlb_t))
    sd_t = np.sqrt(crlb_t[i]) * 60.0
    sd_dose = np.sqrt(crlb_dose[i])
    print(f"{variant:3} {noise:9} {label:22} "
          f"t*={t_h[i]*60:7.2f} min   SD(t)={sd_t:9.2f} min   "
          f"SD(dose)={sd_dose:9.2f} ({sd_dose/dose*100:5.1f}%)   "
          f"corr={corr[i]:+.3f}")


print("=" * 100)
print("Unified CRLB, same trajectory / same (dose,t) parameterization, only the noise model swapped")
print("=" * 100)
for variant in ("ox", "er"):
    # photon-threshold-masked (what the Fisher report actually prints)
    report(variant, "poisson", label="MIN_PHOTONS=10 mask", use_photon_mask=True)
    # pure interior minimum, no detection mask -- the number actually
    # comparable to the ablation's decoding_crlb(), which has no photon
    # detection threshold of its own
    report(variant, "poisson", label="no detection mask", use_photon_mask=False)
    report(variant, "lognormal", cv=0.25, label="CV=0.25, no mask", use_photon_mask=False)
    print()

print("=" * 100)
print("Fixed-readout-time comparison (t = 180 min = PRIMARY_T = 3h, ablation's calibration convention)")
print("=" * 100)
TARGET_MIN = 180.0
for variant in ("ox", "er"):
    for noise in ("poisson", "lognormal"):
        t_h, FIM, valid_phot, dose, green_c, red_c = fisher_matrix_noise(variant, noise, cv=0.25)
        t_min = t_h * 60.0
        i = int(np.argmin(np.abs(t_min - TARGET_MIN)))
        crlb_dose, crlb_t, corr = crlb(t_h, FIM, np.ones_like(valid_phot, dtype=bool))
        sd_t = np.sqrt(crlb_t[i]) * 60.0
        sd_dose = np.sqrt(crlb_dose[i])
        print(f"{variant:3} {noise:9} at t={t_min[i]:6.1f} min   "
              f"SD(t)={sd_t:9.2f} min   SD(dose)={sd_dose:9.2f} ({sd_dose/dose*100:6.1f}%)   "
              f"corr={corr[i]:+.3f}   green={green_c[i]:.1f}ph red={red_c[i]:.1f}ph")
