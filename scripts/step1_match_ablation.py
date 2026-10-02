"""
The decisive run: put the Fisher code on the ablation's EXACT protocol
(pre-equilibrated, constant S, no clearance) and see whether its sigma_t
lands near the ablation's 182.4 min at t = 3 h.

If it does, the two implementations are reconciled and the entire 5x was
protocol, not noise model. If it does not, something structural remains.

  E  pre-eq 50h, k_clear = 0   <- the ablation's protocol exactly
  F  no pre-eq,  k_clear = 0   <- isolates pre-eq under constant S
"""
import numpy as np
import fisher_info_v3_1 as fi

TARGET_MIN = 180.0
ABLATION_SIGMA_T = 182.4


def run_cfg(variant, pre_t, k_clear):
    fi.PRE_T_HOURS = pre_t
    fi.K_CLEAR_PER_HOUR = {"ox": k_clear, "er": k_clear}
    t_h, FIM, valid, dose = fi.fisher_matrix(variant)
    crlb_dose, crlb_t, corr, crlb_t_valid, det = fi.crlb_from_fim(
        t_h, FIM, valid)
    t_min = t_h * 60.0

    j = int(np.argmin(np.abs(t_min - TARGET_MIN)))
    if np.linalg.det(FIM[j]) > 1e-30:
        inv = np.linalg.inv(FIM[j])
        sd_t_3h = np.sqrt(inv[1, 1]) * 60.0
        sd_d_3h = np.sqrt(inv[0, 0]) / dose * 100.0
        corr_3h = inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1])
    else:
        sd_t_3h = sd_d_3h = corr_3h = float("nan")

    if np.all(np.isnan(crlb_t_valid)):
        return sd_t_3h, sd_d_3h, corr_3h, None, None
    i = int(np.nanargmin(crlb_t_valid))
    return (sd_t_3h, sd_d_3h, corr_3h, t_min[i], np.sqrt(crlb_t[i]) * 60.0)


CONFIGS = [
    ("A  no pre-eq, kcl 1.777", 0.0,  1.777),
    ("E  pre-eq,    kcl 0", 50.0, 0.0),
    ("F  no pre-eq, kcl 0", 0.0,  0.0),
]

hdr = (f"{'config':26} {'br':3} {'SD(t)@3h':>11} {'SD(d)@3h %':>11} "
       f"{'corr@3h':>8} {'t* (min)':>9} {'SD(t)@t*':>10}")
print("=" * len(hdr))
print(hdr)
print("=" * len(hdr))
for label, pre_t, kcl in CONFIGS:
    for variant in ("ox", "er"):
        s3, d3, c3, tstar, sdstar = run_cfg(variant, pre_t, kcl)
        ts = f"{tstar:9.1f}" if tstar is not None else f"{'--':>9}"
        ss = f"{sdstar:10.2f}" if sdstar is not None else f"{'--':>10}"
        print(f"{label:26} {variant:3} {s3:>11.1f} {d3:>11.1f} "
              f"{c3:>+8.3f} {ts} {ss}")
    print("-" * len(hdr))
print(f"\nablation target (ox, intact, 1 cell, lognormal CV=0.25, 3 h): "
      f"sigma_t = {ABLATION_SIGMA_T:.1f} min, sigma_lnD = 0.915, corr = -0.859")
