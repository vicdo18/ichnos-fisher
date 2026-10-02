"""
Master plan, Level 1 -> Step 1: what does each protocol fix actually cost?

One-variable-at-a-time over the two differences between fisher_info_v2.py
and the ablation study, so the contribution of each is separable instead of
confounded:

  A  no pre-eq,  k_clear 1.777 both    <- v2 as published (must give 929.16)
  B  pre-eq 50h, k_clear 1.777 both    <- pre-equilibration alone
  C  no pre-eq,  k_clear per branch    <- fitted clearance alone
  D  pre-eq 50h, k_clear per branch    <- both (the corrected run)

Reported per branch: the best-available readout point under the real
MIN_PHOTONS=10 threshold, and -- separately -- sigma_t at t = 180 min,
which is the ablation's PRIMARY_T = 3 h and therefore the only
apples-to-apples comparison against its 182 min.
"""
import numpy as np
import fisher_info_v3_1 as fi

CONFIGS = [
    ("A  v2 as published", 0.0,  {"ox": 1.777, "er": 1.777}),
    ("B  + pre-eq only",  50.0,  {"ox": 1.777, "er": 1.777}),
    ("C  + k_clear only",  0.0,  {"ox": 1.969, "er": 0.5032}),
    ("D  + both",         50.0,  {"ox": 1.969, "er": 0.5032}),
]
TARGET_MIN = 180.0   # ablation PRIMARY_T = 3 h

rows = []
for label, pre_t, kcl in CONFIGS:
    fi.PRE_T_HOURS = pre_t
    fi.K_CLEAR_PER_HOUR = kcl
    for variant in ("ox", "er"):
        try:
            t_h, FIM, valid, dose = fi.fisher_matrix(variant)
        except Exception as exc:
            rows.append((label, variant, None, None, None, None, None, None,
                         f"{type(exc).__name__}"))
            continue
        crlb_dose, crlb_t, corr, crlb_t_valid, det = fi.crlb_from_fim(
            t_h, FIM, valid)
        t_min = t_h * 60.0

        # best-available point under the real detection threshold
        if np.all(np.isnan(crlb_t_valid)):
            t_star = sd_t = sd_d = cr = None
            flag = "no point clears MIN_PHOTONS"
        else:
            i = int(np.nanargmin(crlb_t_valid))
            first_valid = int(np.argmax(valid)) if np.any(valid) else None
            t_star = t_min[i]
            sd_t = np.sqrt(crlb_t[i]) * 60.0
            sd_d = np.sqrt(crlb_dose[i]) / dose * 100.0
            cr = corr[i]
            flag = "boundary" if i == first_valid else "interior"

        # sigma_t at the ablation's 3 h readout, no detection mask, so the
        # comparison is against its number and not against our threshold
        j = int(np.argmin(np.abs(t_min - TARGET_MIN)))
        det_all = np.linalg.det(FIM[j])
        if det_all > 1e-30:
            inv = np.linalg.inv(FIM[j])
            sd_t_3h = np.sqrt(inv[1, 1]) * 60.0
        else:
            sd_t_3h = float("inf")

        rows.append((label, variant, t_star, sd_t, sd_d, cr, flag,
                     sd_t_3h, ""))

hdr = (f"{'config':20} {'br':3} {'t* (min)':>9} {'SD(t) @t*':>11} "
       f"{'SD(d) %':>8} {'corr':>7} {'kind':>9} {'SD(t) @3h':>11}")
print("=" * len(hdr))
print(hdr)
print("=" * len(hdr))
last = None
for (label, br, t_star, sd_t, sd_d, cr, flag, sd_t_3h, err) in rows:
    if last is not None and label != last:
        print("-" * len(hdr))
    last = label
    if err:
        print(f"{label:20} {br:3}  FAILED: {err}")
        continue
    if t_star is None:
        print(f"{label:20} {br:3} {'--':>9} {'--':>11} {'--':>8} {'--':>7} "
              f"{flag:>9} {sd_t_3h:>11.1f}")
        continue
    print(f"{label:20} {br:3} {t_star:>9.1f} {sd_t:>11.2f} {sd_d:>8.1f} "
          f"{cr:>+7.3f} {flag:>9} {sd_t_3h:>11.1f}")
print("=" * len(hdr))
print("\nablation decoding_crlb(), ox, intact, 1 cell, lognormal CV=0.25,")
print("at its PRIMARY_T = 3 h, constant S, pre-equilibrated:  sigma_t = 182.4 min")
print("(ichnos_ablation.py / protocol.py; report section 6.1, 'full' row = 3.040 h)")
