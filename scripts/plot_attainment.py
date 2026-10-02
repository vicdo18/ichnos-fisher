"""
ICHNOS -- figure for the CRLB attainment test.

Two panels, answering two different questions:

  (a) WHY the bound fails at low N. Each panel is the cloud of maximum-
      likelihood estimates from independent synthetic measurements, with
      the 95% CRLB confidence ellipse drawn on top. If the bound were
      attained the cloud would fill the ellipse and no more. At N = 1 the
      cloud is visibly larger than the ellipse and displaced from the
      truth -- the estimator is both noisier and biased.

  (b) WHERE it starts to hold, which is the number the imaging protocol
      needs. Ratio of empirical SD to CRLB SD against pooled cell count,
      for timing and for dose separately, with the attainment line at 1.

Run scripts/crlb_attainment.py first; this reads its saved output so the
figure can never show a different run than the table.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

# Validated categorical pair (scripts/validate_palette.js, light surface:
# ALL CHECKS PASS, worst adjacent dE 32.3 protan / 38.3 normal).
C_TIME = "#2563eb"
C_DOSE = "#d97706"
INK = "#1c1c1a"
INK_MUTED = "#6b6b66"
GRID = "#e6e6e2"
SURFACE = "#ffffff"

PANEL_N = (1, 10, 100)      # which N to show as scatter panels
CHI2_95_2DF = 5.991         # 95% contour for a 2-parameter ellipse


def crlb_ellipse(cov, **kw):
    """95% confidence ellipse of a 2x2 covariance, centred on the truth."""
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * np.sqrt(CHI2_95_2DF * vals)
    return Ellipse((0, 0), width=w, height=h, angle=angle, **kw)


def main():
    blob = np.load(os.path.join(RESULTS, "crlb_attainment.npy"),
                   allow_pickle=True)[0]
    rows = blob["rows"]
    C1 = blob["crlb_cov_n1"]
    ln_d0 = blob["ln_dose_true"]
    t0 = blob["t0_min"]
    by_n = {r["n"]: r for r in rows}

    fig = plt.figure(figsize=(11.6, 4.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.35], wspace=0.42)

    # ---- (a) estimate clouds vs the CRLB ellipse -------------------------
    # Each panel is scaled by ITS OWN CRLB standard deviations, so the
    # reference ellipse is the same shape in all three (dividing a
    # covariance by its marginal SDs leaves the correlation, and the
    # correlation does not depend on N because C_N = C_1 / N). The bound
    # is therefore a fixed target and what changes across panels is only
    # whether the estimates fit inside it. Plotting raw units instead
    # would shrink both cloud and ellipse together and show nothing.
    corr_mat = np.array([[1.0, C1[0, 1] / np.sqrt(C1[0, 0] * C1[1, 1])],
                         [C1[0, 1] / np.sqrt(C1[0, 0] * C1[1, 1]), 1.0]])
    LIM = 6.0
    for i, n in enumerate(PANEL_N):
        ax = fig.add_subplot(gs[0, i])
        r = by_n[n]
        est = r["est"]
        sd_d = np.sqrt(C1[0, 0] / n)
        sd_t = np.sqrt(C1[1, 1] / n)
        dx = (est[:, 0] - ln_d0) / sd_d
        dy = (est[:, 1] - t0) / sd_t
        outside = int(np.sum(dx ** 2 + dy ** 2 > LIM ** 2))

        ax.add_patch(crlb_ellipse(corr_mat, facecolor="none", edgecolor=INK,
                                  lw=1.6, ls=(0, (5, 3)), zorder=3))
        ax.scatter(dx, dy, s=9, c=C_TIME, alpha=0.30, linewidths=0,
                   zorder=2, clip_on=True)
        ax.axhline(0, color=GRID, lw=1, zorder=1)
        ax.axvline(0, color=GRID, lw=1, zorder=1)
        ax.plot(0, 0, marker="+", ms=11, mew=2.0, color=INK, zorder=4)

        ax.set_title(f"N = {n}", fontsize=10.5, color=INK, pad=7)
        ax.set_xlabel("dose error / CRLB SD", fontsize=9, color=INK_MUTED)
        if i == 0:
            ax.set_ylabel("timing error / CRLB SD", fontsize=9,
                          color=INK_MUTED)
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_aspect("equal")
        ax.set_xticks([-6, -3, 0, 3, 6])
        ax.set_yticks([-6, -3, 0, 3, 6])
        ax.tick_params(labelsize=8, colors=INK_MUTED, length=3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.set_facecolor(SURFACE)
        if outside:
            ax.annotate(f"{100.0 * outside / len(dx):.0f}% beyond this view",
                        xy=(0.5, -0.30), xycoords="axes fraction",
                        ha="center", fontsize=8, color=INK_MUTED)
        # Direct label instead of a legend box: one annotation, first panel.
        if i == 0:
            ax.annotate("95% CRLB", xy=(0.04, 0.90), xycoords="axes fraction",
                        fontsize=8.5, color=INK)
            # The vertical line of points is real, not a plotting artifact:
            # estimates that ran into the upper dose bound of the search
            # range. At N=1 that is 42% of them.
            ax.annotate("estimates pinned\nat the dose bound",
                        xy=(0.86, -1.2), xytext=(2.6, -4.2), fontsize=7.5,
                        color=INK_MUTED, ha="center",
                        arrowprops=dict(arrowstyle="-", lw=0.8,
                                        color=INK_MUTED))

    # ---- (b) attainment ratio vs N --------------------------------------
    ax = fig.add_subplot(gs[0, 3])
    ns = np.array([r["n"] for r in rows], float)
    ratio_t = np.array([r["sd_t_emp"] / r["sd_t_crlb"] for r in rows])
    ratio_d = np.array([r["sd_d_emp"] / r["sd_d_crlb"] for r in rows])

    ax.axhline(1.0, color=INK_MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate("bound attained", xy=(13, 0.60), fontsize=8.5,
                color=INK_MUTED, ha="left")
    ax.plot(ns, ratio_t, "-o", color=C_TIME, lw=2, ms=8,
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3,
            label="timing")
    ax.plot(ns, ratio_d, "-s", color=C_DOSE, lw=2, ms=8,
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3,
            label="dose")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cells pooled per measurement, N", fontsize=9,
                  color=INK_MUTED)
    ax.set_ylabel("empirical SD / CRLB SD", fontsize=9, color=INK_MUTED)
    ax.set_title("Where the bound becomes honest", fontsize=10.5, color=INK,
                 pad=7)
    ax.set_yticks([0.5, 1, 2, 5])
    ax.set_yticklabels(["0.5", "1", "2", "5"])
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{int(v)}" for v in ns])
    ax.tick_params(labelsize=8, colors=INK_MUTED, length=3)
    ax.grid(True, which="major", axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_facecolor(SURFACE)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right",
                    handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.suptitle(
        "The Cramér–Rao bound is not attainable from a single cell",
        x=0.012, y=0.975, ha="left", fontsize=12.5, color=INK)
    fig.text(0.012, 0.905,
             f"{blob['variant']} branch, readout at t = {t0:g} min. "
             f"Maximum-likelihood recovery of (dose, t) from "
             f"{blob['n_replicates']} synthetic Poisson measurements per N.",
             ha="left", fontsize=8.5, color=INK_MUTED)
    fig.text(0.012, 0.845,
             f"Photon scale {blob['photons_per_unit']:g} ph/unit is PROVISIONAL. "
             f"The search range (dose within a factor of 4 of the reference) acts as "
             f"an implicit prior, and is what the pinned estimates hit.",
             ha="left", fontsize=8.5, color=INK_MUTED)

    fig.subplots_adjust(top=0.70, bottom=0.235, left=0.055, right=0.985)
    out = os.path.join(RESULTS, "crlb_attainment.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"saved -> {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
