from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from common import SRC, load_results
from fig5_trials import (ENDPOINTS, EP_COL, PLOT_COL, PLOT_STAGES, SHORT, STAGES, STATUS, STATUS_COL, STATUS_LAB, pooled, stacked)
from figstyle import FS_ANN, FS_LAB, FS_LETTER, INK, MM, NAVY, RED, save, setup, tint

W, H = 180.0, 126.0
BG = "#EEF2F8"
PLOT_COL = {**PLOT_COL, "Steatosis or unspecified": "#FEE391", "Steatohepatitis or fibrosis (non-cirrhotic)": "#FDB863",
            "Compensated cirrhosis enrolled": "#E66101", "Decompensated cirrhosis": "#B2182B", "HCC or transplant": "#67001F"}


def frect(x, y, w, h):
    return [x / W, 1 - (y + h) / H, w / W, h / H]


def rect(ax, x, y, w, h, fc, z=1):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="none", zorder=z))


def header(L, x, y, w, title):
    rect(L, x, y, w, 5.2, NAVY, z=2)
    L.text(x + 1.8, y + 2.7, title, fontsize=FS_LAB, color="white", fontweight="bold", va="center", zorder=5)


def main():
    setup()
    R = load_results()["trial_landscape"]
    fig = plt.figure(figsize=(W * MM, H * MM))
    L = fig.add_axes([0, 0, 1, 1]); L.set_xlim(0, W); L.set_ylim(H, 0); L.axis("off")
    boxes = [(0, 4, 88, 58, "Disease stage enrolled"), (92, 4, 88, 58, "Primary endpoint"),
             (0, 66, 88, 60, "Advanced liver disease of any cause, 2020–2025"), (92, 66, 88, 60, "Reporting of completed trials")]
    for x, y, w, h, t in boxes:
        pass
    for x, y, letter in [(0.2, 0.4, "a"), (92.2, 0.4, "b"), (0.2, 62.4, "c"), (92.2, 62.4, "d")]:
        L.text(x, y, letter, fontsize=FS_LETTER, fontweight="bold", va="top", color=INK)

    ax = fig.add_axes(frect(25, 11.5, 59, 28)); ax.set_facecolor("none")
    per = ["2010-2014", "2015-2019", "2020-2025"]
    rows = [(f"{p.replace('-', '–')}\n(n = {int(sum(R['stage_by_period'][p].values())):,})", pooled(R["stage_by_period"][p])) for p in per]
    rows.append((f"Participants\n(n = {int(sum(R['enrollment_by_stage'].values())):,})", pooled(R["enrollment_share_by_stage"])))
    stacked(ax, rows[::-1], PLOT_STAGES, PLOT_COL, ys=[0, 1.25, 2.25, 3.25])
    ax.set_ylim(-0.5, 3.75)
    ax.axhline(0.62, color=INK, lw=0.4, ls=(0, (2, 1.5)))
    ax.set_xlabel("Share of trials or participants (%)", labelpad=1)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=PLOT_COL[s]) for s in PLOT_STAGES], [SHORT[s] for s in PLOT_STAGES], loc="upper left",
              bbox_to_anchor=(-0.37, -0.33), ncol=2, fontsize=FS_ANN, handlelength=0.9, columnspacing=0.8, labelspacing=0.25,
              handletextpad=0.4, frameon=False)

    ax = fig.add_axes(frect(124, 11.5, 52, 28)); ax.set_facecolor("none")
    eps = R["endpoint_by_stage"]
    late = {e: sum(eps[s][e] for s in ("Decompensated cirrhosis", "Liver cancer (HCC)", "Liver transplantation")) for e in ENDPOINTS}
    rows = [("Steatosis or unspecified", eps[STAGES[0]]), ("Steatohepatitis, F1–F3", eps[STAGES[1]]),
            ("Compensated cirrhosis", eps[STAGES[2]]), ("Decompensated,\nHCC or transplant", late)]
    rows = [(f"{lab} ({sum(d.values())})", d) for lab, d in rows]
    stacked(ax, rows[::-1], ENDPOINTS, EP_COL, ys=list(range(len(rows))))
    for y, (_, d) in zip(range(len(rows)), rows[::-1]):
        k, tot = d["Clinical outcome"], sum(d.values())
        if k and 100 * k / tot >= 8:
            ax.text(100 * k / tot / 2, y, f"{k}", ha="center", va="center", fontsize=FS_ANN, color="white")
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlabel("Primary endpoint (% of trials)", labelpad=1)
    ax.legend([plt.Rectangle((0, 0), 1, 1, facecolor=EP_COL[e], edgecolor=INK if e == "Other" else EP_COL[e], lw=0.3) for e in ENDPOINTS],
              ENDPOINTS, loc="upper left", bbox_to_anchor=(-0.58, -0.33), ncol=3, fontsize=FS_ANN, handlelength=0.9, columnspacing=0.8,
              labelspacing=0.25, handletextpad=0.4, frameon=False)

    ax = fig.add_axes(frect(29, 74, 54, 41)); ax.set_facecolor("none")
    A = pd.read_csv(SRC / "trial_landscape_advanced_any_aetiology.csv")
    grp = {"Compensated cirrhosis enrolled": "Compensated cirrhosis", "Decompensated cirrhosis": "Decompensated cirrhosis",
           "Liver cancer (HCC)": "HCC", "Liver transplantation": "Transplantation"}
    adv_rows = [(grp[c], A[A.stage == c]) for c in grp] + [("All advanced", A[A.stage.isin(list(grp))])]
    for i, (lab, g) in enumerate(adv_rows[::-1]):
        for j, s in enumerate(STATUS):
            v = 100 * (g.masld_status == s).mean()
            y = i + (1 - j) * 0.24
            ax.barh(y, v, height=0.22, color=STATUS_COL[s], edgecolor=NAVY, lw=0.6 if s == "Excluded" else 0)
            ax.text(v + 0.15, y, f"{v:.1f}", va="center", fontsize=FS_ANN, color=INK)
    ax.set_yticks(range(len(adv_rows))); ax.set_yticklabels([f"{lab}\n(n = {len(g):,})" for lab, g in adv_rows[::-1]])
    ax.set_xlim(0, 13.5); ax.set_ylim(-0.55, len(adv_rows) - 0.45)
    ax.set_xlabel("Registrations (%)", labelpad=1)
    ax.legend([plt.Rectangle((0, 0), 1, 1, facecolor=STATUS_COL[s], edgecolor=NAVY, lw=0.6) for s in STATUS], [STATUS_LAB[s] for s in STATUS],
              loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=FS_ANN, handlelength=0.9, labelspacing=0.25, frameon=False)

    ax = fig.add_axes(frect(112, 74, 64, 29)); ax.set_facecolor("none")
    cv = pd.read_csv(SRC / "trial_publication_km_curves.csv")
    pub = R["publication"]
    curves = [("overall", "All completed or terminated trials", NAVY), ("phase:Phase 2-4", "Phase 2–4", RED),
              ("phase:Phase 1 or not applicable", "Phase 1 or not applicable", tint(RED, 0.45))]
    for key, lab, col in curves:
        g = cv[cv.curve == key].sort_values("months")
        g = g[g.months <= 60]
        g = pd.concat([g, g.tail(1).assign(months=60)])
        if key == "overall":
            ax.fill_between(g.months, 100 * g.lo, 100 * g.hi, step="post", color=tint(NAVY, 0.82), lw=0)
        ax.step(g.months, 100 * g.published, where="post", color=col, lw=1.2 if key == "overall" else 0.9, label=lab)
    o36 = pub["overall"]["36"]
    ax.scatter([36], [100 * o36["published"]], s=10, color=NAVY, zorder=4)
    ax.set_xlim(0, 60); ax.set_ylim(0, 65); ax.set_xticks([0, 12, 24, 36, 48, 60])
    ax.set_xlabel("Months after primary completion", labelpad=1)
    ax.set_ylabel("Trials with a results paper (%)")
    ax.legend(loc="upper left", fontsize=FS_ANN, handlelength=1.4, labelspacing=0.25, frameon=False)
    risk = [("All", pub["overall"]["n_at_risk"], NAVY), ("Phase 2–4", pub["by_group"]["phase"]["Phase 2-4"]["km"]["n_at_risk"], RED),
            ("Phase 1/NA", pub["by_group"]["phase"]["Phase 1 or not applicable"]["km"]["n_at_risk"], tint(RED, 0.45))]
    ax.text(-0.1, -0.31, "At risk", transform=ax.transAxes, ha="right", fontsize=FS_ANN, color=INK)
    rend = fig.canvas.get_renderer()
    for k, (lab, nr, col) in enumerate(risk):
        yy = -0.40 - 0.085 * k
        t_ = ax.text(-0.1, yy, lab, transform=ax.transAxes, ha="right", va="center", fontsize=FS_ANN, color=INK)
        x0 = ax.transAxes.inverted().transform(t_.get_window_extent(rend))[0][0]
        mm_ax = 1.0 / (ax.get_window_extent(rend).width / fig.dpi * 25.4)
        ax.plot([x0 - 4.0 * mm_ax, x0 - 1.0 * mm_ax], [yy, yy], transform=ax.transAxes, color=col, lw=1.2,
                clip_on=False)
        for m in (0, 12, 24, 36, 48, 60):
            ax.text(m / 60, yy, f"{nr[str(m)]}", transform=ax.transAxes, ha="center", va="center", fontsize=FS_ANN, color=INK)
    save(fig, "SupplementaryFig6")


if __name__ == "__main__":
    main()
