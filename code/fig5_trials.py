from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import SRC, load_results
from figstyle import FS_ANN, FULL, INK, INK2, MM, NAVY, PURPLE, RED, panel, save, setup, tint, trim

STAGES = ["Steatosis or unspecified", "Steatohepatitis or fibrosis (non-cirrhotic)", "Compensated cirrhosis enrolled",
          "Decompensated cirrhosis", "Liver cancer (HCC)", "Liver transplantation", "Hepatic impairment or healthy volunteers"]
STAGE_COL = dict(zip(STAGES, ["#F9D88C", "#F4A259", "#E8855A", "#B23A48", "#5B1A3A", "#5B1A3A", "#A7B4CE"]))
PLOT_STAGES = ["Steatosis or unspecified", "Steatohepatitis or fibrosis (non-cirrhotic)", "Compensated cirrhosis enrolled",
               "Decompensated cirrhosis", "HCC or transplant", "Hepatic impairment or healthy volunteers"]
PLOT_COL = {**STAGE_COL, "HCC or transplant": "#5B1A3A"}
SHORT = {"Steatosis or unspecified": "Steatosis or unspecified", "Steatohepatitis or fibrosis (non-cirrhotic)": "Steatohepatitis, F1–F3",
         "Compensated cirrhosis enrolled": "Compensated cirrhosis", "Decompensated cirrhosis": "Decompensated cirrhosis",
         "HCC or transplant": "HCC or transplantation", "Hepatic impairment or healthy volunteers": "Phase 1, healthy or hepatic impairment"}
ENDPOINTS = ["Clinical outcome", "Histology", "Imaging or biomarker", "Pharmacokinetics or safety", "Other"]
EP_COL = dict(zip(ENDPOINTS, ["#1B7837", PURPLE, "#76B7B2", "#C2C78A", "#E3E7EF"]))
STATUS = ["Targeted", "Listed as eligible", "Excluded"]
STATUS_LAB = {"Targeted": "MASLD-specific", "Listed as eligible": "MASLD listed as eligible", "Excluded": "MASLD excluded"}
STATUS_COL = {"Targeted": NAVY, "Listed as eligible": "#8193BD", "Excluded": "white"}


def pooled(d):
    out = {k: d.get(k, 0) for k in STAGES if k not in ("Liver cancer (HCC)", "Liver transplantation")}
    out["HCC or transplant"] = d.get("Liver cancer (HCC)", 0) + d.get("Liver transplantation", 0)
    return out


def stacked(ax, rows, order, colours, ys):
    for (lab, d), y in zip(rows, ys):
        tot = sum(d.get(s, 0) for s in order); left = 0.0
        for s in order:
            w = 100 * d.get(s, 0) / tot if tot else 0
            ax.barh(y, w, left=left, height=0.62, color=colours[s], lw=0.4, edgecolor="white")
            left += w
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 100)


def main():
    setup()
    R = load_results()["trial_landscape"]
    fig = plt.figure(figsize=(FULL, 124 * MM))

    ax = fig.add_axes([0.13, 0.665, 0.30, 0.25])
    per = ["2010-2014", "2015-2019", "2020-2025"]
    rows = [(f"{p.replace('-', '–')}\n(n = {int(sum(R['stage_by_period'][p].values())):,})", pooled(R["stage_by_period"][p])) for p in per]
    n_part = int(sum(R["enrollment_by_stage"].values()))
    rows.append((f"Participants\n(n = {n_part:,})", pooled(R["enrollment_share_by_stage"])))
    stacked(ax, rows[::-1], PLOT_STAGES, PLOT_COL, ys=[0, 1.25, 2.25, 3.25])
    ax.set_ylim(-0.5, 3.75)
    ax.axhline(0.62, color=INK2, lw=0.4, ls=(0, (2, 1.5)))
    ax.set_xlabel("Share of trials or participants (%)")
    adv = R["advanced_share_of_targeted"]
    ax.set_title("Disease stage enrolled", loc="left", x=-0.33)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=PLOT_COL[s]) for s in PLOT_STAGES], [SHORT[s] for s in PLOT_STAGES], loc="upper left",
              bbox_to_anchor=(-0.33, -0.27), ncol=3, fontsize=FS_ANN, handlelength=0.9, columnspacing=0.8, labelspacing=0.25,
              handletextpad=0.4)
    panel(ax, "a")

    ax = fig.add_axes([0.70, 0.665, 0.27, 0.25])
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
    ax.set_xlabel("Primary endpoint (% of trials)")
    ce = R["clinical_endpoint_share"]
    ax.set_title("Primary endpoint", loc="left", x=-0.66)
    ax.legend([plt.Rectangle((0, 0), 1, 1, facecolor=EP_COL[e], edgecolor=INK2 if e == "Other" else EP_COL[e], lw=0.3) for e in ENDPOINTS],
              ENDPOINTS, loc="upper left", bbox_to_anchor=(-0.66, -0.27), ncol=3, fontsize=FS_ANN, handlelength=0.9, columnspacing=0.8,
              labelspacing=0.25, handletextpad=0.4)
    panel(ax, "b")

    ax = fig.add_axes([0.16, 0.085, 0.27, 0.33])
    A = pd.read_csv(SRC / "trial_landscape_advanced_any_aetiology.csv")
    grp = {"Compensated cirrhosis enrolled": "Compensated cirrhosis", "Decompensated cirrhosis": "Decompensated cirrhosis",
           "Liver cancer (HCC)": "HCC", "Liver transplantation": "Transplantation"}
    adv_rows = [(grp[c], A[A.stage == c]) for c in grp] + [("All advanced", A[A.stage.isin(list(grp))])]
    for i, (lab, g) in enumerate(adv_rows[::-1]):
        for j, s in enumerate(STATUS):
            v = 100 * (g.masld_status == s).mean()
            y = i + (1 - j) * 0.24
            ax.barh(y, v, height=0.22, color=STATUS_COL[s], edgecolor=NAVY, lw=0.6 if s == "Excluded" else 0)
            ax.text(v + 0.15, y, f"{v:.1f}", va="center", fontsize=FS_ANN, color=INK2)
    ax.set_yticks(range(len(adv_rows))); ax.set_yticklabels([f"{lab}\n(n = {len(g):,})" for lab, g in adv_rows[::-1]])
    ax.set_xlim(0, 13.5); ax.set_ylim(-0.55, len(adv_rows) - 0.45)
    ax.set_xlabel("Registrations (%)")
    st = R["advanced_staged_masld_status_share"]
    ax.set_title("Advanced liver disease, any cause, 2020–2025", loc="left", x=-0.33)
    ax.legend([plt.Rectangle((0, 0), 1, 1, facecolor=STATUS_COL[s], edgecolor=NAVY, lw=0.6) for s in STATUS], [STATUS_LAB[s] for s in STATUS],
              loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=FS_ANN, handlelength=0.9, labelspacing=0.25)
    panel(ax, "c")

    ax = fig.add_axes([0.62, 0.175, 0.35, 0.245])
    cv = pd.read_csv(SRC / "trial_publication_km_curves.csv")
    pub = R["publication"]
    curves = [("overall", "All completed or terminated trials", NAVY), ("phase:Phase 2-4", "Phase 2–4", RED),
              ("phase:Phase 1 or not applicable", "Phase 1 or not applicable", tint(RED, 0.5))]
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
    ax.legend(loc="upper left", fontsize=FS_ANN, handlelength=1.4, labelspacing=0.25)
    risk = [("All", pub["overall"]["n_at_risk"], NAVY), ("Phase 2–4", pub["by_group"]["phase"]["Phase 2-4"]["km"]["n_at_risk"], RED),
            ("Phase 1/NA", pub["by_group"]["phase"]["Phase 1 or not applicable"]["km"]["n_at_risk"], tint(RED, 0.2))]
    ax.text(-0.09, -0.25, "At risk", transform=ax.transAxes, ha="right", fontsize=FS_ANN, color=INK2)
    for k, (lab, nr, col) in enumerate(risk):
        yy = -0.33 - 0.075 * k
        ax.text(-0.09, yy, lab, transform=ax.transAxes, ha="right", va="center", fontsize=FS_ANN, color=col)
        for m in (0, 12, 24, 36, 48, 60):
            ax.text(m / 60, yy, f"{nr[str(m)]}", transform=ax.transAxes, ha="center", va="center", fontsize=FS_ANN, color=col)
    ax.set_title("Reporting of completed trials", loc="left", x=-0.14)
    panel(ax, "d")

    rows = [("advanced_share", adv["share"]), ("decomp_or_hcc_share", R["decomp_or_hcc_share_of_targeted"]),
            ("clinical_endpoint_all", ce["all"]), ("clinical_endpoint_phase3", ce["phase3"]),
            ("published_36m", o36["published"]), ("results_posted_any", pub["results_posted_any"]),
            ("advanced_staged_n", R["advanced_staged_n"])] + [(f"advanced_staged_{k}", v) for k, v in st.items()]
    pd.DataFrame(rows, columns=["quantity", "value"]).to_csv(SRC / "fig5_summary.csv", index=False)
    trim(fig, [(124, 4), (60, 7)])
    save(fig, "trial_landscape_overview")


if __name__ == "__main__":
    raise SystemExit("Retired figure script (2026-11-20): its main() would overwrite Figure6, which is now another figure. "
                     "Only its helpers are imported (fig1_design, fig_trials_ed).")
