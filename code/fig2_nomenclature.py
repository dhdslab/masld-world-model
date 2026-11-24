from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from common import SRC, load_results
from matplotlib.patches import Patch

from figstyle import BAND, DIV, FS_ANN, FULL, INK, INK2, MAIN, MM, MUTED, NEUT, ORIG, CardLayout, panel, refline, save, setup, tint, trim

DIV_NAMES = DIV.reversed()


def main():
    setup()
    R = load_results()["nomenclature"]
    pts = pd.read_csv(SRC / "nomenclature_map_points.csv")
    overall = R["n_current"] / R["n_era_labelled_2023_2026"]
    C = CardLayout(180, 118)
    fig = C.fig
    WARM = "#A0473A"
    C.card(0, 4, 52, 58, "Original text", "a", color=WARM)
    C.card(56, 4, 70, 58, "Disease names neutralized", "b", color=WARM)
    C.card(130, 4, 50, 58, "Separability of naming era", "c", color=WARM)
    C.card(0, 67, 126, 51, "Topics by share of papers using current names", "d", color=WARM)
    C.card(130, 67, 50, 51, "Topic–era dependence", "e", color=WARM)
    norm = TwoSlopeNorm(vmin=0, vcenter=overall, vmax=1)
    maps = [("original", "Original text", "a"), ("neutral", "Disease names neutralized", "b")]
    hb = None
    for i, (key, title, letter) in enumerate(maps):
        ax = C.ax(3.5 + i * 56, 11.5, 45, 47)
        d = pts[pts["map"] == key]
        hb = ax.hexbin(d.x, d.y, C=d.era_current, reduce_C_function=np.mean, gridsize=55, mincnt=5,
                       cmap=DIV_NAMES, norm=norm, linewidths=0, edgecolors="face")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    cax = C.ax(107.5, 20.5, 2.0, 30)
    cb = fig.colorbar(hb, cax=cax, ticks=[0, overall, 1])
    cb.ax.set_yticklabels(["0%", f"{100 * overall:.0f}%", "100%"])
    cb.outline.set_visible(False); cb.ax.tick_params(length=2, width=0.5)
    cax.set_title("Papers using\ncurrent names", fontsize=FS_ANN, loc="left", pad=4)
    ax = C.ax(143.5, 12.0, 33, 36)
    groups = [("era_auroc_2023_2026", "2023-2026"), ("era_auroc_2024_only", "2024 only")]
    drift = [R["neutral"]["drift_2023_vs_2025_current_only"]["mean"], R["neutral"]["drift_2023_vs_2025_legacy_only"]["mean"]]
    DRIFT = tint(NEUT, 0.82)
    ax.axhspan(min(drift), max(drift), color=DRIFT, lw=0)
    for j, (key, lab) in enumerate(groups):
        for k, (run, col) in enumerate([("original", ORIG), ("neutral", NEUT)]):
            r = R[run][key]
            x = j + (k - 0.5) * 0.32
            ax.plot([x, x], [min(r["folds"]), max(r["folds"])], color=col, lw=0.9)
            ax.plot(x, r["mean"], "o", color=col, ms=3.6, label=("Original text" if run == "original" else "Names neutralized") if j == 0 else None)
            ax.text(x + 0.06, r["mean"], f"{r['mean']:.2f}", fontsize=FS_ANN, color=INK2, va="center", ha="left")
    refline(ax, 0.5, "y")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Papers\n2023–2026", "Papers\n2024 only"])
    ax.set_xlim(-0.55, 1.65); ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("AUROC for naming era")
    h_, l_ = ax.get_legend_handles_labels()
    ax.legend(h_ + [Patch(color=DRIFT, lw=0)], l_ + ["Same names, 2023 vs 2025"], loc="lower left", bbox_to_anchor=(0.0, 0.11),
              handlelength=0.8)
    ax = C.ax(18, 75.5, 104, 31)
    rng = np.random.default_rng(1)
    for k, (run, lab, col) in enumerate([("neutral", "Names neutralized", NEUT), ("original", "Original text", ORIG)]):
        sh = np.array(list(R[run]["topic_current_share"].values())) * 100
        y = k + rng.uniform(-0.18, 0.18, len(sh))
        seg = (sh >= 80) | (sh <= 20)
        ax.scatter(sh[~seg], y[~seg], s=7, color=col, alpha=0.35, lw=0)
        ax.scatter(sh[seg], y[seg], s=8, color=col, lw=0)
    for v in (20, 80):
        refline(ax, v, "x")
    pass
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Neutralized", "Original"])
    ax.set_xlim(0, 100); ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Papers in the topic using current names (%)")
    ax.text(10, 1.45, "Segregated", fontsize=FS_ANN, color=INK2, ha="center", va="top")
    ax.text(90, 1.45, "Segregated", fontsize=FS_ANN, color=INK2, ha="center", va="top")
    ax = C.ax(147.5, 75.5, 28.5, 31)
    vals = [R["original"]["nmi_topic_era"], R["neutral"]["nmi_topic_era"]]
    ax.bar([0, 1], vals, width=0.55, color=[ORIG, NEUT])
    for x, v in zip([0, 1], vals):
        ax.text(x, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=FS_ANN, color=INK2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Original", "Neutralized"])
    ax.set_ylabel("NMI, topic vs naming era")
    ax.set_ylim(0, max(vals) * 1.25); ax.set_xlim(-0.6, 1.6)
    pd.DataFrame([{"run": run, "measure": m, "value": R[run][m]["mean"] if isinstance(R[run][m], dict) else R[run][m]}
                  for run in ("original", "neutral") for m in ["era_auroc_2023_2026", "era_auroc_2024_only",
                                                                 "drift_2023_vs_2025_legacy_only", "drift_2023_vs_2025_current_only",
                                                                 "nmi_topic_era", "name_segregated_topics", "topics_evaluated"]]
                 ).to_csv(SRC / "fig2_summary.csv", index=False)
    save(fig, "SupplementaryFig3")


if __name__ == "__main__":
    main()
