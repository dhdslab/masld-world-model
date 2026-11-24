from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from common import load_results
from figstyle import BROWN, FS_ANN, FULL, GOLD, INK, MM, NAVY, PURPLE, SKY, CardLayout, panel, refline, save, setup, tint

KEY = "foundation_world_model_dev" if "--dev" in sys.argv else "foundation_world_model"
ORANGE, PINK, GREEN = "#F28E2B", "#B07AA1", "#59A14F"
METH = [("hazard_v1_neural", "World model (same run)", NAVY),
        ("world_model", "Foundation-model extension", GREEN),
        ("history_embeddings", "History + extension\nembeddings", tint(GREEN, 0.45)),
        ("history_language_state", "History + PubMedBERT state", PURPLE),
        ("history", "History only", SKY),
        ("resource_allocation", "Resource allocation", ORANGE),
        ("degree_product", "Degree product", PINK)]
COL = {k: c for k, _, c in METH}
LAB = {k: l for k, l, _ in METH}


def ci_rows(ax, rows, vk, ck):
    for i, (k, m) in enumerate(rows):
        v, c = m.get(vk), m.get(ck)
        if v is None or not np.isfinite(v):
            continue
        if c is not None and all(np.isfinite(c)):
            ax.plot(c, [i, i], color=COL[k], lw=1.2, solid_capstyle="round")
        ax.scatter([v], [i], s=14, color=COL[k], zorder=3, edgecolor="white", lw=0.5)


def guides(ax, n):
    return None


def fig4(R):
    F = R[KEY]
    P = F["primary"]
    M, CP = P["metrics"], P["contrasts_popularity_adjusted"]
    D = F["dynamics"]
    C = CardLayout(180, 128)
    fig = C.fig
    C.card(0, 4, 60, 59, "Every test task", "a", color=GREEN)
    C.card(64, 4, 70, 59, "Primary task", "b", color=GREEN)
    C.card(138, 4, 42, 59, "Precision", "c", color=GREEN)
    C.card(0, 69, 60, 59, "What each component adds", "d", color=GREEN)
    C.card(64, 69, 70, 59, f"Latent state, {D['cutoff']} → {D['target_year']}", "e", color=GREEN)
    C.card(138, 69, 42, 59, "Research activity", "f", color=GREEN)
    G = pd.DataFrame(F["grid"])
    tasks = sorted({(r.cutoff, r.horizon) for r in G.itertuples()})
    for j, (vk, dk, xl) in enumerate([("auroc", "diff:world_model_minus_hazard_v1_neural", "AUROC difference,\nextension − world model"),
                                      ("auroc_popularity_adjusted", "diffpop:world_model_minus_hazard_v1_neural",
                                       "Popularity-adjusted\ndifference")]):
        ax = C.ax(15.5 + j * 22.0, 12.5, 18.5, 38)
        guides(ax, len(tasks))
        for i, (T, d) in enumerate(tasks[::-1]):
            g = G[(G.cutoff == T) & (G.horizon == d)].set_index("method")
            v = g.loc["world_model", vk] - g.loc["hazard_v1_neural", vk]
            c = g.loc[dk, "auroc_ci"]
            pre = bool(g.loc["world_model", "prespecified_task"])
            ax.plot(c, [i, i], color=GREEN, lw=1.0, solid_capstyle="round")
            ax.scatter([v], [i], s=12 if pre else 14, marker="o" if pre else "D", color=GREEN, edgecolor="white", lw=0.4,
                       zorder=3)
        ax.axvline(0, color=INK, lw=0.5, ls=(0, (2, 2)))
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels([f"{T}, Δ = {d}" for T, d in tasks[::-1]] if j == 0 else [])
        ax.tick_params(axis="y", labelsize=FS_ANN, length=0)
        ax.set_ylim(-0.6, len(tasks) - 0.4)
        ax.set_xlabel(xl, fontsize=FS_ANN + 0.5)
    ks = [k for k, *_ in METH if k in M][::-1]
    rows = [(k, M[k]) for k in ks]
    for j, (vk, ck, x0, xl) in enumerate([("auroc", "auroc_ci", 97.5, "AUROC\n(95% CI)"),
                                          ("auroc_popularity_adjusted", "auroc_popularity_adjusted_ci", 115.5,
                                           "Popularity-adjusted\nAUROC (95% CI)")]):
        ax = C.ax(x0, 12.5, 15.0, 38)
        guides(ax, len(ks))
        ci_rows(ax, rows, vk, ck)
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels([LAB[k] for k in ks] if j == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0.7, 1.0)
        ax.set_xticks([0.7, 0.8, 0.9, 1.0]); ax.set_xticklabels(["0.7", "0.8", "0.9", "1"])
        ax.set_ylim(-0.6, len(ks) - 0.4)
        ax.set_xlabel(xl)
    EX = F["exact_primary"]
    KC = EX["topk_ci"]
    ax = C.ax(153, 12.5, 24, 38)
    kf = [k for k in ("hazard_v1_neural", "world_model", "history", "resource_allocation") if f"{k}@1000" in KC]
    groups = [(1000, "Top\n1,000"), (10000, "Top\n10,000")]
    for gi, (kk, lab) in enumerate(groups[::-1]):
        for j, k in enumerate(kf):
            yv = gi + 0.3 - j * 0.2
            c = KC[f"{k}@{kk}"]
            ax.plot([100 * c[0], 100 * c[1]], [yv, yv], color=COL[k], lw=1.0, solid_capstyle="round")
            ax.scatter([100 * EX["metrics"][k][f"precision_at_{kk}"]], [yv], s=12, color=COL[k], edgecolor="white", lw=0.4,
                       zorder=3)
    ax.set_yticks(range(len(groups))); ax.set_yticklabels([g[1] for g in groups[::-1]])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.55, len(groups) - 0.45); ax.set_xlim(0, None)
    ax.set_xlabel("Reported within\n3 years (%, 95% CI)")
    ax = C.ax(29.5, 77.5, 27, 38)
    full = M["world_model"]["auroc_popularity_adjusted"]
    blocks = {"language": "Language block", "attention": "Attention block", "context": "Context block",
              "rollout": "Latent rollouts", "timesfm": "TimesFM activity\nforecasts"}
    crow = [(lab, full - M[f"minus_{g}"]["auroc_popularity_adjusted"],
             CP.get(f"world_model_minus_minus_{g}"), GREEN) for g, lab in blocks.items()]
    crow.append(("Embeddings added\nto history", M["history_embeddings"]["auroc_popularity_adjusted"]
                 - M["history"]["auroc_popularity_adjusted"], CP.get("history_embeddings_minus_history"), tint(GREEN, 0.45)))
    crow.append(("PubMedBERT state\nadded to history", M["history_language_state"]["auroc_popularity_adjusted"]
                 - M["history"]["auroc_popularity_adjusted"], CP.get("history_language_state_minus_history"), PURPLE))
    crow.append(("Graph encoder instead\nof PubMedBERT state", M["history_embeddings"]["auroc_popularity_adjusted"]
                 - M["history_language_state"]["auroc_popularity_adjusted"],
                 CP.get("history_embeddings_minus_history_language_state"), tint(GREEN, 0.45)))
    for i, (lab, v, c, col) in enumerate(crow[::-1]):
        ax.barh([i], [v], height=0.62, color=col)
        if c:
            ax.plot(c, [i, i], color=INK, lw=0.7, zorder=4)
    ax.axvline(0, color=INK, lw=0.5)
    ax.set_yticks(range(len(crow))); ax.set_yticklabels([x[0] for x in crow[::-1]], fontsize=FS_ANN + 0.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(crow) - 0.4)
    ax.set_xlabel("Gain in popularity-adjusted\nAUROC (95% CI)")
    meth = [("rollout_forecast_activity", "Rollout, forecast activity", GREEN),
            ("rollout_observed_activity", "Rollout, observed activity", tint(GREEN, 0.3)),
            ("rollout_shuffled_activity", "Rollout, shuffled activity", tint(GREEN, 0.3)),
            ("rollout_no_action", "Transition without actions", tint(GREEN, 0.3)),
            ("ridge", "Ridge on state and change", INK), ("drift", "Mean drift", INK),
            ("timesfm_latent", "TimesFM on each dimension", BROWN)]
    meth = [m for m in meth if m[0] in D["skill_vs_locf"]]
    ax = C.ax(97.5, 77.5, 32, 38)
    guides(ax, len(meth))
    for i, (k, lab, col) in enumerate(meth[::-1]):
        c = D["skill_ci"][k]
        ax.plot([100 * c[0], 100 * c[1]], [i, i], color=col, lw=1.0, solid_capstyle="round")
        ax.scatter([100 * D["skill_vs_locf"][k]], [i], s=14, color=col, edgecolor="white", lw=0.4, zorder=3)
    ax.axvline(0, color=INK, lw=0.5, ls=(0, (2, 2)))
    ax.set_yticks(range(len(meth))); ax.set_yticklabels([m[1] for m in meth[::-1]], fontsize=FS_ANN + 0.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(meth) - 0.4)
    ax.set_xlabel("Error reduction against\nthe unchanged state (%)")
    ax = C.ax(152.5, 77.5, 24.5, 38)
    A = F["attention_forecast"]
    yrs = [A["cutoff"] + 1 + i for i in range(len(A["by_year"]["timesfm"]))]
    ax.plot(yrs, A["by_year"]["last_observation"], color=INK, lw=0.8, ls=(0, (2, 2)), marker="o", ms=3.4, mfc="none", mew=0.7,
            label="Last year's value", zorder=4)
    for k, lab, col in (("timesfm", "TimesFM", BROWN), ("timesfm_adapter_no_embedding", "TimesFM + adapter", tint(GREEN, 0.45)),
                        ("timesfm_adapter", "TimesFM + adapter\nwith embedding", GREEN)):
        ax.plot(yrs, A["by_year"][k], color=col, marker="s" if k == "timesfm" else "o", ms=3.0, mec="white", mew=0.4, lw=1.2,
                label=lab)
    vals = [v for k in A["by_year"] for v in A["by_year"][k]]
    ax.set_ylim(min(vals) - 0.005, max(vals) + 0.05)
    ax.set_xticks(yrs); ax.set_xlim(yrs[0] - 0.3, yrs[-1] + 0.3)
    ax.set_xlabel("Forecast year"); ax.set_ylabel("Mean absolute error,\nlog(1 + papers per year)")
    ax.legend(loc="upper left", fontsize=FS_ANN, handlelength=1.4, labelspacing=0.25, frameon=False, borderaxespad=0.1)
    save(fig, "Figure6")


def main():
    setup()
    fig4(load_results())


if __name__ == "__main__":
    main()
