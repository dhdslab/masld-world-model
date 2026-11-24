from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import SRC, load_results
from figstyle import FS_ANN, FULL, GOLD, INK, MM, NAVY, PURPLE, RED, SKY, TEAL, panel, refline, save, setup, tint

COL = {"neural_world_model": NAVY, "hazard_world_model": "#6C7FAA", "logistic_ra_semantic": "#A63A79",
       "resource_allocation": "#F28E2B", "semantic": PURPLE, "degree_product": "#8C510A"}
LAB = {"neural_world_model": "World model", "hazard_world_model": "World model, linear form",
       "logistic_ra_semantic": "Topology + semantic", "resource_allocation": "Resource allocation", "semantic": "Semantic similarity",
       "degree_product": "Degree product"}
XLAB = {"cos": "Semantic proximity", "triad": "Neighbourhood similarity", "proj": "Projected proximity",
        "conv": "Semantic convergence", "drift": "Field-drift alignment", "cn": "Common neighbours", "ra": "Resource allocation",
        "aa": "Adamic–Adar", "jac": "Jaccard", "pa": "Degree product", "n_lo": "Attention, less-studied entity",
        "n_hi": "Attention, more-studied entity", "g_lo": "Growth, slower entity", "g_hi": "Growth, faster entity"}


def ed5(R):
    W = R["world_model"]
    fig = plt.figure(figsize=(FULL, 100 * MM))
    ax = fig.add_axes([0.07, 0.58, 0.40, 0.34])
    py = {int(k): v for k, v in W["papers_by_year"].items()}
    ys = sorted(py)
    col = [tint(NAVY, 0.55) if y < 2020 else (GOLD if y == 2026 else NAVY) for y in ys]
    ax.bar(ys, [py[y] for y in ys], color=col, width=0.75)
    ax.set_xlim(1999.3, 2026.7); ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("Papers")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in (tint(NAVY, 0.55), NAVY, GOLD)],
              ["2000–2019", "2020–2025", "January–June 2026"], loc="upper left", fontsize=FS_ANN, handlelength=1.0)
    ax.set_title("World-model corpus", loc="left")
    panel(ax, "a")
    T = pd.DataFrame(W["task_sizes"]).sort_values("cutoff")
    ax = fig.add_axes([0.58, 0.58, 0.38, 0.34])
    ax.plot(T.cutoff, T.nodes, color=INK, marker="o", ms=2.4, label="Entities in the graph")
    ax.plot(T.cutoff, T.known_edges_in_graph, color=RED, marker="o", ms=2.4, label="Associations already reported")
    ax.set_yscale("log"); ax.set_xticks([2010, 2014, 2018, 2022])
    lo_, hi_ = min(T.nodes.min(), T.known_edges_in_graph.min()), max(T.nodes.max(), T.known_edges_in_graph.max())
    yt = [t for t in (100, 1000, 10000, 100000, 1000000) if lo_ / 3 <= t <= hi_ * 3]
    ax.set_yticks(yt); ax.set_yticklabels([f"{t:,}" for t in yt]); ax.minorticks_off()
    ax.set_ylabel("Count (log scale)"); ax.set_xlabel("Cutoff year")
    ax.legend(loc="lower right", fontsize=FS_ANN)
    ax.set_title("State size by cutoff", loc="left")
    panel(ax, "b")
    ax = fig.add_axes([0.07, 0.10, 0.40, 0.33])
    for d, c in zip([1, 2, 3], ["#74C476", "#31A354", "#00441B"]):
        k = f"positives_{d}"
        g = T[T[k].notna()] if k in T else T.iloc[:0]
        ax.plot(g.cutoff, g[k], color=c, marker="o", ms=2.4, label=f"Δ = {d} year{'s' if d > 1 else ''}")
    ax.set_xticks([2010, 2014, 2018, 2022])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("New associations"); ax.set_xlabel("Cutoff year")
    ax.legend(loc="upper left", fontsize=FS_ANN)
    ax.set_title("Associations first reported after each cutoff", loc="left")
    panel(ax, "c")
    ax = fig.add_axes([0.70, 0.10, 0.26, 0.33])
    co = W["primary"]["hazard_coefficients"]["standardized"]
    keys = [k for k in XLAB if k in co]
    vals = np.array([co[k] for k in keys])
    o = np.argsort(vals)
    ax.barh(range(len(keys)), vals[o], color=SKY, height=0.65)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([XLAB[keys[i]] for i in o])
    ax.axvline(0, color=INK, lw=0.5)
    ax.set_xlabel("Standardized log-hazard coefficient")
    ax.set_title("World model (hazard)", loc="left")
    panel(ax, "d")
    save(fig, "ExtendedDataFig3")


def _ci_row(ax, v, c, y, col, s=9):
    if c is not None and len(c) == 2 and np.all(np.isfinite(c)):
        ax.plot(c, [y, y], color=col, lw=0.8, solid_capstyle="butt")
    ax.scatter([v], [y], s=s, color=col, lw=0, zorder=3)


def ed6(R):
    W = R["world_model"]
    A8 = W["a8"]
    COL.update({"no_language_neural": TEAL})
    LAB.update({"no_language_neural": "No language model (neural)"})
    fig = plt.figure(figsize=(FULL, 112 * MM))
    key = [plt.Line2D([], [], marker="o", lw=0, color=COL[m], ms=3) for m in ["neural_world_model", "semantic", "resource_allocation"]]
    fig.legend(key, [LAB[m] for m in ["neural_world_model", "semantic", "resource_allocation"]], loc="upper left", ncol=3,
               fontsize=FS_ANN, bbox_to_anchor=(0.18, 0.997), frameon=False, handletextpad=0.1, columnspacing=1.2)
    S = pd.DataFrame(W.get("sensitivity", []))
    ax = fig.add_axes([0.18, 0.56, 0.27, 0.37])
    names = {"label_1_papers": "Label: 1 paper", "label_3_papers": "Label: 3 papers",
             "primary_research_labels": "Primary research only", "no_time_decay": "No time decay",
             "shuffled_training_years": "Shuffled training years"}
    rows = [("primary", "Primary analysis")] + [(k, v) for k, v in names.items() if not S.empty and k in set(S.analysis)]
    rows += [("cleaned", "Doubtful entities removed")]
    CL = A8["cleaned"]["metrics"]
    for i, (k, lab) in enumerate(rows[::-1]):
        for m, dy in [("resource_allocation", -0.2), ("semantic", 0.0), ("neural_world_model", 0.2)]:
            if k == "primary":
                r = W["primary"]["metrics"][m]
            elif k == "cleaned":
                r = CL[m]
            else:
                r = S[(S.analysis == k) & (S.method == m)].iloc[0].to_dict()
            c = r.get("auroc_ci")
            _ci_row(ax, r["auroc"], c if isinstance(c, (list, tuple)) else None, i + dy, COL[m])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for _, l in rows][::-1])
    refline(ax, 0.5)
    ax.set_xlim(0.45, 1.01); ax.set_xticks([0.5, 0.75, 1.0]); ax.set_xticklabels(["0.5", "0.75", "1"])
    ax.set_xlabel("AUROC (95% CI)"); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_title("Sensitivity analyses", loc="left")
    panel(ax, "a")
    ax = fig.add_axes([0.66, 0.56, 0.31, 0.37])
    EP = A8["encoders_paired"]["metrics"]
    enc = [("", "PubMedBERT (primary)"), ("__neuml", "Sentence-tuned PubMedBERT"), ("__random", "Random weights")]
    for i, (suf, _) in enumerate(enc[::-1]):
        for m, dy in [("semantic", -0.12), ("neural_world_model", 0.12)]:
            r = EP.get(m + suf)
            if r:
                _ci_row(ax, r["auroc"], r.get("auroc_ci"), i + dy, COL[m])
    ax.set_yticks(range(len(enc))); ax.set_yticklabels([l for _, l in enc][::-1])
    ax.set_ylim(-0.6, len(enc) - 0.4)
    refline(ax, 0.5)
    ax.set_xlim(0.45, 1.01); ax.set_xticks([0.5, 0.75, 1.0]); ax.set_xticklabels(["0.5", "0.75", "1"]); ax.set_xlabel("AUROC (95% CI)")
    ax.set_title("Encoder", loc="left")
    panel(ax, "b")
    EX = A8["exact_primary"]["metrics"]
    ms = [m for m in ["neural_world_model", "no_language_neural", "resource_allocation", "degree_product"] if m in EX]
    mets = [("auroc_degree_matched", "Degree-matched\nAUROC", (0.5, 1.0)), ("entity_mean_auroc", "Per-entity\nAUROC", (0.5, 1.0)),
            ("entity_hits_at_10", "Hits among\n10 partners", (0, 1)), ("entity_mrr", "Mean reciprocal\nrank", (0, 1))]
    for j, (mk, ml, xl) in enumerate(mets):
        ax = fig.add_axes([0.18 + j * 0.088, 0.09, 0.062, 0.32])
        for i, m in enumerate(ms[::-1]):
            ax.scatter([EX[m][mk]], [i], s=10, color=COL[m], lw=0, zorder=3)
        ax.set_xlim(*xl); ax.set_ylim(-0.6, len(ms) - 0.4)
        ax.set_xticks(xl); ax.set_xticklabels([f"{t:g}" for t in xl])
        ax.set_yticks(range(len(ms))); ax.set_yticklabels([LAB[m] for m in ms[::-1]] if j == 0 else [])
        ax.set_xlabel(ml)
        if j == 0:
            ax.set_title("Beyond popularity", loc="left")
            panel(ax, "c")
    SF = A8["state_forecast"]
    ax = fig.add_axes([0.66, 0.09, 0.12, 0.32])
    sem = SF["semantic"]
    ax.bar([0, 1], [sem["cosine_persistence"], sem["cosine_projection"]], color=[tint(NAVY, 0.55), NAVY], width=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Persistence", "Projection"])
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_ylabel("Cosine with realized state")
    ax.set_title("Semantic state", loc="left")
    panel(ax, "d")
    ax = fig.add_axes([0.85, 0.09, 0.12, 0.32])
    att = SF["attention"]
    ax.bar([0, 1], [att["spearman_persistence"], att["spearman_model"]], color=[tint(NAVY, 0.55), NAVY], width=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Persistence", "Transition\nmodel"])
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_ylabel("Spearman correlation with\nnew papers, 2023–2025")
    ax.set_title("Attention", loc="left")
    save(fig, "SupplementaryFig4")


def ed4(R):
    import matplotlib.patheffects as pe
    W = R["world_model"]
    A8 = W["a8"]
    COL.update({"no_language_neural": SKY, "resource_allocation": "#F28E2B", "degree_product": "#B07AA1"})
    LAB.update({"no_language_neural": "No language model (neural)"})
    from figstyle import CardLayout
    C = CardLayout(180, 175)
    fig = C.fig
    C.card(0, 4, 110, 56, "Ablation of state components", "a")
    C.card(114, 4, 66, 56, "Calibration", "b")
    C.card(0, 65, 110, 53, "Sensitivity analyses", "c")
    C.card(114, 65, 66, 53, "Encoder", "d")
    C.card(0, 123, 110, 52, "Beyond popularity", "e")
    C.card(114, 123, 66, 52, "State forecasts", "f")
    key = [plt.Line2D([], [], marker="o", lw=0, color=COL[m], ms=3) for m in ["neural_world_model", "semantic", "resource_allocation"]]
    ax = C.ax(31, 13.5, 75, 33)
    A = pd.DataFrame(W["ablations"])
    AI = A8["ablation_intervals"]
    MET = "auroc_popularity_adjusted"
    full = A[A.removed == "none"].set_index("model")[MET]
    blocks = [b for b in ["semantic_state", "semantic_dynamics", "field_drift", "attention", "graph"] if b in set(A.removed)]
    blocks += ["all_language_features"]
    names = {"semantic_state": "Semantic state", "semantic_dynamics": "Semantic dynamics", "field_drift": "Field drift",
             "attention": "Attention and growth", "graph": "Graph topology", "all_language_features": "All language-model\nfeatures"}
    for mdl, dy in [("neural_world_model", 0.18), ("hazard_world_model", -0.18)]:
        for i, b in zip(np.arange(len(blocks))[::-1], blocks):
            if b == "all_language_features":
                v = AI["all_language_features"][f"all_language_features__{mdl}"][MET] - full[mdl]
            else:
                v = A[(A.removed == b) & (A.model == mdl)][MET].iloc[0] - full[mdl]
            iv = AI["popadj_change_ci"].get(f"{b}__{mdl}")
            ax.barh([i + dy], [v], height=0.34, color=COL[mdl], label=LAB[mdl] if i == len(blocks) - 1 else None)
            if iv:
                ax.plot([-iv[1], -iv[0]], [i + dy] * 2, color=INK, lw=0.7, zorder=4,
                        path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
    ax.axvline(0, color=INK, lw=0.5)
    ax.set_yticks(range(len(blocks))); ax.set_yticklabels([names[b] for b in blocks][::-1])
    ax.set_ylim(-0.6, len(blocks) - 0.4)
    ax.set_xlabel("Change in popularity-adjusted AUROC when removed (95% CI)")
    ax.legend(loc="upper left", fontsize=FS_ANN, handlelength=1.0, frameon=False)
    ax = C.ax(130, 13.5, 46, 33)
    cal = pd.DataFrame(W["calibration"]["bins"])
    k = 1e4
    lo = 10 ** np.floor(np.log10(min(cal.mean_predicted.min(), cal.observed_lo[cal.observed_lo > 0].min()) * k / 1.5))
    hi = 10 ** np.ceil(np.log10(max(cal.mean_predicted.max(), cal.observed_hi.max()) * k * 1.2))
    ax.plot([lo, hi], [lo, hi], color=INK, lw=0.5, ls=(0, (2, 2)))
    for r in cal.itertuples():
        ax.plot([r.mean_predicted * k] * 2, [max(r.observed_lo * k, lo * 1.3), r.observed_hi * k], color=NAVY, lw=0.7)
        if r.observed <= 0:
            ax.scatter([r.mean_predicted * k], [lo * 1.3], marker="v", s=10, color=NAVY, lw=0, zorder=3)
    pos_ = cal[cal.observed > 0]
    ax.plot(pos_.mean_predicted * k, pos_.observed * k, color=NAVY, marker="o", ms=2.6, lw=0.9)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ticks = [t for t in (0.001, 0.01, 0.1, 1, 10, 100, 1000) if lo <= t <= hi]
    xt = [t for t in ticks if t in (0.001, 0.1, 10, 1000)] or ticks
    ax.set_xticks(xt); ax.set_xticklabels([f"{t:g}" for t in xt]); ax.set_yticks(xt); ax.set_yticklabels([f"{t:g}" for t in xt])
    ax.minorticks_off()
    ax.set_xlabel("Predicted risk per 10,000 pairs"); ax.set_ylabel("Observed per 10,000 pairs (95% CI)")
    fig.legend(key, [LAB[m] for m in ["neural_world_model", "semantic", "resource_allocation"]], loc="upper left", ncol=3,
               fontsize=FS_ANN, bbox_to_anchor=(3.0 / 180, 1 - (72.0 - C.TRIM) / C.H), frameon=False, handletextpad=0.1, columnspacing=1.2)
    S = pd.DataFrame(W.get("sensitivity", []))
    ax = C.ax(32, 78.5, 74, 28)
    snames = {"label_1_papers": "Label: 1 paper", "label_3_papers": "Label: 3 papers",
              "primary_research_labels": "Primary research only", "no_time_decay": "No time decay",
              "shuffled_training_years": "Shuffled training years"}
    rows = [("primary", "Primary analysis")] + [(k_, v) for k_, v in snames.items() if not S.empty and k_ in set(S.analysis)]
    rows += [("cleaned", "Doubtful entities removed")]
    CL = A8["cleaned"]["metrics"]
    for i, (k_, lab) in enumerate(rows[::-1]):
        for m, dy in [("resource_allocation", -0.2), ("semantic", 0.0), ("neural_world_model", 0.2)]:
            r_ = W["primary"]["metrics"][m] if k_ == "primary" else (CL[m] if k_ == "cleaned" else
                                                                     S[(S.analysis == k_) & (S.method == m)].iloc[0].to_dict())
            c = r_.get("auroc_ci")
            _ci_row(ax, r_["auroc"], c if isinstance(c, (list, tuple)) else None, i + dy, COL[m])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for _, l in rows][::-1])
    refline(ax, 0.5)
    ax.set_xlim(0.45, 1.01); ax.set_xticks([0.5, 0.75, 1.0]); ax.set_xticklabels(["0.5", "0.75", "1"])
    ax.set_xlabel("AUROC (95% CI)"); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax = C.ax(146.5, 74.5, 29.5, 32)
    EP = A8["encoders_paired"]["metrics"]
    enc = [("", "PubMedBERT (primary)"), ("__neuml", "Sentence-tuned PubMedBERT"), ("__random", "Random weights")]
    for i, (suf, _) in enumerate(enc[::-1]):
        for m, dy in [("semantic", -0.12), ("neural_world_model", 0.12)]:
            r_ = EP.get(m + suf)
            if r_:
                _ci_row(ax, r_["auroc"], r_.get("auroc_ci"), i + dy, COL[m])
    ax.set_yticks(range(len(enc))); ax.set_yticklabels([l for _, l in enc][::-1])
    ax.set_ylim(-0.6, len(enc) - 0.4)
    refline(ax, 0.5)
    ax.set_xlim(0.45, 1.01); ax.set_xticks([0.5, 0.75, 1.0]); ax.set_xticklabels(["0.5", "0.75", "1"]); ax.set_xlabel("AUROC (95% CI)")
    EX = A8["exact_primary"]["metrics"]
    ms = [m for m in ["neural_world_model", "no_language_neural", "resource_allocation", "degree_product"] if m in EX]
    mets = [("auroc_degree_matched", "Degree-matched\nAUROC", (0.5, 1.0)), ("entity_mean_auroc", "Per-entity\nAUROC", (0.5, 1.0)),
            ("entity_hits_at_10", "Hits among\n10 partners", (0, 1)), ("entity_mrr", "Mean reciprocal\nrank", (0, 1))]
    for j, (mk, ml, xl) in enumerate(mets):
        ax = C.ax(33 + j * 19.5, 132, 14.5, 27)
        for i, m in enumerate(ms[::-1]):
            ax.scatter([EX[m][mk]], [i], s=10, color=COL[m], lw=0, zorder=3)
        ax.set_xlim(*xl); ax.set_ylim(-0.6, len(ms) - 0.4)
        ax.set_xticks(xl); ax.set_xticklabels([f"{t:g}" for t in xl])
        ax.set_yticks(range(len(ms))); ax.set_yticklabels([LAB[m] for m in ms[::-1]] if j == 0 else [])
        ax.set_xlabel(ml)
    SF = A8["state_forecast"]
    ax = C.ax(126.5, 134.5, 20, 25)
    sem = SF["semantic"]
    ax.bar([0, 1], [sem["cosine_persistence"], sem["cosine_projection"]], color=[tint(NAVY, 0.55), NAVY], width=0.6)
    for x_, v_ in enumerate([sem["cosine_persistence"], sem["cosine_projection"]]):
        ax.text(x_, v_ + 0.03, f"{v_:.2f}", ha="center", va="bottom", fontsize=FS_ANN, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Unchanged\nstate", "Projected\nstate"], fontsize=FS_ANN)
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_ylabel("Semantic state: cosine\nwith realized state")
    ax = C.ax(158, 134.5, 20, 25)
    att = SF["attention"]
    ax.bar([0, 1], [att["spearman_persistence"], att["spearman_model"]], color=[tint(NAVY, 0.55), NAVY], width=0.6)
    for x_, v_ in enumerate([att["spearman_persistence"], att["spearman_model"]]):
        ax.text(x_, v_ + 0.03, f"{v_:.2f}", ha="center", va="bottom", fontsize=FS_ANN, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Persistence", "Transition\nmodel"])
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_ylabel("Attention: Spearman correlation\nwith new papers, 2023–2025")
    save(fig, "SupplementaryFig4")


def main():
    setup()
    ed4(load_results())


if __name__ == "__main__":
    main()
