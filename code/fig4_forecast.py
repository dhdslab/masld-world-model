from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from common import SRC, load_results
from figstyle import FS_ANN, FULL, INK, MM, NAVY, PURPLE, SKY, TEAL, CardLayout, panel, refline, save, setup, tint

ENC = "biomedbert"
NOLANG_COL = TEAL
METH = [("neural_world_model", "World model", NAVY),
        ("hazard_world_model", "World model, linear form", "#6C7FAA"),
        ("no_language_neural", "No language model (neural)", SKY),
        ("gbdt_no_language", "No language model\n(boosted trees)", tint(SKY, 0.45)),
        ("logistic_ra_semantic", "Topology + semantic", "#A63A79"),
        ("resource_allocation", "Resource allocation", "#F28E2B"),
        ("semantic", "Semantic similarity", PURPLE),
        ("degree_product", "Degree product", "#B07AA1")]
COL = {k: c for k, _, c in METH}
LAB = {k: l for k, l, _ in METH}
TP_LAB = {"Chemical-Disease": "Chemical–disease", "Disease-Gene": "Gene–disease", "Chemical-Gene": "Chemical–gene",
          "Disease-Disease": "Disease–disease", "Gene-Gene": "Gene–gene", "Chemical-Chemical": "Chemical–chemical",
          "Disease-Variant": "Variant–disease", "Gene-Variant": "Variant–gene", "Chemical-Variant": "Variant–chemical",
          "Variant-Variant": "Variant–variant"}
ROC_SHOWN = ("neural_world_model", "no_language_neural", "resource_allocation", "semantic")


def roc_panel(ax, df, aurocs, title, compact=False):
    for k in ROC_SHOWN[::-1]:
        g = df[(df.method == k) & (df.fpr > 0)]
        if g.empty:
            continue
        ax.step(g.fpr, g.tpr, where="post", color=COL[k], lw=1.3 if k == "neural_world_model" else 0.9,
                label=f"{LAB[k].split(chr(10))[0]} {aurocs[k]:.3f}" if compact else LAB[k], zorder=3 if k == "neural_world_model" else 2)
    xs = np.logspace(-5, 0, 50)
    ax.plot(xs, xs, color=INK, lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.set_xscale("log"); ax.set_xlim(1e-5, 1); ax.set_ylim(0, 1)
    ax.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]); ax.set_xticklabels(["0.00001", "0.0001", "0.001", "0.01", "0.1", "1"])
    ax.minorticks_off()
    ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xlabel("False-positive rate (log scale)"); ax.set_ylabel("True-positive rate")
    h, l = ax.get_legend_handles_labels()
    if compact:
        ax.legend(h[::-1], l[::-1], loc="upper left", fontsize=FS_ANN, handlelength=1.0, borderaxespad=0.1, labelspacing=0.25)
    else:
        ax.legend(h[::-1], l[::-1], loc="upper left", fontsize=5.0, handlelength=0.9, borderaxespad=0.1, labelspacing=0.2,
                  handletextpad=0.4)
    if title:
        ax.set_title(title, loc="left")


def guides(ax, n):
    return None


def ci_dots(ax, rows, x0, metric_ci, value_key, ci_key):
    for i, (k, m) in enumerate(rows):
        v, c = m.get(value_key), m.get(ci_key)
        if v is None or not np.isfinite(v):
            continue
        if c is not None and all(np.isfinite(c)):
            ax.plot(c, [i, i], color=COL[k], lw=1.2, solid_capstyle="round")
        ax.scatter([v], [i], s=14, color=COL[k], zorder=3, edgecolor="white", lw=0.5)


def fig4(R):
    W = R["world_model"]
    P, A8 = W["primary"], W["a8"]
    EX = A8["exact_primary"]
    C = CardLayout(180, 128)
    fig = C.fig
    C.card(0, 4, 58, 59, f"Associations first reported {P['label_window'].replace('-', '–')}", "a")
    C.card(62, 4, 66, 59, "Primary task", "b")
    C.card(132, 4, 48, 59, "Precision over all candidates", "c")
    C.card(0, 69, 58, 59, "Every test task", "d")
    C.card(62, 69, 66, 59, "Reported January–June 2026", "e")
    C.card(132, 69, 48, 59, "Evaluation design", "f")
    ax = C.ax(11.5, 13.5, 43, 38)
    roc = pd.read_csv(SRC / f"wm_roc_exact_primary_{ENC}.csv")
    roc_panel(ax, roc, {k: v["auroc"] for k, v in EX["metrics"].items()}, "")
    M = {**P["metrics"], **A8["baselines"]["metrics"]}
    ks = [k for k, *_ in METH if k in M][::-1]
    rows = [(k, M[k]) for k in ks]
    for j, (vk, ck, x0, xl) in enumerate([("auroc", "auroc_ci", 92.0, "AUROC\n(95% CI)"),
                                          ("auroc_popularity_adjusted", "auroc_popularity_adjusted_ci", 111.0,
                                           "Popularity-adjusted\nAUROC (95% CI)")]):
        ax = C.ax(x0, 12.5, 15.0, 38)
        guides(ax, len(ks))
        ci_dots(ax, rows, x0, None, vk, ck)
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels([LAB[k] for k in ks] if j == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0.7, 1.0)
        ax.set_xticks([0.7, 0.8, 0.9, 1.0]); ax.set_xticklabels(["0.7", "0.8", "0.9", "1"])
        ax.set_ylim(-0.6, len(ks) - 0.4)
        ax.set_xlabel(xl)
    ax = C.ax(143.5, 12.5, 33, 39)
    pr = pd.read_csv(SRC / f"wm_pr_exact_primary_{ENC}.csv")
    KC = EX.get("topk_ci", {})
    shown = ["semantic", "degree_product", "resource_allocation", "no_language_neural", "hazard_world_model", "neural_world_model"]
    prev = EX["n_positives"] / EX["n_candidates"]
    y_floor = max(0.005, 100 * prev / 2)
    for k in shown:
        g = pr[(pr.method == k) & (100 * pr.precision >= y_floor)]
        ax.plot(g.k, 100 * g.precision, color=COL[k], lw=1.4 if k == "neural_world_model" else 0.85, label=LAB[k].replace("\n", " "))
    for k, dx in (("neural_world_model", 1.0), ("hazard_world_model", 1.3), ("resource_allocation", 1.7)):
        for kk in (1000, 10000):
            c = KC.get(f"{k}@{kk}")
            if c:
                ax.plot([kk * dx] * 2, [100 * max(c[0], 1e-4), 100 * c[1]], color=COL[k], lw=0.8)
                ax.scatter([kk * dx], [100 * EX["metrics"][k][f"precision_at_{kk}"]], s=9, color=COL[k], edgecolor="white",
                           lw=0.4, zorder=3)
    refline(ax, 100 * prev, "y")
    ax.text(2.6e6, 100 * prev * 1.25, f"prevalence {100 * prev:.3f}%", ha="right", va="bottom", fontsize=5.0, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(10, 3e6); ax.set_ylim(y_floor, 100)
    ax.set_xticks([10, 1e2, 1e3, 1e4, 1e5, 1e6]); ax.set_xticklabels(["10", "", "1,000", "", "100,000", ""])
    yt = [t for t in (0.01, 0.1, 1, 10, 100) if ax.get_ylim()[0] <= t <= 100]
    ax.set_yticks(yt); ax.set_yticklabels([f"{t:g}" for t in yt]); ax.minorticks_off()
    ax.set_xlabel("Highest-ranked pairs, k"); ax.set_ylabel("Reported within 3 years (%)")
    h_, l_ = ax.get_legend_handles_labels()
    ax.legend(h_[::-1], l_[::-1], loc="lower left",
              bbox_to_anchor=(0.0, 0.19), fontsize=5.0, handlelength=1.0, labelspacing=0.15, handletextpad=0.4,
              borderaxespad=0.1, frameon=False)
    G = pd.DataFrame(W["grid"])
    GI = pd.DataFrame(A8["grid_intervals"])
    hz = sorted(G.horizon.unique())
    for j, d in enumerate(hz):
        ax = C.ax(13, 76.5 + j * 13.2, 41.5, 11.2)
        for k, dx, cik in [("resource_allocation", 0.12, "ra_auroc_ci"), ("neural_world_model", -0.12, "neural_auroc_ci")]:
            g = G[(G.horizon == d) & (G.method == k)].sort_values("cutoff")
            for r in g.itertuples():
                iv = GI[(GI.cutoff == r.cutoff) & (GI.horizon == d)]
                if not iv.empty:
                    ax.plot([r.cutoff + dx] * 2, iv[cik].iloc[0], color=COL[k], lw=1.0)
                pre = bool(r.cutoff in (2020, 2021, 2022, 2023))
                ax.scatter([r.cutoff + dx], [r.auroc], s=12 if pre else 14, marker="o" if pre else "D", color=COL[k],
                           edgecolor="white", lw=0.4, zorder=3)
        ax.set_ylim(0.93, 1.0); ax.set_yticks([0.94, 0.97, 1.0]); ax.set_yticklabels(["0.94", "0.97", "1"])
        ax.set_xlim(2019.5, 2024.5); ax.set_xticks([2020, 2021, 2022, 2023, 2024])
        if j < len(hz) - 1:
            ax.set_xticklabels([])
        ax.text(0.02, 0.06, f"Δ = {d} year{'s' if d > 1 else ''}", transform=ax.transAxes, fontsize=FS_ANN, va="bottom", ha="left")
        if j == len(hz) - 1:
            ax.set_xlabel("Cutoff year T")
        if j == 1:
            ax.set_ylabel("AUROC (95% CI)")
    ax = C.ax(73.5, 78.5, 51, 38)
    EH = A8["exact_holdout"]
    roc = pd.read_csv(SRC / f"wm_roc_exact_holdout_{ENC}.csv")
    roc_panel(ax, roc, {k: v["auroc"] for k, v in EH["metrics"].items()}, "")
    ax = C.ax(146, 78.5, 30.5, 38)
    DC = W["design_contrast"]
    ks = ["degree_product", "resource_allocation", "semantic"]
    guides(ax, len(ks))
    for i, k in enumerate(ks[::-1]):
        a, b = DC["random_holdout_never_negatives"][k], DC["temporal_primary"][k]
        ax.plot([a, b], [i + 0.13, i - 0.13], color=tint(COL[k], 0.5), lw=0.9, zorder=1)
        for v, cik, filled in ((a, DC.get("random_holdout_ci", {}).get(k), False), (b, DC.get("temporal_primary_ci", {}).get(k), True)):
            if cik:
                ax.plot(cik, [i + (0.13 if not filled else -0.13)] * 2, color=COL[k], lw=0.9)
            ax.scatter([v], [i + (0.13 if not filled else -0.13)], s=16, color=COL[k] if filled else "white", edgecolor=COL[k],
                       lw=0.8, zorder=3)
    ax.set_yticks(range(len(ks))); ax.set_yticklabels([LAB[k].replace(" ", "\n", 1) for k in ks[::-1]])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.7, 1.0); ax.set_xticks([0.7, 0.8, 0.9, 1.0]); ax.set_xticklabels(["0.7", "0.8", "0.9", "1"])
    ax.set_ylim(-0.6, len(ks) - 0.4)
    ax.set_xlabel("AUROC (95% CI)")
    ax.legend([Line2D([], [], marker="o", lw=0, mfc="white", mec=INK, ms=3.2, mew=0.7),
               Line2D([], [], marker="o", lw=0, color=INK, ms=3.2)], ["Random edge hold-out", "Temporal"],
              loc="upper left", fontsize=FS_ANN, handletextpad=0.1, borderaxespad=0.2)
    save(fig, "Figure4")


def dots(ax, rows, order, labels, title, xlabel=True, methods=("resource_allocation", "semantic", "neural_world_model")):
    offs = np.linspace(-0.22, 0.22, len(methods))
    for i, lv in enumerate(order[::-1]):
        for k, dy in zip(methods, offs):
            r = rows[(rows.level == lv) & (rows.method == k)]
            if r.empty:
                continue
            r = r.iloc[0]
            c = r.get("auroc_ci")
            if isinstance(c, (list, tuple)) and len(c) == 2 and all(np.isfinite(c)):
                ax.plot(c, [i + dy] * 2, color=COL[k], lw=0.9, solid_capstyle="round")
            ax.scatter([r.auroc], [i + dy], s=12, color=COL[k], zorder=3, edgecolor="white", lw=0.4)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([labels[l] for l in order[::-1]])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0.40, 1.01); ax.set_xticks([0.5, 0.75, 1.0]); ax.set_xticklabels(["0.5", "0.75", "1"])
    refline(ax, 0.5)
    ax.tick_params(axis="y", length=0)
    if xlabel:
        ax.set_xlabel("AUROC (95% CI)")
    if title:
        ax.set_title(title, loc="left")


def fig5(R):
    W = R["world_model"]
    S = pd.DataFrame(W["subgroups"])
    C = CardLayout(180, 66.5)
    fig = C.fig
    AX_W = 31.0
    XA, XBC, XD = 33.0, 89.0, 146.0
    C.card(0, 4, 68, 66, "By entity type", "a")
    C.card(71, 4, 54, 31, "Shared neighbours", "b")
    C.card(71, 39, 54, 31, "Advanced liver disease", "c")
    C.card(127, 4, 53, 66, "Reported first in", "d")
    shown = ["neural_world_model", "semantic", "resource_allocation"]
    key = [Line2D([], [], marker="o", lw=0, color=COL[k], ms=3.2, mec="white", mew=0.4) for k in shown]
    fig.legend(key, [LAB[k] for k in shown], loc="upper left", ncol=3, fontsize=FS_ANN, frameon=False, handletextpad=0.1,
               columnspacing=1.2, bbox_to_anchor=(3.0 / 180, 1 - (62.5 - C.TRIM) / C.H))
    ax = C.ax(XA, 12.5, AX_W, 38)
    tp = S[(S.group == "type_pair") & (S.method == "neural_world_model")].sort_values("n_pos", ascending=False)
    order = tp.level.tolist()
    lab = {l: f"{TP_LAB.get(l, l)} ({int(tp[tp.level == l].n_pos.iloc[0]):,})" for l in order}
    dots(ax, S[S.group == "type_pair"], order, lab, "")
    ax = C.ax(XBC, 12.2, AX_W, 11.5)
    g = S[S.group == "common_neighbours"]
    nb = lambda lv: int(g[(g.level == lv) & (g.method == "neural_world_model")].n_pos.iloc[0])
    dots(ax, g, ["none", "one_or_more"], {"none": f"None ({nb('none'):,})", "one_or_more": f"≥1 ({nb('one_or_more'):,})"}, "")
    ax = C.ax(XBC, 47.2, AX_W, 11.5)
    g = S[S.group == "advanced_liver_disease"]
    na = lambda lv: int(g[(g.level == lv) & (g.method == "neural_world_model")].n_pos.iloc[0])
    dots(ax, g, ["involves", "does_not_involve"], {"involves": f"Advanced ({na('involves'):,})",
                                                     "does_not_involve": f"Other ({na('does_not_involve'):,})"}, "")
    ax = C.ax(XD, 12.5, AX_W, 46)
    g = S[S.group == "first_reports_include"]
    order = [l for l in ["interventional", "observational", "basic", "synthesis"] if l in set(g.level)]
    names = {"interventional": "Trial", "observational": "Observational", "basic": "Basic", "synthesis": "Review or\nguideline"}
    dots(ax, g, order, {l: f"{names[l]}\n({int(g[(g.level == l) & (g.method == 'neural_world_model')].n_pos.iloc[0]):,})"
                        for l in order}, "")
    save(fig, "Figure5")


def main():
    setup()
    R = load_results()
    fig4(R)
    fig5(R)


if __name__ == "__main__":
    main()
