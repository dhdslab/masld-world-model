from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from scipy.cluster.hierarchy import dendrogram

from common import TOPICS_ANA, REF_RUN, SRC, load_results
from figstyle import (DOMAIN_COLOR, DOMAIN_SHORT, FS_ANN, FS_LAB, INK, NAVY, CardLayout, endash, refline, save, setup, tint)

DC = DOMAIN_COLOR
TWO_LINE = {"Mechanistic reviews, omics and AI": "Mechanisms,\nomics and AI",
            "Obesity, diabetes and cardiometabolic medicine": "Cardiometabolic\nmedicine",
            "Non-invasive diagnosis and imaging": "Non-invasive\ndiagnosis",
            "Clinical trials and interventions": "Trials and\ninterventions",
            "Advanced liver disease and liver cancer": "Advanced liver\ndisease and HCC",
            "Case-based and differential hepatology": "Case-based\nhepatology",
            "Clinical epidemiology and comorbidity": "Clinical\nepidemiology",
            "Experimental models and preclinical therapeutics": "Experimental\nmodels",
            "Gut microbiome": "Gut\nmicrobiome", "Veterinary and livestock fatty liver": "Veterinary"}
TAG = DOMAIN_SHORT


def pipeline(C, R):
    S, F = R["structure"], R["corpus"]["flow"]
    L = C.L
    steps = [(f"{F['included_literature']:,} papers", "titles and abstracts,\n2020–2026"),
             ("Embeddings", "PubMedBERT-based;\nnames neutralized"),
             ("PCA", "50 components"),
             ("UMAP", "10 dimensions;\ncosine, 30 neighbours"),
             ("HDBSCAN", f"min. cluster size 150;\n{100 * S['noise_fraction_literature']:.0f}% to nearest topic"),
             (f"{S['n_research_topics']} topics", f"{S['research_papers']:,} papers;\n{S['n_meta_topics']} meta-topics excluded"),
             (f"{S['n_domains']} domains", "Ward clustering of\ntopic centroids (b)")]
    n = len(steps)
    x0, x1, y, h = 4.0, 176.0, 15.0 - C.TRIM, 16.0
    gap = 4.6
    w = (x1 - x0 - gap * (n - 1)) / n
    for i, (t, sub) in enumerate(steps):
        x = x0 + i * (w + gap)
        last = i == n - 1
        fc = NAVY if last else tint(NAVY, 0.88 - 0.1 * i)
        L.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec="none", zorder=3))
        col = "white" if last else INK
        L.text(x + w / 2, y + 3.6, t, ha="center", va="center", fontsize=FS_LAB, fontweight="bold", color=col, zorder=4)
        L.text(x + w / 2, y + 9.8, sub, ha="center", va="center", fontsize=FS_ANN, color=col, zorder=4, linespacing=1.15)
        if i < n - 1:
            L.add_patch(FancyArrowPatch((x + w + 0.5, y + h / 2), (x + w + gap - 0.5, y + h / 2), arrowstyle="-|>",
                                        mutation_scale=9, color=NAVY, lw=1.2, zorder=4))


def _on_colour(hex_color):
    def lum(h):
        c = [int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    lb, li = lum(hex_color), lum(INK)
    return "white" if 1.05 / (lb + 0.05) > (lb + 0.05) / (li + 0.05) else INK


def tree(C, R, y0):
    Z = np.load(TOPICS_ANA / f"{REF_RUN}_linkage_ward.npy")
    cat = pd.read_csv(TOPICS_ANA / "topic_catalogue_named.csv")
    r = cat[~cat.meta_topic].reset_index(drop=True)
    dom = r.domain_name.values
    n = len(r)
    assert Z.shape[0] == n - 1, "linkage and topics disagree"
    members = {i: {dom[i]} for i in range(n)}
    for k, (a, b, _, _) in enumerate(Z):
        members[n + k] = members[int(a)] | members[int(b)]
    ax = C.ax(8, y0, 166, 19)
    dn = dendrogram(Z, ax=ax, no_labels=True,
                    link_color_func=lambda k: DC[next(iter(members[k]))] if len(members[k]) == 1 else INK)
    for coll in ax.collections:
        coll.set_linewidth(0.7)
    ax.set_xlim(0, 10 * n); ax.set_xticks([])
    ax.set_yticks([]); ax.set_ylim(0, Z[:, 2].max() * 1.04)
    for sp in ax.spines.values():
        sp.set_visible(False)
    order = dn["leaves"]
    leaves = dom[order]
    spans, start = [], 0
    for i in range(1, n + 1):
        if i == n or leaves[i] != leaves[start]:
            spans.append((leaves[start], start, i)); start = i
    assert len(spans) == len({s[0] for s in spans}), "a domain is split in the dendrogram"
    mm = lambda i: 8 + 166 * i / n
    L = C.L
    y0 = y0 - C.TRIM
    for k, (d, a, b) in enumerate(spans, 1):
        L.add_patch(Rectangle((mm(a) + 0.15, y0 + 19.4), mm(b) - mm(a) - 0.3, 2.6, fc=DC[d], ec="none", zorder=3))
        L.text((mm(a) + mm(b)) / 2, y0 + 20.7, str(k), ha="center", va="center", fontsize=FS_ANN - 0.5, fontweight="bold",
               color=_on_colour(DC[d]), zorder=4)
        x, y = 4 + 35.0 * ((k - 1) % 5), y0 + 25.0 + 3.6 * ((k - 1) // 5)
        L.add_patch(Rectangle((x, y - 1.3), 3.4, 2.6, fc=DC[d], ec="none", zorder=3))
        L.text(x + 1.7, y, str(k), ha="center", va="center", fontsize=FS_ANN - 0.5, fontweight="bold", color=_on_colour(DC[d]),
               zorder=4)
        L.text(x + 4.4, y, TAG[d], ha="left", va="center", fontsize=FS_ANN, color=INK, zorder=4)
    pd.DataFrame({"leaf_order": range(n), "topic": r.topic.values[order], "topic_label": r.topic_label.values[order],
                  "domain": leaves}).to_csv(SRC / "ed3_dendrogram_leaves.csv", index=False)
    pd.DataFrame(Z, columns=["child_a", "child_b", "ward_distance", "n_topics"]).to_csv(SRC / "ed3_ward_linkage.csv", index=False)


def main():
    setup()
    R = load_results()
    S = R["structure"]
    C = CardLayout(180, 229)
    fig = C.fig
    C.card(0, 4, 180, 36, "From papers to research topics and domains", "a")
    C.card(0, 45, 180, 43, f"{S['n_research_topics']} topics merge into {S['n_domains']} domains (Ward clustering of topic centroids)",
           "b")
    C.card(0, 93, 108, 62, "Research domains", "c")
    C.card(112, 93, 68, 62, "Domain size", "d")
    C.card(0, 160, 108, 69, "Topics with significant share trends, 2020–2025", "e")
    C.card(112, 160, 68, 69, "Domain shifts, 2020 to 2025", "f")
    pipeline(C, R)
    tree(C, R, 53.0)
    pts = pd.read_csv(SRC / "fig3a_map_points.csv")
    order = pts.domain.value_counts().index.tolist()
    bg = pts.sample(min(len(pts), 30000), random_state=7)
    for k, d in enumerate(order):
        r, c = divmod(k, 5)
        ax = C.ax(3.0 + c * 20.8, 105.5 + r * 25.0, 19.6, 19.5)
        ax.scatter(bg.x, bg.y, s=0.15, color="#E3E8F2", lw=0, rasterized=True)
        g = pts[pts.domain == d]
        ax.scatter(g.x, g.y, s=0.15, color=DC[d], lw=0, alpha=0.9, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(TWO_LINE.get(d, DOMAIN_SHORT[d]), fontsize=FS_ANN, loc="center", pad=1.0, linespacing=1.05, color=INK)
    ax = C.ax(146, 101.5, 22, 42)
    ds = pd.Series(S["domain_sizes"]).sort_values()
    core = pd.Series(S["domain_core_sizes"]).reindex(ds.index).fillna(0)
    y = np.arange(len(ds))
    ax.barh(y, core.values / 1000, height=0.66, color=[DC[d] for d in ds.index])
    ax.barh(y, (ds - core).values / 1000, left=core.values / 1000, height=0.66, color=[tint(DC[d], 0.55) for d in ds.index])
    for yi, v in zip(y, ds.values):
        ax.text(v / 1000 + 0.4, yi, f"{v:,}", va="center", fontsize=FS_ANN, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([DOMAIN_SHORT[d] for d in ds.index], fontsize=FS_ANN)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, ds.max() / 1000 * 1.42)
    ax.set_xticks([0, 10, 20])
    ax.set_xlabel("Papers (thousands)")
    d0 = ds.index[-1]
    ax.legend([Patch(color=DC[d0]), Patch(color=tint(DC[d0], 0.55))], ["Cluster core", "Assigned to nearest topic"],
              loc="upper left", bbox_to_anchor=(-1.2, -0.17), ncol=2, fontsize=FS_ANN, handlelength=1.0, frameon=False,
              columnspacing=1.0)
    dyn = pd.read_csv(SRC / "topic_dynamics.csv")
    em = dyn[dyn.trend == "emerging"].sort_values("rgr", ascending=False).head(10)
    de = dyn[dyn.trend == "declining"].sort_values("rgr").head(10)
    sel = pd.concat([em, de.iloc[::-1]])
    ax = C.ax(44, 168.0, 42, 49)
    yy = np.arange(len(sel))[::-1]
    ax.axhline(9.5, color=INK, lw=0.4, ls=(0, (1, 1.5)))
    for yi, r in zip(yy, sel.itertuples()):
        col = DC[r.domain_name]
        ax.plot([r.rgr_lo, r.rgr_hi], [yi, yi], color=col, lw=1.0, solid_capstyle="round")
        ax.scatter([r.rgr], [yi], s=12, color=col, edgecolor="white", lw=0.4, zorder=3)
    refline(ax, 1, "x")
    ax.set_xscale("log")
    ax.set_xticks([0.7, 1.0, 1.5]); ax.set_xticklabels(["0.7", "1.0", "1.5"])
    ax.set_xlim(0.6, 1.85)
    ax.minorticks_off()
    ax.set_yticks(yy); ax.set_yticklabels([endash(t) for t in sel.topic_label.values], fontsize=FS_ANN)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.8, len(sel) - 0.2)
    ax.text(0.02, 0.985, f"10 fastest of {S['dynamics']['emerging']}\ngrowing topics", transform=ax.transAxes, fontsize=FS_ANN,
            color=INK, ha="left", va="top")
    ax.text(0.98, 0.015, f"10 fastest of {S['dynamics']['declining']}\ndeclining topics", transform=ax.transAxes,
            fontsize=FS_ANN, color=INK, ha="right", va="bottom")
    ax.set_xlabel("Relative change in topic share\nper year (95% CI; log scale)")
    tr = ax.get_yaxis_transform()
    for yi, r in zip(yy, sel.itertuples()):
        ax.scatter([1.04], [yi], s=6, marker="s", color=DC[r.domain_name], lw=0, transform=tr, clip_on=False)
        ax.text(1.075, yi, TAG[r.domain_name], transform=tr, fontsize=5.0, color=INK, va="center", ha="left", clip_on=False)
    ax = C.ax(146, 168.0, 30, 49)
    ch = (pd.Series(S["domain_share_2025"]) - pd.Series(S["domain_share_2020"])) * 100
    ch = ch.sort_values()
    yd = np.arange(len(ch))
    ax.barh(yd, ch.values, height=0.66, color=[DC[d] for d in ch.index])
    for yi, v in zip(yd, ch.values):
        lab = "0.0" if abs(v) < 0.05 else f"{v:+.1f}".replace("-", "−")
        ax.text(v + (0.3 if v >= 0 else -0.3), yi, lab, va="center", ha="left" if v >= 0 else "right", fontsize=FS_ANN, color=INK)
    ax.axvline(0, color=INK, lw=0.5)
    ax.set_yticks(yd); ax.set_yticklabels([DOMAIN_SHORT[d] for d in ch.index], fontsize=FS_ANN)
    ax.tick_params(axis="y", length=0)
    lim = np.abs(ch.values).max() * 1.5
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Change in share of papers\n(percentage points)")
    sel.assign(group=["growing"] * len(em) + ["declining"] * len(de))[
        ["group", "topic", "topic_label", "domain_name", "rgr", "rgr_lo", "rgr_hi", "q", "n_total"]].to_csv(
        SRC / "ed3_topics_shown.csv", index=False)
    pd.DataFrame({"domain": list(S["domain_sizes"]), "papers": list(S["domain_sizes"].values()),
                  "cluster_core": [S["domain_core_sizes"].get(d, 0) for d in S["domain_sizes"]],
                  "share_2020": [S["domain_share_2020"][d] for d in S["domain_sizes"]],
                  "share_2025": [S["domain_share_2025"][d] for d in S["domain_sizes"]]}).to_csv(SRC / "ed3_domains.csv", index=False)
    save(fig, "Figure2")


if __name__ == "__main__":
    main()
