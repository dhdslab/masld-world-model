from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram

from common import TOPICS_ANA, REF_RUN, SRC, load_results
from figstyle import (DOMAIN_COLOR, DOMAIN_SHORT, FS_ANN, GRID, INK, LIGHT, MUTED, CardLayout, save, setup, tint)

DC = DOMAIN_COLOR
TWO = {"Clinical epidemiology and comorbidity": "Clinical\nepidemiology",
       "Experimental models and preclinical therapeutics": "Experimental\nmodels",
       "Mechanistic reviews, omics and AI": "Mechanisms,\nomics and AI",
       "Obesity, diabetes and cardiometabolic medicine": "Cardiometabolic\nmedicine",
       "Non-invasive diagnosis and imaging": "Non-invasive\ndiagnosis",
       "Clinical trials and interventions": "Trials and\ninterventions",
       "Advanced liver disease and liver cancer": "Advanced liver\ndisease and HCC",
       "Case-based and differential hepatology": "Case-based\nhepatology",
       "Gut microbiome": "Gut\nmicrobiome", "Veterinary and livestock fatty liver": "Veterinary"}
INPUT_LABEL = {"world_model": "World\nmodel", "cos": "Semantic\nsimilarity", "triad": "Neighbourhood\nsimilarity",
               "proj3": "Projected\nproximity", "conv": "Semantic\nconvergence", "drift": "Field-drift\nalignment",
               "n_lo": "Attention,\nlower", "n_hi": "Attention,\nhigher", "g_lo": "Growth,\nlower", "g_hi": "Growth,\nhigher",
               "cn": "Common\nneighbours", "ra": "Resource\nallocation", "aa": "Adamic–\nAdar", "jac": "Jaccard",
               "pa": "Degree\nproduct"}
AXES_C = list(INPUT_LABEL)
METRIC_LABEL = {"attention": "Attention", "growth": "Growth", "connectivity": "Degree", "speed": "Semantic\nspeed",
                "drift_alignment": "Drift\nalignment", "coherence": "Neighbour\ncoherence",
                "distinctiveness": "Distance from\nfield centre"}
ACRONYM = {"hcc": "HCC", "ast": "AST", "alt": "ALT", "ros": "ROS", "apob": "APOB", "ald": "ALD", "nafld": "NAFLD",
           "alanine aminotransferase": "ALT"}
GAP_B = np.deg2rad(20)
B_CENTRE, B_MAX = 0.30, 1.15
WHITE_BOX = dict(fc="white", ec="none", pad=0.3)


def polar(ax):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_axis_off()
    return ax


def ring(ax, r, lw=0.35, color=GRID, ls="-", z=0.5):
    t = np.linspace(0, 2 * np.pi, 361)
    ax.plot(t, np.full_like(t, r), color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")


def align(theta):
    s = np.sin(theta)
    return "left" if s > 0.15 else "right" if s < -0.15 else "center"


def valign(theta):
    c = np.cos(theta)
    return "bottom" if c > 0.5 else "top" if c < -0.5 else "center"


def on_colour(hex_color):
    def lum(h):
        c = [int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    lb, li = lum(hex_color), lum(INK)
    return "white" if 1.05 / (lb + 0.05) > (lb + 0.05) / (li + 0.05) else INK


def number(ax, theta, r, k, color):
    ax.text(theta, r, str(k), fontsize=FS_ANN - 0.5, fontweight="bold", color=on_colour(color), ha="center", va="center",
            zorder=6)


def radial_tree(ax, order):
    Z = np.load(TOPICS_ANA / f"{REF_RUN}_linkage_ward.npy")
    rings = pd.read_csv(SRC / "atlas_topic_rings.csv").sort_values("topic")
    cat = pd.read_csv(TOPICS_ANA / "topic_catalogue_named.csv")
    r = cat[~cat.meta_topic].reset_index(drop=True)
    assert (r.topic.values == rings.topic.values).all()
    dom = r.domain_name.values
    n = len(r)
    members = {i: {dom[i]} for i in range(n)}
    for k, (a, b, _, _) in enumerate(Z):
        members[n + k] = members[int(a)] | members[int(b)]
    dn = dendrogram(Z, no_plot=True, link_color_func=lambda k: DC[next(iter(members[k]))] if len(members[k]) == 1 else INK)
    hmax = Z[:, 2].max()
    th = lambda x: 2 * np.pi * np.asarray(x) / (10 * n)
    rad = lambda h: 1.0 - np.asarray(h) / hmax * 0.88
    for xs, ys, col in zip(dn["icoord"], dn["dcoord"], dn["color_list"]):
        kw = dict(color=col, lw=0.6 if col != INK else 0.5, solid_capstyle="butt", zorder=2)
        ax.plot([th(xs[0])] * 2, [rad(ys[0]), rad(ys[1])], **kw)
        ax.plot([th(xs[3])] * 2, [rad(ys[3]), rad(ys[2])], **kw)
        tt = np.linspace(th(xs[1]), th(xs[2]), 40)
        ax.plot(tt, np.full_like(tt, rad(ys[1])), **kw)
    leaves = dn["leaves"]
    w = 2 * np.pi / n
    tl = th(5 + 10 * np.arange(n))
    rg = rings.set_index("topic").loc[r.topic.values[leaves]]
    size = np.sqrt(rg.n_assigned.values / rg.n_assigned.max())
    lv = dom[leaves]
    cols = [DC[d] for d in lv]
    ax.bar(tl, 0.10, width=w * 0.96, bottom=1.02, color=cols, lw=0, zorder=3)
    ax.bar(tl, 0.30 * size, width=w * 0.70, bottom=1.15, color=[tint(c, 0.35) for c in cols], lw=0, zorder=3)
    for t, tr, sz in zip(tl, rg.trend.values, size):
        if tr in ("emerging", "declining"):
            ax.plot([t], [1.15 + 0.30 * sz + 0.07], marker="^" if tr == "emerging" else "v", ms=2.4, color=INK, mew=0,
                    zorder=4)
    for k, d in enumerate(order, 1):
        idx = np.flatnonzero(lv == d)
        number(ax, tl[idx].mean(), 1.07, k, DC[d])
    ax.set_ylim(0, 1.53)
    big = rg.n_assigned.values.argmax()
    return {"largest_topic": str(rg.topic_label.values[big]), "largest_n": int(rg.n_assigned.values[big]),
            "emerging": int((rg.trend == "emerging").sum()), "declining": int((rg.trend == "declining").sum())}


def entity_wheel(ax, order):
    E = pd.read_csv(SRC / "atlas_entity_states.csv", dtype={"name": str})
    E = E[E.domain.notna()].copy()
    rmap = lambda v: 1.22 * (np.asarray(v) - B_CENTRE) / (B_MAX - B_CENTRE)
    assert E.raw_distinctiveness.between(B_CENTRE, B_MAX).all()
    att = (E.raw_attention - E.raw_attention.min()) / (E.raw_attention.max() - E.raw_attention.min())
    E["s"] = 0.25 + 9.0 * att ** 1.6
    E["theta"] = GAP_B / 2 + E.theta * (2 * np.pi - GAP_B) / (2 * np.pi)
    E = E.sort_values("s", ascending=False)
    for v in (0.6, 0.8, 1.0):
        tt = np.linspace(GAP_B / 2 - np.deg2rad(4), 2 * np.pi - GAP_B / 2 + np.deg2rad(4), 361)
        ax.plot(tt, np.full_like(tt, rmap(v)), color=GRID, lw=0.35, zorder=0.5)
    ax.scatter(E.theta, rmap(E.raw_distinctiveness), s=E.s, c=[DC[d] for d in E.domain], lw=0, alpha=0.8, zorder=3,
               rasterized=True)
    for d in order:
        t = E.theta[E.domain == d]
        tt = np.linspace(t.min(), t.max(), 60)
        ax.fill_between(tt, 1.28, 1.38, color=DC[d], lw=0, zorder=3)
    for v in (0.6, 0.8, 1.0):
        ax.text(0, rmap(v), f"{v:.1f}", fontsize=FS_ANN - 0.5, color=INK, ha="center", va="center", zorder=5, bbox=WHITE_BOX)
    top = E.sort_values("n_papers", ascending=False).groupby("domain").head(1)
    ax.scatter(top.theta, rmap(top.raw_distinctiveness), s=top.s, c=[DC[d] for d in top.domain], lw=0.4, edgecolors=INK,
               zorder=5)
    lab = sorted([[e.theta, float(rmap(e.raw_distinctiveness)), ACRONYM.get(str(e["name"]).lower(), str(e["name"]))]
                  for _, e in top.iterrows()], key=lambda x: x[0])
    tl = [x[0] for x in lab]
    gap = np.deg2rad(13)
    for _ in range(200):
        moved = False
        for i in range(len(tl)):
            j = (i + 1) % len(tl)
            d = (tl[j] - tl[i]) % (2 * np.pi)
            if d < gap:
                tl[i] -= (gap - d) / 2; tl[j] += (gap - d) / 2; moved = True
        if not moved:
            break
    cross = []
    for (t, rr, name), t2 in zip(lab, tl):
        ax.plot([t, t2], [rr, 1.43], color=MUTED, lw=0.35, zorder=4)
        ax.text(t2, 1.47, name, fontsize=FS_ANN, color=INK, ha=align(t2), va=valign(t2), zorder=5)
        cross.append(t + (t2 - t) * (1.33 - rr) / (1.43 - rr))
    clear = np.deg2rad(8.0)
    for k, d in enumerate(order, 1):
        t = E.theta[E.domain == d]
        mid = 0.5 * (t.min() + t.max())
        cand = np.linspace(t.min() + np.deg2rad(3), t.max() - np.deg2rad(3), 61)
        if t.max() - t.min() < np.deg2rad(6):
            cand = np.array([mid])
        gap_ = np.array([min([abs(c - x) for x in cross] + [9.0]) for c in cand])
        ok = gap_ >= clear
        pos = cand[ok][np.argmin(np.abs(cand[ok] - mid))] if ok.any() else cand[np.argmax(gap_)]
        number(ax, pos, 1.33, k, DC[d])
    ax.set_ylim(0, 1.40)
    return E


def auroc_radar(ax, order):
    A = pd.read_csv(SRC / "atlas_pair_input_auroc.csv")
    k = len(AXES_C)
    th = 2 * np.pi * np.arange(k) / k
    lo = 0.4
    rr = lambda a: (np.asarray(a) - lo) / (1 - lo)
    for v in (0.6, 0.7, 0.8, 0.9, 1.0):
        ring(ax, rr(v))
    ring(ax, rr(0.5), color=MUTED, lw=0.5, ls=(0, (2.5, 2)))
    for t in th:
        ax.plot([t, t], [0, 1], color=GRID, lw=0.35, zorder=0.5)
    for v, after in ((0.7, "g_lo"), (0.9, "g_lo"), (0.5, "conv")):
        ax.text(2 * np.pi * (AXES_C.index(after) + 0.7) / k, rr(v), f"{v:.1f}", fontsize=FS_ANN - 0.5, color=INK,
                ha="center", va="center", zorder=6, bbox=WHITE_BOX)
    close = lambda v: np.r_[v, v[:1]]
    allp = A[A.group == "all"].set_index("input").auroc.reindex(AXES_C)
    D = A[A.group.isin(order)].pivot(index="input", columns="group", values="auroc").reindex(AXES_C)
    lo_, hi_ = D.min(1).values, D.max(1).values
    ax.fill(np.r_[close(th), close(th)[::-1]], np.r_[close(rr(lo_)), close(rr(hi_))[::-1]], color=LIGHT, lw=0, zorder=2)
    dev = D.sub(allp, axis=0).abs().max()
    out = dev.idxmax()
    ax.plot(close(th), close(rr(D[out].values)), color=DC[out], lw=0.8, zorder=3)
    ax.plot(close(th), close(rr(allp.values)), color=INK, lw=1.2, zorder=4)
    ax.plot(th, rr(allp.values), "o", ms=1.8, color=INK, zorder=5)
    for t, c in zip(th, AXES_C):
        if np.cos(t) < -0.75:
            ax.text(t, 1.09, INPUT_LABEL[c].replace("\n", " "), fontsize=FS_ANN, color=INK, va="top",
                    ha="left" if np.sin(t) > 0 else "right")
            continue
        ax.text(t, 1.09, INPUT_LABEL[c], fontsize=FS_ANN, color=INK, ha=align(t), va=valign(t), linespacing=1.0,
                fontweight="bold" if c == "world_model" else "normal")
    ax.set_ylim(0, 1.0)
    return out


def profile_radar(ax, vals, color, key=False):
    k = len(METRIC_LABEL)
    th = 2 * np.pi * np.arange(k) / k
    rr = lambda z: (np.clip(np.asarray(z, float), -1, 1) + 1) / 2
    for v in (-0.5, 0.5, 1.0):
        ring(ax, rr(v))
    ring(ax, rr(0), color=MUTED, lw=0.5, ls=(0, (2.5, 2)))
    for t in th:
        ax.plot([t, t], [0, 1], color=GRID, lw=0.35, zorder=0.5)
    if vals is not None:
        close = lambda v: np.r_[v, v[:1]]
        ax.fill(close(th), close(rr(vals)), color=color, alpha=0.30, lw=0, zorder=2)
        ax.plot(close(th), close(rr(vals)), color=color, lw=0.9, zorder=3)
    if key:
        for t, lab in zip(th, METRIC_LABEL.values()):
            ax.text(t, 1.20, lab, fontsize=FS_ANN, color=INK, ha=align(t), va=valign(t), linespacing=1.0)
        tk = np.deg2rad(360 / k / 2)
        for v, s in ((1, "+1 s.d."), (0, "0")):
            ax.text(tk, rr(v), s, fontsize=FS_ANN - 0.5, color=INK, ha="center", va="center", zorder=6, bbox=WHITE_BOX)
        ax.text(0, 0, "−1", fontsize=FS_ANN - 0.5, color=INK, ha="center", va="center", zorder=6, bbox=WHITE_BOX)
    ax.set_ylim(0, 1.0)


def main():
    setup()
    R = load_results()["semantic_atlas"]
    order = R["domains_order"]
    C = CardLayout(180, 162)
    L = C.L
    C.card(0, 4, 58, 64, letter="a")
    a_info = radial_tree(polar(C.ax(2, 7.5, 54, 54, projection="polar")), order)
    yk = 64.5 - C.TRIM
    L.plot([4.0], [yk], marker="^", ms=2.4, color=INK, mew=0)
    L.text(5.4, yk, "growing topic", fontsize=FS_ANN, va="center")
    L.plot([19.5], [yk], marker="v", ms=2.4, color=INK, mew=0)
    L.text(20.9, yk, "declining topic", fontsize=FS_ANN, va="center")
    L.text(36.5, yk, "bar length, topic size", fontsize=FS_ANN, va="center")
    C.card(61, 4, 58, 64, letter="b")
    entity_wheel(polar(C.ax(66.5, 13, 46, 46, projection="polar")), order)
    C.card(122, 4, 58, 64, letter="c")
    out = auroc_radar(polar(C.ax(135.5, 15, 30, 30, projection="polar")), order)
    yl = 59.5 - C.TRIM
    L.plot([127, 133], [yl, yl], color=INK, lw=1.2)
    L.text(134.5, yl, f"All candidate pairs ({R['positives_all']:,} new associations)", fontsize=FS_ANN, va="center")
    L.add_patch(Rectangle((127, yl + 3.2 - 1.0), 6, 2.0, fc=LIGHT, ec="none"))
    L.text(134.5, yl + 3.2, "Range over the pairs involving each domain", fontsize=FS_ANN, va="center")
    L.plot([127, 133], [yl + 6.4] * 2, color=DC[out], lw=0.8)
    L.text(134.5, yl + 6.4, f"Pairs involving {DOMAIN_SHORT[out].lower()}", fontsize=FS_ANN, va="center")
    y0 = 72.5 - C.TRIM
    for i, d in enumerate(order):
        x = 4 + 35.5 * (i % 5)
        y = y0 + 4.0 * (i // 5)
        L.add_patch(Rectangle((x, y - 1.3), 3.4, 2.6, fc=DC[d], ec="none"))
        L.text(x + 1.7, y, str(i + 1), fontsize=FS_ANN - 0.5, fontweight="bold", color=on_colour(DC[d]), ha="center",
               va="center")
        L.text(x + 4.4, y, DOMAIN_SHORT[d], fontsize=FS_ANN, va="center", color=INK)
    C.card(0, 83, 180, 79, letter="d")
    profile_radar(polar(C.ax(16.5, 109, 21, 21, projection="polar")), None, None, key=True)
    prof = pd.read_csv(SRC / "atlas_domain_profiles.csv").set_index("domain")
    for i, d in enumerate(order):
        cx = 44 + 28 * (i % 5) + 12
        cy = 89 + 37 * (i // 5) + 10.5
        ax = polar(C.ax(cx - 10.5, cy - 10.5, 21, 21, projection="polar"))
        profile_radar(ax, prof.loc[d, list(METRIC_LABEL)].values, DC[d])
        C.text(cx, cy + 13.0, f"{i + 1}. {TWO[d]}", ha="center", va="top", linespacing=1.05)
        C.text(cx, cy + 17.9, f"{int(prof.loc[d, 'entities']):,} entities",
               ha="center", va="top", color=INK)
    save(C.fig, "Figure3")
    pd.DataFrame([{**a_info, "highlighted_domain_c": out}]).to_csv(SRC / "fig3_atlas_ring_summary.csv", index=False)


if __name__ == "__main__":
    main()
