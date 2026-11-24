from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from common import (AXIS, BLUE_RAMP, CAT, DIV_BLUE, DIV_MID, DIV_RED, FIG, GRID, INK, INK2, MUTED, SURFACE)

MM = 1 / 25.4
FULL = 180 * MM
HALF = 88 * MM

FS_ANN, FS_TICK, FS_LAB, FS_LETTER = 5.5, 6.0, 6.5, 8.0

NAVY, RED, TEAL, GOLD, SKY, PURPLE = CAT
BROWN = "#8C510A"
LIT = "#1A1A1A"
LIGHT = "#DCE2EE"
BAND = "#EDF0F6"
STREAM_COLOR = {"literature": LIT, "trial": RED, "nih_grant": TEAL, "news": GOLD}
MAIN = NAVY
SENS1, SENS2 = PURPLE, "#F28E2B"
PUBLIC = NAVY
ORIG, NEUT = "#C0507A", "#2E86AB"
NAMES = {"legacy": "#9C5B2E", "interim": PURPLE, "current": NAVY}
DOMAIN_COLOR = {
    "Clinical epidemiology and comorbidity": "#4E79A7",
    "Experimental models and preclinical therapeutics": "#9C755F",
    "Mechanistic reviews, omics and AI": "#B07AA1",
    "Obesity, diabetes and cardiometabolic medicine": "#F28E2B",
    "Non-invasive diagnosis and imaging": "#76B7B2",
    "Clinical trials and interventions": "#E15759",
    "Advanced liver disease and liver cancer": "#2E7D32",
    "Case-based and differential hepatology": "#FF9DA7",
    "Gut microbiome": "#EDC948",
    "Veterinary and livestock fatty liver": "#8F8F32",
}


def tint(hex_color, f):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(round(c + (255 - c) * f)) for c in (r, g, b))

DOMAIN_SHORT = {
    "Clinical epidemiology and comorbidity": "Clinical epidemiology",
    "Experimental models and preclinical therapeutics": "Experimental models",
    "Mechanistic reviews, omics and AI": "Mechanisms, omics and AI",
    "Obesity, diabetes and cardiometabolic medicine": "Cardiometabolic medicine",
    "Non-invasive diagnosis and imaging": "Non-invasive diagnosis",
    "Clinical trials and interventions": "Trials and interventions",
    "Advanced liver disease and liver cancer": "Advanced liver disease and HCC",
    "Case-based and differential hepatology": "Case-based hepatology",
    "Gut microbiome": "Gut microbiome",
    "Veterinary and livestock fatty liver": "Veterinary",
}
STREAM_LABEL = {"literature": "Literature", "trial": "Trial registrations", "nih_grant": "NIH projects", "news": "News"}

SEQ = LinearSegmentedColormap.from_list("navy_seq", [BLUE_RAMP[k] for k in (100, 250, 400, 550, 700)])
DIV = LinearSegmentedColormap.from_list("navy_brown", [NAVY, "#A9B6D3", DIV_MID, "#D9BE8F", BROWN])


def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": FS_TICK,
        "axes.titlesize": FS_LAB, "axes.labelsize": FS_LAB, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_ANN, "legend.title_fontsize": FS_ANN,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.5, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.titleweight": "regular", "axes.titlelocation": "left", "axes.titlepad": 4, "axes.labelpad": 2.5,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "xtick.major.pad": 1.8, "ytick.major.pad": 1.8, "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": GRID, "grid.linewidth": 0.35, "grid.linestyle": "-",
        "lines.linewidth": 1.0, "lines.markersize": 3.0, "lines.markeredgewidth": 0,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "patch.linewidth": 0, "errorbar.capsize": 0, "hatch.linewidth": 0.4,
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.5, "legend.labelspacing": 0.35,
        "legend.borderaxespad": 0.2, "legend.columnspacing": 1.0, "legend.borderpad": 0.2,
        "axes.prop_cycle": mpl.cycler(color=CAT),
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "text.color": INK, "axes.unicode_minus": True,
    })


def panel(ax, letter, x=None, y=None, fig=None):
    fig = fig or ax.figure
    fig.__dict__.setdefault("_panels", []).append((ax, letter, x, y))


def panel_at(fig, letter, x, y):
    fig.__dict__.setdefault("_panels", []).append((None, letter, x, y))


def _place_letters(fig):
    items = fig.__dict__.get("_panels", [])
    if not items:
        return
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    gap = 3.4 / 180
    pos = []
    for ax, letter, x, y in items:
        if ax is None or (x is not None and y is not None):
            pos.append([x, y, letter, True])
            continue
        bb = inv.transform_bbox(ax.get_tightbbox(r))
        x0, y1 = bb.x0, bb.y1
        t = ax.title if ax.title.get_text() else (ax._left_title if ax._left_title.get_text() else None)
        if t is not None:
            tb = inv.transform_bbox(t.get_window_extent(r))
            x0 = min(x0, tb.x0 - gap)
            y1 = max(y1, tb.y1)
        pos.append([x if x is not None else x0, y if y is not None else y1, letter, False])
    for key, tol, fn in ((1, 0.035, max), (0, 0.07, min)):
        srt = sorted(pos, key=lambda p: p[key])
        groups, cur = [], []
        for p in srt:
            if cur and abs(p[key] - cur[-1][key]) > tol:
                groups.append(cur); cur = []
            cur.append(p)
        if cur:
            groups.append(cur)
        for g in groups:
            fixed = [p[key] for p in g if p[3]]
            v = fixed[0] if fixed else fn(p[key] for p in g)
            for p in g:
                if not p[3]:
                    p[key] = v
    for x, y, letter, _ in pos:
        fig.text(max(x, 0.004), min(y, 0.997), letter, fontsize=FS_LETTER, fontweight="bold", color=INK,
                 ha="left", va="top")


def hgrid(ax, axis="y"):
    return None


def refgrid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.35)
    ax.set_axisbelow(True)


def refline(ax, v, axis="x", **kw):
    f = ax.axvline if axis == "x" else ax.axhline
    return f(v, color=kw.pop("color", MUTED), lw=kw.pop("lw", 0.5), ls=kw.pop("ls", (0, (2.5, 2))), zorder=kw.pop("zorder", 0.5), **kw)


def direct_labels(ax, x, items, dx_pt=3, min_gap_pt=None, fontsize=FS_ANN, ha="left"):
    if not items:
        return
    fig = ax.figure
    min_gap = (min_gap_pt or fontsize * 1.25) * fig.dpi / 72
    tr = ax.transData
    disp = sorted([(tr.transform((x, y))[1], txt, col) for y, txt, col in items], key=lambda t: t[0])
    ys = [d[0] for d in disp]
    for _ in range(50):
        moved = False
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < min_gap:
                push = (min_gap - (ys[i] - ys[i - 1])) / 2
                ys[i - 1] -= push; ys[i] += push; moved = True
        if not moved:
            break
    xd = tr.transform((x, 0))[0] + dx_pt * fig.dpi / 72
    inv = ax.transData.inverted()
    for yd, (_, txt, col) in zip(ys, disp):
        xx, yy = inv.transform((xd, yd))
        ax.text(xx, yy, txt, color=col, fontsize=fontsize, va="center", ha=ha, clip_on=False)


def place_labels(ax, anchors, obstacles, fontsize=FS_ANN, color=INK2, pad_pt=1.0):
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    k = fig.dpi / 72
    axbb = ax.get_window_extent(r)
    obs = [(*ax.transData.transform((x, y)), rad * k) for x, y, rad in obstacles]
    from matplotlib.legend import Legend
    placed = [a.get_window_extent(r) for a in [ax.get_legend(), *ax.artists, *ax.texts]
              if a is not None and (isinstance(a, Legend) or a.get_visible())]
    near = [(5, 0, "left", "center"), (-5, 0, "right", "center"), (4, 6, "left", "bottom"), (4, -6, "left", "top"),
            (-4, 6, "right", "bottom"), (-4, -6, "right", "top"), (0, 7, "center", "bottom"), (0, -7, "center", "top")]
    far = [(dx * 2.6, dy * 2.6 if dy else 9, ha, va if dy else "bottom") for dx, dy, ha, va in near]
    for item in anchors:
        x, y, txt = item[:3]
        prefs = list(item[3]) if len(item) > 3 else []
        ax_, ay_ = ax.transData.transform((x, y))
        own = [o for o in obs if abs(o[0] - ax_) < 0.5 and abs(o[1] - ay_) < 0.5]
        rown = max([o[2] for o in own], default=0) / k
        others = [o for o in obs if o not in own]
        cands = [(dx + (rown if dx > 0 else -rown if dx < 0 else 0), dy + (rown if dy > 0 else -rown if dy < 0 else 0), ha, va)
                 for dx, dy, ha, va in prefs + near + far]
        best = None
        for i, (dx, dy, ha, va) in enumerate(cands):
            t = ax.annotate(txt, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize, color=color,
                            ha=ha, va=va)
            bb = t.get_window_extent(r).expanded(1.0, 1.0)
            bb = type(bb).from_extents(bb.x0 - pad_pt * k, bb.y0 - pad_pt * k, bb.x1 + pad_pt * k, bb.y1 + pad_pt * k)
            bad = sum(1 for ox, oy, rad in others if bb.x0 - rad < ox < bb.x1 + rad and bb.y0 - rad < oy < bb.y1 + rad)
            bad += 5 * sum(1 for p in placed if bb.overlaps(p))
            bad += 10 * (bb.x0 < axbb.x0 or bb.x1 > axbb.x1 or bb.y0 < axbb.y0 or bb.y1 > axbb.y1)
            t.remove()
            if best is None or bad < best[0]:
                best = (bad, i, dx, dy, ha, va, bb)
            if bad == 0:
                break
        bad, i, dx, dy, ha, va, bb = best
        far_pos = (dx * dx + dy * dy) > (14 + rown) ** 2
        ax.annotate(txt, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize, color=color, ha=ha, va=va,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=MUTED, shrinkA=0.5, shrinkB=1.5) if far_pos else None)
        placed.append(bb)


def wilson_ci(k, n, z=1.96):
    import numpy as _np
    k, n = _np.asarray(k, float), _np.asarray(n, float)
    p = _np.where(n > 0, k / _np.maximum(n, 1), 0)
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * _np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def endash(s):
    import re as _re
    s = _re.sub(r"(\d)-(\d)", "\\1–\\2", str(s))
    for a, b in (("Gut-liver", "Gut–liver"), ("Jensen-Shannon", "Jensen–Shannon"), ("glucose-index", "glucose index")):
        s = s.replace(a, b)
    return s


def trim(fig, cuts):
    W, H = fig.get_size_inches()
    Hm = H * 25.4
    axes = {ax: [*ax.get_position().bounds] for ax in fig.axes}
    for v in axes.values():
        v[1] *= Hm; v[3] *= Hm
    texts = {t: list(t.get_position()) for t in fig.texts if t.get_transform() == fig.transFigure}
    for v in texts.values():
        v[1] *= Hm
    panels = [list(p) for p in fig.__dict__.get("_panels", [])]
    for p in panels:
        if p[3] is not None:
            p[3] *= Hm
    for y_mm, d in sorted(cuts, key=lambda c: -c[0]):
        for v in axes.values():
            if v[1] >= y_mm - 1e-6:
                v[1] -= d
        for v in texts.values():
            if v[1] >= y_mm - 1e-6:
                v[1] -= d
        for p in panels:
            if p[3] is not None and p[3] >= y_mm - 1e-6:
                p[3] -= d
        Hm -= d
    fig.set_size_inches(W, Hm / 25.4)
    for ax, v in axes.items():
        ax.set_position([v[0], v[1] / Hm, v[2], v[3] / Hm])
    for t, v in texts.items():
        t.set_position((v[0], v[1] / Hm))
    for p in panels:
        if p[3] is not None:
            p[3] /= Hm
    if "_panels" in fig.__dict__:
        fig.__dict__["_panels"] = [tuple(p) for p in panels]


CARD_BG = "#EEF2F8"


class CardLayout:

    TRIM = 4.5

    def __init__(self, w_mm, h_mm):
        self.W, self.H = w_mm, h_mm - self.TRIM
        self.fig = plt.figure(figsize=(w_mm * MM, self.H * MM))
        L = self.fig.add_axes([0, 0, 1, 1])
        L.set_xlim(0, w_mm); L.set_ylim(self.H, 0); L.axis("off"); L.set_zorder(-5)
        self.L = L

    def card(self, x, y, w, h, title=None, letter=None, color=None, band=5.2):
        if letter:
            self.L.text(x + 0.2, y - self.TRIM + 1.0, letter, fontsize=FS_LETTER, fontweight="bold", va="top", ha="left",
                        color=INK)

    def ax(self, x, y, w, h, **kw):
        y = y - self.TRIM
        a = self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H], **kw)
        a.set_facecolor("none")
        return a

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", FS_ANN); kw.setdefault("color", INK)
        return self.L.text(x, y - self.TRIM, s, **kw)


def save(fig, name):
    _place_letters(fig)
    for ext, kw in [("pdf", {"dpi": 600}), ("svg", {"dpi": 600}), ("png", {"dpi": 600})]:
        fig.savefig(FIG / f"{name}.{ext}", **kw)
    plt.close(fig)


__all__ = ["CAT", "INK", "INK2", "MUTED", "GRID", "AXIS", "BLUE_RAMP", "DIV_BLUE", "DIV_RED", "DIV_MID", "SEQ", "DIV", "ORIG", "NEUT", "NAMES", "DOMAIN_COLOR", "tint",
           "FULL", "HALF", "MM", "DOMAIN_SHORT", "STREAM_LABEL", "STREAM_COLOR", "NAVY", "RED", "TEAL", "GOLD", "SKY",
           "PURPLE", "BROWN", "LIT", "LIGHT", "BAND", "MAIN", "SENS1", "SENS2", "PUBLIC", "wilson_ci", "endash", "FS_ANN", "FS_TICK", "FS_LAB", "FS_LETTER",
           "setup", "panel", "panel_at", "trim", "hgrid", "refgrid", "refline", "direct_labels", "place_labels", "save", "CardLayout",
           "CARD_BG"]
