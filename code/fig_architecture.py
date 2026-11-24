from __future__ import annotations

import math

import numpy as np

import w3_world_model as W
import w6_foundation_world_model as F6
from common import load_results
from fig1_design import GREEN, Icons, MUTE, export
from figstyle import INK, NAVY, RED, tint

W_MM = 180
TEAL_G = "#00A087"
ORANGE_A = "#E07B39"
BROWN_T = "#8C510A"
PURPLE_E = "#8E6BB8"
BLOCK_COL = {"semantic": "#3C5488", "dynamics": "#6F86B8", "drift": "#9C6FB0", "graph": TEAL_G, "attention": ORANGE_A}
BLOCKS = [("semantic", 2), ("dynamics", 2), ("drift", 1), ("graph", 5), ("attention", 4)]
KEY_NAME = {"semantic": "semantic state", "dynamics": "semantic dynamics", "drift": "field drift", "graph": "graph topology",
            "attention": "attention and growth"}
PALE = "#D5DAE5"


def bar_h(dim):
    return 1.2 + 0.58 * math.sqrt(dim)


def bar(d, x, yc, dim, color, label=None, w=2.6, label_y=None, sub=None, stroke="none"):
    h = bar_h(dim)
    d.rect(x, yc - h / 2, w, h, fill=color, stroke=stroke, sw=0.5)
    ly = label_y if label_y is not None else yc + h / 2 + 0.6
    if label is not None:
        d.box(x + w / 2 - 6, ly, 12, 3.0, label, fill="none", rounded=False, align="center", size=5.5, pad=0)
    if sub:
        d.box(x + w / 2 - 8, ly + 3.0, 16, 3.0, sub, fill="none", rounded=False, align="center", size=5.5, pad=0)
    return x + w, h


def arrow(d, x0, x1, y, color=NAVY):
    d.path([(x0 + 0.6, y), (x1 - 0.6, y)], color=color, width=0.7, arrow=True)


def text(d, x, y, w, s, align="left", size=5.5, h=3.0):
    d.box(x, y, w, h, s, fill="none", rounded=False, align=align, size=size, pad=0)


def letter(d, x, y, ch):
    d.box(x, y, 6, 4.0, f"<b>{ch}</b>", fill="none", rounded=False, align="left", valign="top", size=8, pad=0)


def graph_glyph(d, gx, gy, kinds=None, scale=1.0, node=2.0):
    nodes = [(-5.0, -3.5), (-1.0, -6.0), (4.0, -4.0), (-3.5, 3.0), (1.5, 1.0), (5.5, 4.5)]
    edges = kinds or [(0, 1, "co"), (1, 2, "co"), (0, 3, "co"), (1, 4, "co"), (3, 4, "co"), (4, 2, "co"), (4, 5, "co")]
    for a, b, k in edges:
        d.path([(gx + scale * nodes[a][0], gy + scale * nodes[a][1]), (gx + scale * nodes[b][0], gy + scale * nodes[b][1])],
               color=TEAL_G if k == "co" else PURPLE_E, width=0.5, dashed=k == "af")
    for nx, ny in nodes:
        d.dot(gx + scale * nx, gy + scale * ny, node, tint(TEAL_G, 0.15), stroke="#FFFFFF", sw=0.3)


class Shifted(Icons):

    def __init__(self):
        super().__init__()
        self.dy = 0.0

    def shape(self, x, y, *a, **k):
        return super().shape(x, y + self.dy, *a, **k)

    def box(self, x, y, *a, **k):
        return super().box(x, y + self.dy, *a, **k)

    def path(self, pts, *a, **k):
        return super().path([(x, y + self.dy) for x, y in pts], *a, **k)


def _n(v, d=0):
    return f"{v:,.{d}f}"


def _dims():
    p = W.OUT / "emb_biomedbert.npy"
    if p.exists():
        d_emb = np.load(p, mmap_mode="r").shape[1]
    else:
        import pandas as pd
        from common import SRC
        rec = pd.read_csv(SRC / "architecture_dimensions.csv").set_index("item")["value"]
        d_emb = int(rec["emb"])
    assert len(W.BASE_X) == sum(k for _, k in BLOCKS)
    return d_emb, W.K_WHITE, W.K_Z, len(W.BASE_X), len(W.TYPEPAIRS)


def world_model(d, R):
    D_EMB, K_WHITE, K_Z, n_x, n_tp = _dims()
    letter(d, 0.2, 0.0, "b")
    letter(d, 0.2, 35.6, "c")
    yc = 19.0
    yn1 = yc + bar_h(D_EMB) / 2 + 0.6
    d.shape(4.0, yc - 6.0, 9.0, 12.0, f"shape=mxgraph.basic.document;html=1;fillColor=#FFFFFF;strokeColor={NAVY};"
                                     f"strokeWidth={0.6 / 0.7045:.2f};")
    for i in range(4):
        d.path([(5.6, yc - 3.0 + 1.9 * i), (10.6 - (1.3 if i == 3 else 0), yc - 3.0 + 1.9 * i)], color=tint(NAVY, 0.4), width=0.5)
    text(d, 1.0, yn1, 16, "title + abstract,<br>names neutralized", align="center", h=6.0)
    arrow(d, 14.0, 17.5, yc)
    for k in range(12):
        d.shape(18.0 + 1.25 * k, yc - 8.0, 3.2, 16.0, f"shape=parallelogram;perimeter=parallelogramPerimeter;size=0.35;html=1;"
                                                    f"fillColor={tint(NAVY, 0.75 - 0.05 * k)};strokeColor=#FFFFFF;"
                                                    f"strokeWidth={0.3 / 0.7045:.2f};")
    text(d, 15.5, yn1, 22, "PubMedBERT,<br>12 layers, frozen", align="center", h=6.0)
    arrow(d, 35.2, 38.4, yc)
    x, _ = bar(d, 38.4, yc, D_EMB, NAVY, f"{D_EMB}", label_y=yn1, sub="e<sub>d</sub>")
    d.shape(x + 1.2, yc - bar_h(D_EMB) / 2, 7.0, bar_h(D_EMB),
            f"shape=trapezoid;perimeter=trapezoidPerimeter;direction=south;size=0.28;html=1;fillColor={tint(NAVY, 0.82)};"
            f"strokeColor=none;")
    text(d, x - 2.2, yc - bar_h(D_EMB) / 2 - 7.0, 16, "whitening<br>(papers ≤ T)", align="center", h=6.2)
    x, _ = bar(d, x + 9.4, yc, K_WHITE, tint(NAVY, 0.6), f"{K_WHITE}", label_y=yn1, sub="ẽ<sub>d</sub>")
    text(d, x + 0.6, yc - 6.6, 13, "Σ κ(T − t<sub>d</sub>)", align="center")
    arrow(d, x, x + 13.5, yc)
    xs = x + 14.0
    bar(d, xs, yc, K_WHITE, NAVY, f"{K_WHITE}", label_y=yn1, sub="h<sub>i</sub>(T)")
    bar(d, xs + 5.4, yc, K_WHITE, BLOCK_COL["dynamics"], f"{K_WHITE}", label_y=yn1, sub="v<sub>i</sub>(T)")
    bar(d, xs + 10.8, yc, K_WHITE, BLOCK_COL["drift"], f"{K_WHITE}", label_y=yn1, sub="u<sub>T</sub>")
    xg = xs + 18.5
    d.rect(xg, yc - 4.5, 2.6, 2.6, fill=ORANGE_A, stroke="none")
    d.rect(xg, yc + 1.9, 2.6, 2.6, fill=ORANGE_A, stroke="none")
    text(d, xg + 3.2, yc - 6.2, 12, "n<sub>i</sub>(T)")
    text(d, xg + 3.2, yc + 0.2, 12, "g<sub>i</sub>(T)")
    gx = xg + 18.0
    graph_glyph(d, gx, yc)
    text(d, gx - 8, yn1 + 3.0, 16, "A<sub>T</sub>", align="center")
    text(d, xs - 2.0, yc - 16.8, gx + 8 - xs + 2, "state S<sub>T</sub>", align="center")
    d.path([(xs, yc - 10.6), (xs, yc - 11.4), (gx + 7.5, yc - 11.4), (gx + 7.5, yc - 10.6)], color=INK, width=0.4)
    for i, (name, cnt) in enumerate(BLOCKS + [("entity-type pair", n_tp)]):
        kx, ky = 120.0 + 31.0 * (i // 3), yc - 7.0 + 4.6 * (i % 3)
        d.rect(kx, ky - 0.9, 1.8, 1.8, fill=BLOCK_COL.get(name, PALE), stroke="none")
        d.box(kx + 2.4, ky - 1.6, 28.0, 3.2, f"{KEY_NAME.get(name, name)} ({cnt})", fill="none", rounded=False, size=5.5, pad=0,
              valign="middle")
    text(d, 120.0, yc - 12.4, 58, "features of x<sub>ij</sub> in c, by block", align="left")
    y2 = 54.0
    yr = 39.5
    d.path([(gx, yn1 + 7.6), (gx, yr), (5.0, yr), (5.0, y2 - 5.0)], color=MUTE, width=0.6, arrow=True)
    d.dot(5.0, y2 - 3.0, 2.6, NAVY, stroke="#FFFFFF")
    d.dot(5.0, y2 + 3.0, 2.6, tint(NAVY, 0.45), stroke="#FFFFFF")
    text(d, 7.2, y2 - 4.5, 6, "i")
    text(d, 7.2, y2 + 1.5, 6, "j")
    arrow(d, 10.0, 15.0, y2)
    cell = 0.6
    top = y2 - (n_x + n_tp) * cell / 2
    k = 0
    for name, cnt in BLOCKS:
        for _ in range(cnt):
            d.rect(15.5, top + k * cell, 3.0, cell - 0.12, fill=BLOCK_COL[name], stroke="none")
            k += 1
    for _ in range(n_tp):
        d.rect(15.5, top + k * cell, 3.0, cell - 0.12, fill=PALE, stroke="none")
        k += 1
    text(d, 9.5, top + (n_x + n_tp) * cell + 0.8, 15, f"x<sub>ij</sub>  {n_x} + {n_tp}", align="center")
    yl = y2 - 9.0
    d.path([(19.0, y2 - 3.0), (26.0, yl), (78.0, yl)], color=NAVY, width=0.6)
    d.box(78.0, yl - 2.6, 13.0, 5.2, "β<sup>T</sup>x<sub>ij</sub>", fill="#FFFFFF", stroke=NAVY, rounded=True, align="center",
          valign="middle", size=6.0, pad=0)
    yn = y2 + 7.0
    yn2 = yn + bar_h(2 * K_Z + n_x + n_tp) / 2 + 0.6
    x0 = 45.0
    d.path([(19.0, y2 + 3.0), (x0 + 7.1, y2 + 3.0), (x0 + 7.1, yn - bar_h(n_x + n_tp) / 2 - 0.4)], color=NAVY, width=0.6,
           arrow=True)
    bar(d, x0, yn, K_Z, NAVY, f"{K_Z}", w=2.2, label_y=yn2)
    bar(d, x0 + 3.0, yn, K_Z, NAVY, f"{K_Z}", w=2.2, label_y=yn2)
    bar(d, x0 + 6.0, yn, n_x + n_tp, "#FFFFFF", f"{n_x + n_tp}", w=2.2, label_y=yn2, stroke=NAVY)
    yzl = yn - bar_h(2 * K_Z + n_x + n_tp) / 2 - 9.1
    text(d, x0 + 0.2, yzl, 22, "z<sub>i</sub> × z<sub>j</sub>, |z<sub>i</sub> − z<sub>j</sub>|,<br>x<sub>ij</sub>", align="left",
         h=6.2)
    d.path([(xs + 1.3, yn1 + 7.6), (xs + 1.3, yr + 2.4), (x0 - 2.0, yr + 2.4), (x0 - 2.0, yn), (x0 - 0.3, yn)],
           color=NAVY, width=0.5, dashed=True, arrow=True, extra="jumpStyle=arc;jumpSize=4;")
    text(d, x0 - 26.0, yl + 1.0, 23, f"z: first {K_Z} axes of h", align="right")
    dims = [2 * K_Z + n_x + n_tp, 64, 64, 1]
    xm = x0 + 11.0
    for i, dm in enumerate(dims):
        if i:
            arrow(d, xm - 2.2, xm, yn)
        xm2, _ = bar(d, xm, yn, dm, "#FFFFFF" if i < 3 else RED, f"{dm}", label_y=yn2,
                     stroke=NAVY if i < 3 else "none")
        xm = xm2 + 3.0
    text(d, x0 + 13.5, yn2 + 3.2, 26, "g<sub>θ</sub>: ReLU, dropout 0.1", align="center")
    sx = 96.0
    d.path([(91.0, yl), (sx - 1.6, y2 - 1.2)], color=NAVY, width=0.6, arrow=True)
    d.path([(xm - 2.4, yn), (sx - 1.6, y2 + 1.2)], color=NAVY, width=0.6, arrow=True)
    d.shape(sx - 1.8, y2 - 1.8, 3.6, 3.6, f"ellipse;html=1;fillColor=#FFFFFF;strokeColor={NAVY};strokeWidth={0.6 / 0.7045:.2f};"
                                         f"fontFamily=Arial;fontSize=8;fontColor={NAVY};spacing=0;", "+")
    arrow(d, sx + 1.8, sx + 6.5, y2)
    text(d, sx + 6.7, y2 - 1.6, 8, "η<sub>ij</sub>", size=6.0, h=3.2)
    arrow(d, sx + 11.0, sx + 15.5, y2)
    d.rect(sx + 15.5, y2 - 5.0, 33.0, 10.0, fill="#FFFFFF", stroke=RED, sw=0.6, r=10)
    text(d, sx + 15.5, y2 - 4.2, 33.0, "Pr{new association in (T, T + Δ]}<br>= 1 − exp{−Δ exp(η<sub>ij</sub>)}", align="center", h=8.6)
    text(d, sx + 13.5, y2 + 6.2, 37.0, "trained on cutoffs T′ with T′ + Δ ≤ T", align="center")
    hx, hy = sx + 54.0, y2 + 5.0
    d.path([(hx, hy - 10.0), (hx, hy), (hx + 12.0, hy)], color=INK, width=0.5)
    pts = [(hx + 0.2 + 11.0 * k / 14, hy - 0.2 - 9.0 * (1 - math.exp(-2.2 * k / 14))) for k in range(15)]
    d.path(pts, color=RED, width=0.8, curved=True)
    text(d, hx + 1.0, hy + 0.6, 12, "Δ", align="center")
    rx = 168.0
    for i, w_ in enumerate((9.0, 7.6, 6.4, 5.2, 4.2)):
        d.rect(rx, y2 - 5.6 + 2.3 * i, w_, 1.5, fill=tint(RED, 0.1 + 0.16 * i), stroke="none", r=30)
    text(d, rx - 3.0, y2 + 6.2, 15, f"{_n(R['world_model']['primary']['n_candidates'] / 1e6, 1)} million<br>pairs ranked",
         align="center", h=6.0)
    d.path([(hx + 12.8, y2 - 1.0), (rx - 0.8, y2 - 1.0)], color=NAVY, width=0.7, arrow=True)


def extension(d, R):
    D_EMB, K_WHITE, K_Z, n_x, n_tp = _dims()
    yb = 107.0
    mods = [("language", F6.BLOCKS["language"].stop - F6.BLOCKS["language"].start, tint(NAVY, 0.25)),
            ("attention", F6.BLOCKS["attention"].stop - F6.BLOCKS["attention"].start, ORANGE_A),
            ("context", F6.BLOCKS["context"].stop - F6.BLOCKS["context"].start, TEAL_G)]
    for i, (name, _, col) in enumerate(mods):
        d.rect(4.0, yb - 9.0 + 6.4 * i, 3.4, 4.6, fill=col, stroke="none")
        text(d, 8.2, yb - 9.6 + 6.4 * i, 14, name)
    d.path([(21.0, yb - 8.0), (21.6, yb - 8.0), (21.6, yb + 7.6), (21.0, yb + 7.6)], color=INK, width=0.4)
    arrow(d, 21.6, 25.0, yb)
    ylb = yb + 7.2
    x, _ = bar(d, 25.0, yb, F6.D_HID, tint(GREEN, 0.35), f"{F6.D_HID}", label_y=ylb)
    for k in range(2):
        gx = 36.0 + k * 15.5
        arrow(d, 27.6 + 0.6 if k == 0 else gx - 7.5, gx - 4.2, yb, GREEN)
        graph_glyph(d, gx + 1.5, yb, kinds=[(0, 1, "co"), (1, 2, "co"), (2, 4, "co"), (1, 4, "af"), (0, 3, "af"), (3, 4, "co"),
                                             (4, 5, "co")],
                    scale=0.85, node=1.8)
        text(d, gx - 5.0, ylb, 14, f"SAGE {k + 1}, {F6.D_HID}", align="center")
    text(d, 30.0, yb - 11.6, 34, "graph encoder (after PDFM)", align="center")
    xe0 = 66.5
    arrow(d, 61.0, xe0, yb, GREEN)
    h = bar_h(F6.D_Z)
    y0 = yb - h / 2
    for name, dim, col in mods:
        d.rect(xe0, y0, 2.8, h * dim / F6.D_Z, fill=col, stroke="none")
        y0 += h * dim / F6.D_Z
    text(d, xe0 + 1.4 - 7.5, ylb, 15, f"{F6.D_Z} (" + " + ".join(str(m[1]) for m in mods) + ")", align="center")
    xe = xe0 + 2.8
    lanes = {"recon": yb - 15.0, "dyn": yb - 4.0, "emb": yb + 5.0, "hist": yb + 14.0}
    d.path([(xe + 0.4, yb), (xe + 3.5, yb), (xe + 3.5, lanes["recon"]), (xe + 7.0, lanes["recon"])], color=GREEN, width=0.6,
           arrow=True, dashed=True)
    text(d, xe + 7.6, lanes["recon"] - 1.5, 44, "masked reconstruction (training)")
    for key in ("dyn", "emb"):
        d.path([(xe + 0.4, yb), (xe + 3.5, yb), (xe + 3.5, lanes[key]), (xe + 7.0, lanes[key])], color=GREEN, width=0.6, arrow=True)
    bx0, bw = xe + 7.4, 31.0
    d.box(bx0, lanes["dyn"] - 3.2, bw, 6.4, "latent dynamics F(z, a)<br>(after SensorWM)", fill="#FFFFFF", stroke=GREEN, rounded=True,
          align="center", valign="middle", size=5.5, pad=0)
    d.box(bx0, lanes["emb"] - 2.6, bw, 5.2, "z<sub>i</sub> × z<sub>j</sub>, |z<sub>i</sub> − z<sub>j</sub>|, cosine",
          fill="#FFFFFF", stroke=GREEN, rounded=True, align="center", valign="middle", size=5.5, pad=0)
    hc = 0.9
    hx0 = bx0 + 1.0
    cols = [ORANGE_A] * 4 + [TEAL_G] * 5 + [PALE] * n_tp
    for i, c in enumerate(cols):
        d.rect(hx0 + i * hc, lanes["hist"] - 1.2, hc - 0.15, 2.4, fill=c, stroke="none")
    text(d, bx0 - 8.0, lanes["hist"] + 1.6, bw + 18, "history: attention and growth, graph, entity type", align="center")
    tx, ty = bx0 + bw + 6.0, lanes["dyn"] - 9.0
    d.rect(tx - 1.0, ty - 2.5, 14.0, 6.5, fill="none", stroke=BROWN_T, sw=0.4)
    d.path([(tx, ty + 3.0), (tx + 3.0, ty + 1.2), (tx + 6.0, ty + 2.2), (tx + 9.0, ty - 0.6)], color=BROWN_T, width=0.8)
    d.path([(tx + 9.0, ty - 0.6), (tx + 12.0, ty - 2.0)], color=BROWN_T, width=0.8, dashed=True)
    text(d, tx - 9.0, ty - 7.4, 30, "TimesFM activity forecasts", align="center")
    d.path([(tx - 1.0, ty + 1.5), (bx0 + bw - 4.0, lanes["dyn"] - 3.4)], color=BROWN_T, width=0.6, arrow=True)
    text(d, tx - 17.0, ty + 0.2, 8, "a<sub>t</sub>", align="right")
    tcx, tcy = bx0 + bw + 26.0, yb + 4.0
    for (x1, y1), (x2, y2_) in [((0, -3.0), (-2.4, 0)), ((0, -3.0), (2.4, 0)), ((-2.4, 0), (-3.6, 3.0)), ((-2.4, 0), (-1.2, 3.0)),
                                ((2.4, 0), (1.2, 3.0)), ((2.4, 0), (3.6, 3.0))]:
        d.path([(tcx + x1, tcy + y1), (tcx + x2, tcy + y2_)], color=GREEN, width=0.5)
    for px, py in [(0, -3.0), (-2.4, 0), (2.4, 0), (-3.6, 3.0), (-1.2, 3.0), (1.2, 3.0), (3.6, 3.0)]:
        d.dot(tcx + px, tcy + py, 1.3, GREEN)
    text(d, tcx - 12, tcy + 5.2, 24, "boosted trees<br>(as in Earth AI)", align="center", h=6.0)
    tin = (tcx - 5.0, tcy)
    d.path([(bx0 + bw, lanes["dyn"]), (bx0 + bw + 15.0, lanes["dyn"]), (tin[0], tin[1] - 1.2)], color=GREEN, width=0.6, arrow=True)
    text(d, bx0 + bw + 0.8, lanes["dyn"] + 0.4, 15, "rollout cosines")
    d.path([(bx0 + bw, lanes["emb"]), tin], color=GREEN, width=0.6, arrow=True)
    d.path([(hx0 + len(cols) * hc + 0.6, lanes["hist"]), (bx0 + bw + 6.0, lanes["hist"]), (tin[0], tin[1] + 1.2)], color=MUTE,
           width=0.6, arrow=True)
    d.path([(tx + 6.0, ty + 4.0), (tcx - 1.0, tcy - 4.2)], color=BROWN_T, width=0.6, arrow=True)
    arrow(d, tcx + 4.6, tcx + 10.0, tcy)
    bar(d, tcx + 10.0, tcy, 1, RED, "score", w=2.4, label_y=tcy + 1.8)


def formulation(d, R):
    W_ = R["world_model"]
    pby = {int(k): v for k, v in W_["papers_by_year"].items()}
    T, D = 2022, 3
    letter(d, 0.2, 0.0, "a")
    x0, x1, base, hmax = 9.0, 57.0, 33.0, 22.0
    yrs = sorted(pby)
    bw = (x1 - x0) / len(yrs)
    top = max(pby.values())
    tick = 8000 if top > 8000 else 4000
    for i, y in enumerate(yrs):
        col = NAVY if y <= T else (RED if y <= T + D else INK)
        h = hmax * pby[y] / top
        d.rect(x0 + i * bw + 0.15, base - h, bw - 0.3, h, fill=col, stroke="none")
    d.path([(x0, base), (x1, base)], color=INK, width=0.5)
    d.path([(x0, base), (x0, base - hmax * tick / top)], color=INK, width=0.5)
    for v in (0, tick):
        yy = base - hmax * v / top
        d.path([(x0 - 0.8, yy), (x0, yy)], color=INK, width=0.5)
        text(d, x0 - 8.6, yy - 3.1, 7.6, f"{v:,}", align="right")
    text(d, x0 + 1.2, base - hmax * tick / top + 0.4, 18, "papers per year")
    for y in (2000, 2010, T, 2026):
        i = yrs.index(y)
        text(d, x0 + i * bw + bw / 2 - 5, base + 0.6, 10, "2026<br>Jan–Jun" if y == 2026 else str(y), align="center",
             h=6.0 if y == 2026 else 3.0)
    xT = x0 + (yrs.index(T) + 1) * bw
    xD = x0 + (yrs.index(T + D) + 1) * bw
    yb_ = base - hmax - 1.4
    d.path([(xT, yb_ - 4.6), (xT, base + 0.2)], color=INK, width=0.5, dashed=True)
    d.path([(xT + 0.3, yb_), (xD - 0.3, yb_)], color=RED, width=0.6)
    d.path([(x0 + 0.3, yb_), (xT - 0.3, yb_)], color=NAVY, width=0.6)
    text(d, xT - 26.0, yb_ - 5.2, 25.4, "state S<sub>T</sub> (papers ≤ T)", align="right")
    text(d, xT + 1.3, yb_ - 7.4, 14, "outcomes<br>(T, T + Δ]", align="left", h=6.0)
    text(d, x0, base + 7.2, 50, f"{_n(W_['corpus']['papers'])} PubMed papers; T = {T}, Δ = {D} (primary task)")
    nodes = [(0.0, -6.0), (6.0, -4.0), (-5.5, -2.5), (2.0, 0.5), (-2.5, 4.5), (5.5, 4.0), (9.0, -0.5), (-8.5, 3.0)]
    known = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (1, 6), (2, 7), (4, 7)]
    new = [(5, 6), (0, 4)]
    for k, gx in enumerate((72.0, 98.0)):
        gy = 17.0
        for a, b in known:
            d.path([(gx + nodes[a][0], gy + nodes[a][1]), (gx + nodes[b][0], gy + nodes[b][1])], color=TEAL_G, width=0.6)
        if k == 1:
            for a, b in new:
                d.path([(gx + nodes[a][0], gy + nodes[a][1]), (gx + nodes[b][0], gy + nodes[b][1])], color=RED, width=0.8,
                       dashed=True)
        for nx, ny in nodes:
            d.dot(gx + nx, gy + ny, 2.2, tint(TEAL_G, 0.15), stroke="#FFFFFF", sw=0.3)
        text(d, gx - 8, gy + 6.8, 16, "A<sub>T</sub>" if k == 0 else "A<sub>T + Δ</sub>", align="center", size=6.0, h=3.2)
    d.path([(83.6, 17.0), (86.8, 17.0)], color=NAVY, width=0.7, arrow=True)
    text(d, 60.0, 28.4, 52, "new association (red): two entities first<br>mentioned in one sentence by ≥ 2 papers",
         align="center", h=6.0)
    X = lambda yr: 131.0 + (yr - 2000) * (46.0 / 27.0)
    rows = [(2010, "train"), (2013, "train"), (2016, "train"), (2019, "train"), (T, "test"), (2025, "hold")]
    y0, s, bh = 3.6, 4.8, 2.0
    for k, (t, kind) in enumerate(rows):
        yy = y0 + k * s
        st_col = tint(NAVY, 0.6) if kind == "train" else NAVY
        end_state = X(t + 1)
        d.rect(X(2000), yy, end_state - X(2000), bh, fill=st_col, stroke="none")
        if kind == "train":
            d.rect(X(t + 1), yy, X(t + 1 + D) - X(t + 1), bh, fill=tint(RED, 0.55), stroke="none")
        elif kind == "test":
            d.rect(X(t + 1), yy, X(t + 1 + D) - X(t + 1), bh, fill=RED, stroke="none")
        else:
            d.rect(X(2026), yy, X(2026.5) - X(2026), bh, fill=INK, stroke="none")
    yb_top, yb_bot = y0 - 0.2, y0 + 3 * s + bh + 0.2
    d.path([(129.6, yb_top), (129.0, yb_top), (129.0, yb_bot), (129.6, yb_bot)], color=INK, width=0.4)
    text(d, 103.5, (yb_top + yb_bot) / 2 - 4.0, 25, "training cutoffs T′<br>(T′ + Δ ≤ T)", align="right", h=6.0)
    for k, lab in ((4, "test, T = 2022"), (5, "hold-out, T = 2025")):
        text(d, 103.5, y0 + k * s + bh / 2 - 1.7, 25, lab, align="right")
    ax_y = base
    d.path([(X(2000), ax_y), (X(2027), ax_y)], color=INK, width=0.5)
    for y in (2000, 2010, 2020, 2026):
        d.path([(X(y), ax_y), (X(y), ax_y + 0.8)], color=INK, width=0.5)
        text(d, X(y) - 3.5, ax_y + 0.6, 7, str(y), align="center")
    return ax_y + 4.0


def main():
    import pandas as pd
    from common import SRC
    R = load_results()
    d = Shifted()
    formulation(d, R)
    d.dy = 44.0
    world_model(d, R)
    export(d, "Figure1", W_MM, 118.0, page="Figure 1")
    e = Shifted()
    e.dy = -87.5
    extension(e, R)
    export(e, "ExtendedDataFig5", W_MM, 40.0, page="Extended Data Fig. 5")
    D_EMB, K_WHITE, K_Z, n_x, n_tp = _dims()
    dims = [b.stop - b.start for b in F6.BLOCKS.values()]
    pd.DataFrame([("emb", D_EMB), ("whitening", K_WHITE), ("z_extension_embedding", F6.D_Z),
                  ("z_interaction_axes", K_Z), ("n_x", n_x), ("n_tp", n_tp), ("mlp_input", 2 * K_Z + n_x + n_tp),
                  ("mlp_hidden", "64, 64"), ("hid", F6.D_HID), ("blocks", ", ".join(str(x) for x in dims[:-1]) + f" and {dims[-1]}")],
                 columns=["item", "value"]).to_csv(SRC / "architecture_dimensions.csv", index=False)
    pby = R["world_model"]["papers_by_year"]
    pd.DataFrame({"year": [int(k) for k in pby], "papers": list(pby.values())}).to_csv(SRC / "fig1a_papers_by_year.csv",
                                                                                         index=False)


if __name__ == "__main__":
    main()
