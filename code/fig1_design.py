from __future__ import annotations

import os
import subprocess
from html import escape

import fitz

from common import load_results
from fig3_world_model import D, DRAWIO_DIR, U, find_drawio, fs, PT
from figstyle import DOMAIN_COLOR, FIG, INK, NAVY, RED, tint

W_MM, H_MM = 180, 112
GREEN = "#59A14F"
TYPE_COL = {"Gene": "#4E79A7", "Chemical": "#E15759", "Disease": "#76B7B2", "Variant": "#EDC948"}
MUTE = "#8C96AA"


class Icons(D):

    def shape(self, x, y, w, h, style, value=""):
        self.cells.append(f'<mxCell id="{self._id()}" value="{escape(value, quote=True)}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x * U:.1f}" y="{y * U:.1f}" width="{w * U:.1f}" height="{h * U:.1f}" as="geometry"/></mxCell>')

    def rect(self, x, y, w, h, fill="#FFFFFF", stroke=NAVY, sw=0.5, r=0):
        self.shape(x, y, w, h, f"rounded={1 if r else 0};arcSize={r};html=1;fillColor={fill};strokeColor={stroke};"
                               f"strokeWidth={sw / PT:.2f};")

    def dot(self, cx, cy, d, fill, stroke="none", sw=0.4):
        self.shape(cx - d / 2, cy - d / 2, d, d, f"ellipse;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={sw / PT:.2f};")

    def path(self, pts, color=INK, width=0.6, arrow=False, dashed=False, curved=False, extra=""):
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        mids = "".join(f'<mxPoint x="{x * U:.1f}" y="{y * U:.1f}"/>' for x, y in pts[1:-1])
        st = (f"endArrow={'block;endFill=1;endSize=3' if arrow else 'none'};html=1;rounded=0;{'curved=1;' if curved else ''}"
              f"strokeColor={color};strokeWidth={width / PT:.2f};{'dashed=1;dashPattern=2 1.5;' if dashed else ''}{extra}")
        self.cells.append(f'<mxCell id="{self._id()}" value="" style="{st}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                          f'<mxPoint x="{x0 * U:.1f}" y="{y0 * U:.1f}" as="sourcePoint"/>'
                          f'<mxPoint x="{x1 * U:.1f}" y="{y1 * U:.1f}" as="targetPoint"/>'
                          + (f'<Array as="points">{mids}</Array>' if mids else "") + "</mxGeometry></mxCell>")

    def num(self, cx, cy, k, fill=NAVY):
        self.shape(cx - 1.9, cy - 1.9, 3.8, 3.8, f"ellipse;html=1;fillColor={fill};strokeColor=none;fontColor=#FFFFFF;"
                                                f"fontFamily=Arial;fontSize={fs(5.5)};fontStyle=1;spacing=0;", str(k))

    def label(self, cx, y, w, title, sub="", k=None, color=NAVY):
        if k is not None:
            self.num(cx - w / 2 + 1.9, y + 1.45, k, color)
            self.box(cx - w / 2 + 4.3, y, w - 4.3, 3.2, f"<b>{title}</b>", fill="none", rounded=False, align="left", size=6.5,
                     pad=0, valign="middle")
        else:
            self.box(cx - w / 2, y, w, 3.2, f"<b>{title}</b>", fill="none", rounded=False, align="center", size=6.5, pad=0,
                     valign="middle")
        if sub:
            self.box(cx - w / 2, y + 3.4, w, 6.6, sub, fill="none", rounded=False, align="center" if k is None else "left",
                     size=5.5, pad=0, color=INK)


def ic_literature(d, cx, cy):
    for k, (dx, dy, c) in enumerate(((-4.2, 1.6, tint(NAVY, 0.7)), (-2.1, 0.8, tint(NAVY, 0.45)), (0.0, 0.0, NAVY))):
        d.shape(cx - 4.5 + dx + 2.1, cy - 6.0 + dy, 9.0, 11.5,
                f"shape=mxgraph.basic.document;html=1;fillColor=#FFFFFF;strokeColor={c};strokeWidth={0.6 / PT:.2f};")
    for i in range(4):
        d.path([(cx - 1.2, cy - 2.6 + 1.9 * i), (cx + 3.6 - (1.4 if i == 3 else 0), cy - 2.6 + 1.9 * i)], color=tint(NAVY, 0.4),
               width=0.5)


def ic_entities(d, cx, cy):
    d.rect(cx - 8.5, cy - 5.5, 17, 11, fill="#FFFFFF", stroke=NAVY, sw=0.6, r=8)
    rows = [[("Gene", 3.2), (None, 5.0), ("Disease", 4.0)], [(None, 2.6), ("Chemical", 4.4), (None, 5.6)],
            [("Variant", 3.4), (None, 3.0), ("Gene", 3.6)]]
    for i, row in enumerate(rows):
        x = cx - 7.0
        y = cy - 3.0 + 3.0 * i
        for t, w in row:
            if t:
                d.rect(x, y - 0.9, w, 1.8, fill=TYPE_COL[t], stroke="none", r=30)
            else:
                d.path([(x + 0.3, y), (x + w - 0.3, y)], color=tint(NAVY, 0.55), width=0.5)
            x += w + 0.6
    d.path([(cx - 5.4, cy - 3.9), (cx - 0.5, cy - 8.2), (cx + 5.0, cy - 3.9)], color=INK, width=0.5, curved=True)


def ic_encoder(d, cx, cy):
    cols = [(-6.5, 3), (-2.5, 4), (1.5, 3)]
    pts = []
    for x, n in cols:
        ys = [cy + (i - (n - 1) / 2) * 2.6 for i in range(n)]
        pts.append([(cx + x, y) for y in ys])
    for a, b in zip(pts[:-1], pts[1:]):
        for p in a:
            for q in b:
                d.path([p, q], color=tint(NAVY, 0.6), width=0.3)
    for col in pts:
        for x, y in col:
            d.dot(x, y, 1.7, NAVY)
    for i in range(6):
        d.rect(cx + 4.6, cy - 4.8 + 1.6 * i, 2.2, 1.45, fill=tint(NAVY, 0.15 + 0.13 * i), stroke="none")
    d.path([(cx + 2.6, cy), (cx + 4.3, cy)], color=NAVY, width=0.5, arrow=True)


def ic_state(d, cx, cy):
    nodes = [(-6.5, -3.0, "Gene"), (-2.0, -5.2, "Chemical"), (3.5, -3.6, "Disease"), (-4.0, 2.6, "Disease"),
             (1.5, 1.2, "Gene"), (6.5, 3.4, "Variant")]
    edges = [(0, 1), (1, 2), (0, 3), (1, 4), (3, 4), (4, 2), (4, 5)]
    for a, b in edges:
        d.path([(cx + nodes[a][0], cy + nodes[a][1]), (cx + nodes[b][0], cy + nodes[b][1])], color=MUTE, width=0.5)
    for x, y, t in nodes:
        d.dot(cx + x, cy + y, 2.6, TYPE_COL[t], stroke="#FFFFFF", sw=0.4)
    for i, (dx, dy) in ((1, (2.4, -1.6)), (5, (2.2, 1.9)), (3, (-2.3, 1.8))):
        x, y = nodes[i][0], nodes[i][1]
        d.path([(cx + x + 0.9 * (1 if dx > 0 else -1), cy + y + 0.7 * (1 if dy > 0 else -1)), (cx + x + dx, cy + y + dy)],
               color=INK, width=0.5, arrow=True)


def ic_hazard(d, cx, cy):
    d.dot(cx - 6.6, cy + 2.0, 2.6, TYPE_COL["Gene"], stroke="#FFFFFF")
    d.dot(cx - 0.6, cy + 2.0, 2.6, TYPE_COL["Disease"], stroke="#FFFFFF")
    d.path([(cx - 5.4, cy + 1.2), (cx - 3.6, cy - 1.9), (cx - 1.8, cy + 1.2)], color=RED, width=0.6, dashed=True, curved=True)
    x0, y0 = cx + 2.0, cy + 4.6
    d.path([(x0, y0 - 9.5), (x0, y0), (x0 + 7.5, y0)], color=INK, width=0.5)
    import math
    pts = [(x0 + 0.2 + 7.0 * k / 12, y0 - 0.2 - 8.2 * (1 - math.exp(-2.4 * k / 12))) for k in range(13)]
    d.path(pts, color=RED, width=0.8, curved=True)
    d.box(x0 + 1.0, y0 + 0.3, 6.5, 2.6, "Δ", fill="none", rounded=False, align="center", size=5.5, pad=0)


def ic_forecast(d, cx, cy):
    d.shape(cx - 6.0, cy - 6.2, 10.5, 12.4, f"shape=mxgraph.basic.document;html=1;fillColor=#FFFFFF;strokeColor={NAVY};"
                                             f"strokeWidth={0.6 / PT:.2f};")
    for i, w in enumerate((6.6, 5.4, 4.4, 3.4)):
        d.dot(cx - 4.1, cy - 3.6 + 2.3 * i, 1.0, NAVY)
        d.rect(cx - 3.2, cy - 4.2 + 2.3 * i, w, 1.2, fill=tint(NAVY, 0.1 + 0.2 * i), stroke="none", r=30)


def ic_wheel(d, cx, cy, r=6.4):
    order = list(DOMAIN_COLOR)
    for k, dname in enumerate(order):
        a0, a1 = k / len(order), (k + 1) / len(order) - 0.012
        d.shape(cx - r, cy - r, 2 * r, 2 * r, f"shape=mxgraph.basic.partConcEllipse;html=1;startAngle={a0:.3f};endAngle={a1:.3f};"
                                             f"arcWidth=0.32;fillColor={DOMAIN_COLOR[dname]};strokeColor=none;")
    for x, y in ((-1.5, -1.0), (1.2, -1.6), (0.3, 1.4), (-1.8, 1.3), (2.0, 0.6)):
        d.dot(cx + x, cy + y, 0.9, tint(NAVY, 0.3))


def ic_evaluate(d, cx, cy):
    for i, h in enumerate((4.0, 6.5, 9.0)):
        d.rect(cx - 6.5 + 2.6 * i, cy + 4.5 - h, 1.9, h, fill=tint(NAVY, 0.55 - 0.2 * i), stroke="none")
    d.shape(cx + 0.2, cy - 5.0, 7.5, 7.5, f"shape=mxgraph.ios7.icons.looking_glass;html=1;fillColor=none;strokeColor={NAVY};"
                                         f"strokeWidth={0.8 / PT:.2f};")


def ic_compare(d, cx, cy):
    for i in range(3):
        d.rect(cx - 9.0, cy - 1.6 + 2.5 * i, 6.0, 1.9, fill=tint(NAVY, 0.15 + 0.25 * i), stroke="none", r=20)
        d.rect(cx + 2.0, cy - 1.6 + 2.5 * i, 6.0, 1.9, fill=tint(NAVY, 0.15 + 0.25 * i), stroke="none", r=20)
    for i in range(2):
        d.rect(cx + 2.0, cy - 6.6 + 2.5 * i, 6.0, 1.9, fill=tint(GREEN, 0.1 + 0.3 * i), stroke="none", r=20)
    d.box(cx - 3.0, cy + 0.6, 5.0, 3.0, "vs", fill="none", rounded=False, align="center", size=5.5, pad=0)


def build(R):
    W = R["world_model"]
    C = W["corpus"]
    EX = W["a8"]["exact_primary"]
    cut = sorted({g["cutoff"] for g in W["grid"]})
    first_label, holdout_year = cut[0] + 1, 2026
    ST = R["structure"]
    n = lambda v: f"{int(v):,}"
    kept = C["entities"] - C["entities_excluded_masld_family"] - C.get("entities_excluded_annotation_drift", 0)
    d = Icons()
    xs = [15, 45, 75, 105, 135, 165]
    yi, wl = 13.0, 28.5
    steps = [(ic_literature, "Literature", f"PubMed, 2000–2026<br>{n(C['papers'])} papers"),
             (ic_entities, "Entities", f"{n(kept)} genes, chemicals,<br>diseases and variants"),
             (ic_encoder, "Encoding", "names neutralized;<br>frozen PubMedBERT"),
             (ic_state, "State S<sub>T</sub>", "semantic state, velocity,<br>attention, growth, graph"),
             (ic_hazard, "Transition", "hazard of a new<br>association"),
             (ic_forecast, "Forecast", "every candidate pair<br>ranked")]
    for k, (cx, (icon, title, sub)) in enumerate(zip(xs, steps), 1):
        icon(d, cx, yi)
        d.label(cx, 22.5, wl, title, sub, k=k)
    for a, b in zip(xs[:-1], xs[1:]):
        d.path([(a + 10.6, yi), (b - 10.6, yi)], color=NAVY, width=0.9, arrow=True)
    yb = 49.0
    P = W["primary"]
    XA, XE, XC = 66, 110, 154
    outs = [(XA, ic_wheel, "Semantic atlas", f"{ST['n_research_topics']} topics in {ST['n_domains']} domains;<br>descriptive",
             "Fig. 3"),
            (XE, ic_evaluate, "Forecast evaluation", f"{n(EX['n_positives'])} new associations among<br>"
             f"{n(P['n_candidates'])} pairs (primary task)", "Figs. 4 and 5"),
            (XC, ic_compare, "Comparison", "+ PDFM, SensorWM, TimesFM<br>and Earth AI components", "Fig. 6")]
    for cx, icon, title, sub, ref in outs:
        icon(d, cx, yb)
        d.label(cx, yb + 8.8, 32, title, sub + f"<br><i>{ref}</i>")
    yd = 37.0
    d.path([(98, 32.6), (98, yd), (XA, yd), (XA, yb - 7.6)], color=MUTE, width=0.6, arrow=True)
    d.path([(165, 32.6), (165, yd), (XE, yd), (XE, yb - 7.6)], color=MUTE, width=0.6, arrow=True)
    d.path([(XC, yd), (XC, yb - 7.6)], color=GREEN, width=0.6, arrow=True)
    ic_literature(d, 22.0, yb)
    d.box(7.0, yb + 7.4, 30, 9.8, f"companion corpus,<br>{n(R['corpus']['flow']['included_literature'])} papers, 2020–2026<br><i>Fig. 2</i>",
          fill="none", rounded=False, align="center", size=5.5, pad=0)
    d.path([(29.0, yb), (XA - 8.5, yb)], color=MUTE, width=0.6, arrow=True, dashed=True)
    d.box((29.0 + XA - 8.5) / 2 - 12, yb - 4.4, 24, 3.0, "topic model", fill="none", rounded=False, align="center", size=5.5,
          pad=0)
    y = 98.0
    X = lambda yr: 14 + (yr - 2000) * (156 / 27)
    E = lambda T: X(T + 1)
    d.path([(X(2010), 73.0), (X(2026.5), 73.0)], color=INK, width=0.4)
    d.box(X(2010), 73.6, X(2026.5) - X(2010), 3.0, "<b>retrospective</b> (exploratory)", fill="none", rounded=False,
          align="center", size=5.5, pad=0)
    d.rect(X(2000), 78.4, X(2026.5) - X(2000), 1.5, fill=tint(NAVY, 0.75), stroke="none")
    d.box(X(2000), 80.3, 46, 3.0, "papers up to T build the state S<sub>T</sub>", fill="none", rounded=False, size=5.5, pad=0)
    for y0_, y1_ in ((78.0, 84.4), (89.2, y)):
        d.path([(X(2020) + 0.8, y0_), (X(2020) + 0.8, y1_)], color=INK, width=0.5, dashed=True)
    d.box(X(2020) - 24.8, 80.3, 25, 3.0, "end of PubMedBERT text", fill="none", rounded=False, size=5.5, pad=0, align="right")
    for t in range(min(tt["cutoff"] for tt in W["task_sizes"]), cut[0]):
        d.path([(E(t), 85.4), (E(t), 88.6)], color=tint(NAVY, 0.45), width=0.8)
    for t in cut:
        if t <= 2023:
            d.dot(E(t), 87.0, 2.0, NAVY, stroke=NAVY, sw=0.5)
        else:
            d.shape(E(t) - 1.2, 85.8, 2.4, 2.4, f"rhombus;html=1;fillColor=#FFFFFF;strokeColor={NAVY};strokeWidth={0.5 / PT:.2f};")
    d.dot(E(2022), 87.0, 3.2, "none", stroke=NAVY, sw=0.5)
    d.rect(X(2023), 90.4, X(2026) - X(2023), 1.6, fill=NAVY, stroke="none")
    d.path([(E(2022) - 0.6, 88.4), (X(2023) + 0.4, 90.2)], color=NAVY, width=0.5)
    d.box(X(2023), 92.4, 30, 3.0, "primary outcome window", fill="none", rounded=False, size=5.5, pad=0)
    d.rect(X(2026), 85.0, X(2026.5) - X(2026), 4.0, fill=INK, stroke="none")
    d.path([(X(2000), y), (X(2027), y)], color=INK, width=0.6, arrow=True)
    for yr in (2000, 2010, 2020, 2026):
        d.path([(X(yr), y), (X(yr), y + 1.0)], color=INK, width=0.5)
        d.box(X(yr) - 5, y + 1.2, 10, 3.0, str(yr), fill="none", rounded=False, align="center", size=5.5, pad=0)
    yl = y + 4.8
    d.box(E(2010) - 1, yl, 44, 3.0, "training cutoffs T′ (T′ + Δ ≤ T)", fill="none", rounded=False, size=5.5, pad=0)
    d.box(E(cut[0]) - 1, yl, 40, 3.0, "test cutoffs T (◊ not pre-specified)", fill="none", rounded=False, size=5.5, pad=0)
    d.box(X(2026.5) + 0.8, 84.3, 11, 3.0, "hold-out", fill="none", rounded=False, size=5.5, pad=0)
    return d


def export(d, name, w_mm=W_MM, h_mm=H_MM, page="Figure 1"):
    DRAWIO_DIR.mkdir(exist_ok=True)
    src = DRAWIO_DIR / f"{name}.drawio"
    xml = d.xml().replace('<diagram id="fig3" name="Figure 3">', f'<diagram id="{name}" name="{page}">')
    xml = xml.replace('width="720.0" height="468.0"', f'width="{w_mm * U:.1f}" height="{h_mm * U:.1f}"')
    src.write_text(xml, encoding="utf-8")
    pdf = FIG / f"{name}.pdf"
    r = subprocess.run([find_drawio(), "-x", "-f", "pdf", "--crop", "-b", "0", "-e", "-o", str(pdf), str(src), "--disable-gpu"],
                       capture_output=True, text=True, timeout=180)
    assert pdf.exists(), r.stderr
    doc = fitz.open(pdf)
    page_ = doc[0]
    scale = (w_mm / 25.4 * 72) / page_.rect.width
    out = fitz.open()
    np_ = out.new_page(width=page_.rect.width * scale, height=page_.rect.height * scale)
    np_.show_pdf_page(np_.rect, doc, 0)
    tmp = FIG / f"_{name}_scaled.pdf"
    out.save(tmp, garbage=3, deflate=True)
    out.close(); doc.close()
    os.replace(tmp, pdf)
    doc = fitz.open(pdf)
    doc[0].get_pixmap(dpi=600, alpha=False).save(FIG / f"{name}.png")
    (FIG / f"{name}.svg").write_text(doc[0].get_svg_image(text_as_path=False), encoding="utf-8")
    doc.close()
    return src


def main():
    import pandas as pd
    from common import SRC
    R = load_results()
    d = build(R)
    export(d, "SupplementaryFig1", page="Supplementary Fig. 1")
    W, ST = R["world_model"], R["structure"]
    C = W["corpus"]
    cut = sorted({g["cutoff"] for g in W["grid"]})
    pd.DataFrame([
        ("papers", C["papers"]), ("entities_after_measurement_rules",
                                  C["entities"] - C["entities_excluded_masld_family"] - C.get("entities_excluded_annotation_drift", 0)),
        ("research_topics", ST["n_research_topics"]), ("domains", ST["n_domains"]),
        ("first_training_cutoff", min(t["cutoff"] for t in W["task_sizes"])), ("test_cutoffs", f"{cut[0]}-{cut[-1]}"),
        ("hold_out", "2026-01 to 2026-06")], columns=["item", "value"]).to_csv(
        SRC / "fig1_design_numbers.csv", index=False)


if __name__ == "__main__":
    main()
