from __future__ import annotations

import os
import shutil
import subprocess
from html import escape
from pathlib import Path

import fitz

from common import load_results
from figstyle import FIG, INK, NAVY, RED, TEAL, GOLD, PURPLE, SKY, tint

U = 4.0
PT = 0.7045
W_MM, H_MM = 180, 117
DRAWIO_DIR = FIG / "drawio"
GREEN = "#2E7D32"
BLUE_B = "#5B7DB1"
BLUE_C = "#2C3E66"


def fs(pt):
    return round(pt / PT, 2)


class D:
    def __init__(self):
        self.cells, self.k = [], 1

    def _id(self):
        self.k += 1
        return f"c{self.k}"

    def box(self, x, y, w, h, html, fill="#FFFFFF", stroke="none", rounded=True, align="left", valign="top", size=6.0,
            color=INK, pad=1.4, bold=False):
        html = html.replace("<sub>", f'<sub style="font-size:{fs(5.2)}px">').replace("<sup>", f'<sup style="font-size:{fs(5.2)}px">')
        st = (f"rounded={1 if rounded else 0};arcSize=6;whiteSpace=wrap;html=1;align={align};verticalAlign={valign};"
              f"spacing=0;spacingLeft={pad * U:.1f};spacingRight={pad * U:.1f};spacingTop={0.9 * U:.1f};spacingBottom={0.6 * U:.1f};"
              f"fontFamily=Arial;fontSize={fs(size)};fontColor={color};fillColor={fill};strokeColor={stroke};"
              f"strokeWidth={0.5 / PT:.2f};{'fontStyle=1;' if bold else ''}")
        self.cells.append(f'<mxCell id="{self._id()}" value="{escape(html, quote=True)}" style="{st}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x * U:.1f}" y="{y * U:.1f}" width="{w * U:.1f}" height="{h * U:.1f}" as="geometry"/></mxCell>')

    def text(self, x, y, w, s, size=6.0, bold=False, color=INK, align="left"):
        self.box(x, y, w, size / 2.4, s, fill="none", stroke="none", rounded=False, align=align, size=size, color=color, pad=0,
                 bold=bold)

    def arrow(self, p0, p1, color="#6C7FAA", width=0.9, dashed=False):
        st = (f"endArrow=block;endFill=1;endSize=4;html=1;rounded=0;strokeColor={color};strokeWidth={width / PT:.2f};"
              f"{'dashed=1;dashPattern=3 2;' if dashed else ''}")
        self.cells.append(f'<mxCell id="{self._id()}" value="" style="{st}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                          f'<mxPoint x="{p0[0] * U:.1f}" y="{p0[1] * U:.1f}" as="sourcePoint"/>'
                          f'<mxPoint x="{p1[0] * U:.1f}" y="{p1[1] * U:.1f}" as="targetPoint"/></mxGeometry></mxCell>')

    def line(self, p0, p1, color=INK, width=0.6, dashed=False):
        st = (f"endArrow=none;html=1;rounded=0;strokeColor={color};strokeWidth={width / PT:.2f};"
              f"{'dashed=1;dashPattern=2 2;' if dashed else ''}")
        self.cells.append(f'<mxCell id="{self._id()}" value="" style="{st}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                          f'<mxPoint x="{p0[0] * U:.1f}" y="{p0[1] * U:.1f}" as="sourcePoint"/>'
                          f'<mxPoint x="{p1[0] * U:.1f}" y="{p1[1] * U:.1f}" as="targetPoint"/></mxGeometry></mxCell>')

    def xml(self):
        frame = (f'<mxCell id="frame" value="" style="rounded=0;fillColor=none;strokeColor=none;" vertex="1" parent="1">'
                 f'<mxGeometry x="0" y="0" width="{W_MM * U}" height="{H_MM * U}" as="geometry"/></mxCell>')
        return ('<mxfile><diagram id="fig3" name="Figure 3"><mxGraphModel grid="0" page="0" adaptiveColors="none"><root>'
                '<mxCell id="0"/><mxCell id="1" parent="0"/>' + frame + "".join(self.cells) + "</root></mxGraphModel></diagram></mxfile>")


def header(d, x, y, w, title, color):
    d.box(x, y, w, 5.6, f"<b>{title}</b>", fill=color, color="#FFFFFF", size=6.5, valign="middle", rounded=False, pad=1.6)


def build(R):
    W = R["world_model"]
    C = W["corpus"]
    n = lambda v: f"{int(v):,}"
    d = D()
    BG = "#EEF1F7"
    d.text(0.3, 0.2, 6, "<b>a</b>", size=8)
    header(d, 4, 1.5, 54, "Evidence and encoder", NAVY)
    d.box(4, 7.1, 54, 66, "", fill=BG, rounded=False)
    types = [t for t, k in (("genes", "Gene"), ("chemicals", "Chemical"), ("diseases", "Disease"), ("variants", "Variant"))
             if C["entity_types"].get(k, 0) > 0]
    types = ", ".join(types[:-1]) + " and " + types[-1]
    d.box(6, 9.0, 50, 10.5, f"<b>PubMed, 2000–2026</b><br>{n(C['papers'])} papers on steatotic liver disease", stroke=NAVY)
    d.arrow((33, 19.6), (33, 22.4))
    d.box(10, 22.6, 46, 10.5, f"<b>PubTator3 entities</b><br>{n(C['entities'] - C['entities_excluded_masld_family'] - C.get('entities_excluded_annotation_drift', 0))} {types}",
          stroke=NAVY)
    d.arrow((33, 33.2), (33, 36.0))
    d.box(10, 36.2, 46, 10.5, f"<b>Sentence co-mentions</b><br>{n(C['sentence_co_mentions'])} entity pairs in papers", stroke=NAVY)
    d.arrow((7.6, 19.6), (7.6, 49.8))
    d.box(6, 50.0, 23.5, 21.0, "<b>Disease names neutralized</b><br>NAFLD, MAFLD, MASLD → SLD", fill="#FFFFFF", stroke=NAVY)
    d.arrow((29.6, 60.5), (32.3, 60.5))
    d.box(32.5, 50.0, 23.5, 21.0, "<b>Frozen PubMedBERT</b><br>pretrained on PubMed abstracts collected in 2020; "
          "one 768-d vector e<sub>d</sub> per paper", fill=NAVY, color="#FFFFFF")
    d.text(60.3, 0.2, 6, "<b>b</b>", size=8)
    header(d, 64, 1.5, 54, "World-model state S<sub>T</sub>", BLUE_B)
    d.box(64, 7.1, 54, 66, "", fill="#EEF2F8", rounded=False)
    items = [
        ("Time-respecting whitening", "mean and axes from papers ≤ T only"),
        ("Semantic state h<sub>i</sub>(T)", "weighted mean of paper vectors; z<sub>i</sub>, 32 axes"),
        ("Semantic dynamics", "v<sub>i</sub> = h<sub>i</sub>(T) − h<sub>i</sub>(T−2); projection ĥ<sub>i</sub>(T+Δ)"),
        ("Field drift u<sub>T</sub>", "direction in which recent papers move"),
        ("Attention n<sub>i</sub>(T) and growth g<sub>i</sub>", "papers mentioning each entity"),
        ("Graph topology A<sub>T</sub>", "co-mentions already reported"),
    ]
    for k, (t, s) in enumerate(items):
        y = 9.0 + k * 10.6
        d.box(66, y, 50, 9.4, f"<b>{t}</b><br>{s}", stroke=BLUE_B)
    d.arrow((58.4, 40), (63.6, 40), color=NAVY, width=1.2)
    d.text(120.3, 0.2, 6, "<b>c</b>", size=8)
    header(d, 124, 1.5, 56, "Transition model", BLUE_C)
    d.box(124, 7.1, 56, 66, "", fill="#EBEEF4", rounded=False)
    d.box(126, 9.0, 52, 15.0, "<b>Pair features x<sub>ij</sub>(T)</b><br>semantic proximity, projected proximity, convergence, "
          "neighbourhood similarity, field drift, attention, growth, graph topology", stroke=BLUE_C)
    d.arrow((152, 24.1), (152, 27.0), color=BLUE_C)
    d.box(126, 27.2, 52, 10.5, "<b>Hazard of a new association</b><br>η<sub>ij</sub> = β′x<sub>ij</sub> + "
          "g<sub>θ</sub>(z<sub>i</sub>, z<sub>j</sub>, x<sub>ij</sub>)", stroke=BLUE_C)
    d.arrow((152, 37.8), (152, 40.7), color=BLUE_C)
    d.box(126, 40.9, 52, 10.5, "<b>Forecast</b><br>Pr(first reported in (T, T+Δ]) = 1 − exp(−Δ exp η<sub>ij</sub>)",
          fill=BLUE_C, color="#FFFFFF")
    d.box(126, 55.0, 50, 16.0, "<b>Learning β, θ</b><br>from earlier transitions S<sub>T′</sub> → A<sub>T′+Δ</sub>"
          "<br>with T′ + Δ ≤ T: labels end by the test cutoff", stroke=BLUE_C)
    d.line((176.0, 63.0), (178.9, 63.0), color=BLUE_C, dashed=True)
    d.line((178.9, 63.0), (178.9, 32.5), color=BLUE_C, dashed=True)
    d.arrow((178.9, 32.5), (178.0, 32.5), color=BLUE_C, dashed=True)
    d.arrow((118.4, 40), (123.6, 40), color=BLUE_B, width=1.2)
    d.text(0.3, 76.2, 6, "<b>d</b>", size=8)
    header(d, 4, 77.5, 176, "Strictly temporal evaluation", NAVY)
    x0, x1, y_ax = 12.0, 172.0, 112.5
    yr = lambda y: x0 + (y - 2000) / (2030 - 2000) * (x1 - x0)
    d.line((x0, y_ax), (x1, y_ax), color=INK, width=0.6)
    for y in (2000, 2005, 2010, 2015, 2020, 2025, 2030):
        d.line((yr(y), y_ax), (yr(y), y_ax + 1.2), color=INK, width=0.5)
        d.text(yr(y) - 5, y_ax + 1.0, 10, str(y), size=5.5, align="center")
    d.box(yr(2000), 87.2, yr(2010) - yr(2000), 6.2, "History (state only)", fill=tint(NAVY, 0.85), size=5.5, valign="middle",
          align="center", rounded=False)
    d.box(yr(2010), 87.2, yr(2020) - yr(2010), 6.2, "Training cutoffs T′ from 2010 to T − Δ", fill=tint(NAVY, 0.6), size=5.5,
          valign="middle", align="center", rounded=False)
    d.box(yr(2020), 87.2, yr(2024) - yr(2020), 6.2, "Test cutoffs T", fill=NAVY, size=5.5, valign="middle",
          align="center", rounded=False, color="#FFFFFF")
    d.box(yr(2024), 87.2, yr(2025) - yr(2024), 6.2, "", fill=tint(NAVY, 0.35), rounded=False)
    d.line((yr(2020.5), 84.0), (yr(2020.5), 87.2), color=INK, width=0.8, dashed=True)
    d.text(yr(2020.5) - 30.5, 83.3, 30, "encoder pretraining text ends", size=5.5, color=INK, align="right")
    rows = [(95.6, 2022, 2023.0, 2026.0, RED, "Primary task: state at the end of 2022 → associations 2023–2025"),
            (100.6, 2025, 2026.0, 2026.5, GOLD, "Most recent hold-out: state at the end of 2025 → January–June 2026")]
    for y, st, a, b, col, lab in rows:
        base = col if col != tint(GOLD, 0.45) else GOLD
        d.box(yr(st), y, yr(st + 1) - yr(st), 4.0, "", fill=tint(base, 0.65), rounded=False)
        d.box(yr(a), y, yr(b) - yr(a), 4.0, "", fill=col, rounded=False)
        d.text(yr(st) - 76.5, y - 0.1, 76, lab, size=5.5, color=INK, align="right")
    return d


def find_drawio():
    cands = [shutil.which("draw.io"), os.path.expandvars(r"%LOCALAPPDATA%\Programs\drawio\draw.io.exe"),
             r"C:\Program Files\draw.io\draw.io.exe"]
    return next((c for c in cands if c and Path(c).exists()), None)


def main():
    R = load_results()
    d = build(R)
    DRAWIO_DIR.mkdir(exist_ok=True)
    src = DRAWIO_DIR / "Figure3_world_model.drawio"
    src.write_text(d.xml(), encoding="utf-8")
    pdf = FIG / "Figure3.pdf"
    exe = find_drawio()
    subprocess.run([exe, "-x", "-f", "pdf", "--crop", "-b", "0", "-e", "-o", str(pdf), str(src), "--disable-gpu"],
                   capture_output=True, text=True, timeout=180)
    doc = fitz.open(pdf)
    page = doc[0]
    scale = (W_MM / 25.4 * 72) / page.rect.width
    out = fitz.open()
    np_ = out.new_page(width=page.rect.width * scale, height=page.rect.height * scale)
    np_.show_pdf_page(np_.rect, doc, 0)
    tmp = FIG / "_Figure3_scaled.pdf"
    out.save(tmp, garbage=3, deflate=True)
    out.close(); doc.close()
    os.replace(tmp, pdf)
    doc = fitz.open(pdf)
    doc[0].get_pixmap(dpi=600, alpha=False).save(FIG / "Figure3.png")
    svg = doc[0].get_svg_image(text_as_path=False)
    (FIG / "Figure3.svg").write_text(svg, encoding="utf-8")
    doc.close()


if __name__ == "__main__":
    raise SystemExit("Retired figure script (2026-11-20): its main() would overwrite Figure3, which is now another figure. "
                     "Only its helpers are imported (fig1_design, fig_trials_ed).")
