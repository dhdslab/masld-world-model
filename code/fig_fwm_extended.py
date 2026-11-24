from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common import load_results
from figstyle import FS_ANN, FULL, GOLD, INK, MM, NAVY, PURPLE, RED, TEAL, CardLayout, panel, save, setup, tint


def ed1(R):
    W, F = R["world_model"], R["foundation_world_model"]
    C = CardLayout(180, 117)
    fig = C.fig
    C.card(0, 4, 88, 54, "Corpus", "a")
    C.card(92, 4, 88, 54, "Yearly entity graphs", "b")
    C.card(0, 63, 88, 54, "Associations first reported after each cutoff", "c")
    C.card(92, 63, 88, 54, "Extension encoder reconstruction", "d", color="#59A14F")
    ax = C.ax(15, 12.5, 69, 34)
    py = {int(k): v for k, v in W["papers_by_year"].items()}
    ys = sorted(py)
    col = [tint(NAVY, 0.55) if y < 2020 else (INK if y == 2026 else NAVY) for y in ys]
    ax.bar(ys, [py[y] for y in ys], color=col, width=0.75)
    ax.set_xlim(1999.3, 2026.7); ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("Papers")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in (tint(NAVY, 0.55), NAVY, INK)],
              ["2000–2019", "2020–2025", "January–June 2026"], loc="upper left", fontsize=FS_ANN, handlelength=1.0)
    E = F["encoder"]
    yrs = sorted(int(t) for t in E["nodes_by_snapshot"])
    nodes = [E["nodes_by_snapshot"][str(t)] for t in yrs]
    co = [E["edges_by_snapshot"][str(t)]["co_mention"] for t in yrs]
    af = [E["edges_by_snapshot"][str(t)]["affinity"] for t in yrs]
    ax = C.ax(110, 12.5, 66, 31)
    ax.plot(yrs, nodes, color=INK, marker="o", ms=2.2, label="Entities")
    ax.plot(yrs, co, color=TEAL, marker="o", ms=2.2, label="Co-mention edges")
    ax.plot(yrs, af, color=PURPLE, marker="o", ms=2.2, label="Language-neighbour edges")
    trx = ax.get_xaxis_transform()
    ax.plot([2004.8, F["settings"]["encoder_cutoff"] + 0.2], [0.965, 0.965], color="#59A14F", lw=1.0, transform=trx,
            solid_capstyle="butt")
    ax.text(2005.0, 0.94, "extension encoder training", transform=trx, fontsize=FS_ANN, color=INK, ha="left", va="top")
    ax.set_yscale("log"); ax.set_xticks([2005, 2010, 2015, 2020, 2025])
    lo_, hi_ = min(nodes), max(co)
    yt = [t for t in (100, 1000, 10000, 100000, 1000000) if lo_ / 3 <= t <= hi_ * 3]
    ax.set_yticks(yt); ax.set_yticklabels([f"{t:,}" for t in yt]); ax.minorticks_off()
    ax.set_xlim(2004.5, 2025.5)
    ax.set_ylabel("Count (log scale)"); ax.set_xlabel("Year of the graph")
    ax.legend(loc="lower right", fontsize=FS_ANN)
    T = pd.DataFrame(W["task_sizes"]).sort_values("cutoff")
    ax = C.ax(15, 71.5, 69, 34)
    for d, ls in zip([1, 2, 3], [(0, (1, 1.2)), (0, (3, 1.5)), "-"]):
        k = f"positives_{d}"
        g = T[T[k].notna()] if k in T else T.iloc[:0]
        ax.plot(g.cutoff, g[k], color=INK, ls=ls, marker="o", ms=2.2, lw=0.9, label=f"Δ = {d} year{'s' if d > 1 else ''}")
    ax.set_xticks([2010, 2014, 2018, 2022])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("New associations"); ax.set_xlabel("Cutoff year")
    ax.legend(loc="upper left", fontsize=FS_ANN)
    ax = C.ax(110, 71.5, 66, 34)
    ml, ref = E["masked_node_loss"], E.get("mean_prediction_loss")
    mods = [("language", "Language", tint(NAVY, 0.25)), ("attention", "Attention", "#E07B39"), ("context", "Context", TEAL)]
    xs = range(3)
    ax.bar([x - (0.18 if ref else 0) for x in xs], [ml[k] for k, _, _ in mods], color=[c for *_, c in mods], width=0.34,
           label="Encoder, masked nodes")
    if ref:
        ax.bar([x + 0.18 for x in xs], [ref[k] for k, _, _ in mods], color="white", edgecolor=[c for *_, c in mods], lw=0.8,
               width=0.34, label="Mean of each input")
    ax.set_xticks(list(xs)); ax.set_xticklabels([m[1] for m in mods])
    ax.set_ylabel("Huber reconstruction loss")
    top = max(list(ml.values()) + (list(ref.values()) if ref else []))
    ax.set_ylim(0, top * 1.35)
    if ref:
        ax.legend([plt.Rectangle((0, 0), 1, 1, fc=INK, ec=INK), plt.Rectangle((0, 0), 1, 1, fc="white", ec=INK, lw=0.8)],
                  ["Encoder, masked nodes", "Mean of each input"], loc="upper right", fontsize=FS_ANN, handlelength=1.0,
                  frameon=False)
    save(fig, "SupplementaryFig2")


def main():
    setup()
    ed1(load_results())


if __name__ == "__main__":
    main()
