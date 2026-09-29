"""Figures for the "why this paper" blog post, drawn from the paper's own Table 2.

    pixi run python analysis/plot_paper.py

Reads results/paper/sada2025-table2.csv (transcribed from arXiv:2507.00418v3, 200 tokens x 4 parallel requests)
and writes PNGs to blog/assets/modlearn/.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TABLE2 = REPO / "results" / "paper" / "sada2025-table2.csv"
OUT = REPO / "blog" / "assets" / "modlearn"

# Same blue / vermillion as the pilot figures (validated for colour-vision-deficiency separation).
BLUE, VERMILLION = "#0072B2", "#D55E00"
INK, MUTED, GRID = "#1a1a1a", "#5f5f5f", "#e6e6e6"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 15,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 1,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.frameon": False,
    "legend.fontsize": 13,
})


def load() -> list[dict]:
    rows = []
    with open(TABLE2) as f:
        for r in csv.DictReader(f):
            a_tps, a_w = float(r["a100_tok_s"]), float(r["a100_w"])
            q_tps, q_w = float(r["qaic_tok_s"]), float(r["qaic_w"])
            rows.append({
                "label": f"{r['model']}  ({r['qaic_devices']} vs {r['a100_gpus']}×A100)",
                "model": r["model"],
                "power_x": a_w / q_w,  # how many times less power Qualcomm draws
                "tput_x": a_tps / q_tps,  # how many times less throughput Qualcomm delivers
                "eff_ratio": (q_tps / q_w) / (a_tps / a_w),  # Qualcomm tok/s/W ÷ A100 tok/s/W
            })
    return sorted(rows, key=lambda r: r["eff_ratio"])


def times(v, _pos=None) -> str:
    return f"{v:g}×"


def power_vs_throughput(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(11, 7.4))
    y = list(range(len(rows)))
    for i, r in enumerate(rows):
        ax.plot([r["power_x"], r["tput_x"]], [i, i], color="#b5b5b5", lw=2, zorder=1, solid_capstyle="round")
    ax.scatter([r["power_x"] for r in rows], y, s=90, color=BLUE, marker="o", zorder=3,
               edgecolor="white", linewidth=2, label="Less power drawn")
    ax.scatter([r["tput_x"] for r in rows], y, s=100, color=VERMILLION, marker="D", zorder=3,
               edgecolor="white", linewidth=2, label="Less throughput delivered")
    for i, r in enumerate(rows):
        if r["eff_ratio"] > 1:
            ax.annotate("more efficient per token", (max(r["power_x"], r["tput_x"]), i), xytext=(12, 0),
                        textcoords="offset points", va="center", fontsize=12, color=INK)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(times))
    ax.set_xlim(5, 130)
    ax.set_yticks(y, [r["label"] for r in rows], color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Qualcomm vs the A100 setup: × lower (log scale)")
    ax.set_title("7–35× less power, but 12–75× less throughput", pad=40)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.6,
              borderaxespad=0.2)
    fig.text(0.99, 0.01, "Sada et al., arXiv:2507.00418, Table 2 · 200 output tokens × 4 parallel requests",
             ha="right", va="bottom", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "paper-power-vs-throughput.png", dpi=150)
    plt.close(fig)


def efficiency_ratio(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    y = list(range(len(rows)))
    for i, r in enumerate(rows):
        v = r["eff_ratio"]
        color = BLUE if v > 1 else VERMILLION
        ax.barh(i, v - 1 if v > 1 else 1 - v, left=1 if v > 1 else v, height=0.62, color=color, zorder=2)
        ax.annotate(f"{v:.2f}×", (v, i), xytext=(8 if v > 1 else -8, 0), textcoords="offset points",
                    va="center", ha="left" if v > 1 else "right", fontsize=12, color=INK)
    ax.axvline(1, color=INK, lw=1.5, zorder=3)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 3]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(times))
    ax.set_xlim(0.08, 4)
    ax.set_yticks(y, [r["model"] for r in rows], color=INK)
    ax.grid(axis="y", visible=False)
    ax.text(1.06, len(rows) - 0.4, "Qualcomm better →", color=MUTED, fontsize=12, va="center")
    ax.text(0.94, len(rows) - 0.4, "← A100 better", color=MUTED, fontsize=12, va="center", ha="right")
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("Qualcomm tok/s/W ÷ A100 tok/s/W (log scale, 1× = parity)")
    ax.set_title("Per token, Qualcomm is more efficient on 3 of 12 models")
    fig.text(0.99, 0.01, "Sada et al., arXiv:2507.00418, Table 2 · 200 output tokens × 4 parallel requests",
             ha="right", va="bottom", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "paper-efficiency-ratio.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    power_vs_throughput(rows)
    efficiency_ratio(rows)
    for r in rows:
        print(f"{r['model']:20} power {r['power_x']:5.1f}×  throughput {r['tput_x']:5.1f}×  efficiency {r['eff_ratio']:.2f}×")


if __name__ == "__main__":
    main()
