"""Figures for the M1 pilot blog post (MAX 26.6, SmolLM2-135M, Apple M4 Pro GPU).

    pixi run python analysis/plot_pilot.py

Reads the tracked CSVs in results/ and writes PNGs to blog/assets/modlearn/. Every value plotted is a
median over the 3 repeats of a cell (or over the 8 rounds of the TTFT-gap test).
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = REPO / "blog" / "assets" / "modlearn"

FIRST = RESULTS / "pilot-mac-smollm2-135m-cells.csv"
MINWIN = RESULTS / "pilot-mac-smollm2-135m-minwin5-cells.csv"
TTFT_GAP = RESULTS / "ttft-gap-mac.csv"

TOKENS = [10, 50, 100, 200, 1024, 2048]
CONCURRENCY = [1, 2, 4, 16, 32]
# Okabe-Ito order, checked for colour-vision-deficiency separation. Lines also get direct labels and markers.
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
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
    "lines.linewidth": 2,
    "lines.markersize": 8,
})
WIDTH_IN, DPI = 10, 160  # 1600 px wide


def medians(path: Path, column: str) -> dict[tuple[int, int], float]:
    groups: dict[tuple[int, int], list[float]] = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            groups[(int(r["output_tokens"]), int(r["concurrency"]))].append(float(r[column]))
    return {k: statistics.median(v) for k, v in groups.items()}


def plain_log_ticks(ax, axis: str, ticks: list[float]) -> None:
    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    target.set_major_formatter(fmt)
    target.set_minor_locator(NullLocator())


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print(f"wrote {OUT / name}")


def throughput() -> None:
    tput = medians(MINWIN, "throughput_tok_s")
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 6.2), layout="constrained")
    for i, t in enumerate(TOKENS):
        ys = [tput[(t, c)] for c in CONCURRENCY]
        ax.plot(CONCURRENCY, ys, color=COLORS[i], marker=MARKERS[i], label=f"{t} tokens")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_ticks(ax, "x", CONCURRENCY)
    plain_log_ticks(ax, "y", [50, 100, 200, 500, 1000, 2000])
    ax.set_xlim(0.85, 40)
    ax.set_xlabel("Concurrent requests (log scale)")
    ax.set_ylabel("Throughput (generated tok/s, log scale)")
    ax.set_title("Throughput scales ~23× from 1 to 32 concurrent requests")
    ax.legend(title="Output length", loc="upper left", title_fontsize=13)
    ax.text(0.99, 0.02, "MAX 26.6 · SmolLM2-135M bf16 · Apple M4 Pro GPU · median of 3 repeats",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color=MUTED)
    save(fig, "pilot-throughput.png")


def power_bias() -> None:
    before = medians(FIRST, "tok_s_per_w")
    after = medians(MINWIN, "tok_s_per_w")
    wall_first = medians(FIRST, "wall_s")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_IN, 5.6), layout="constrained",
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    fig.suptitle("Short batches overstated tok/s/W by up to 12×",
                 x=0.01, ha="left", fontsize=17, fontweight="bold", color=INK)

    # Left: concurrency 1, before vs after, per output length.
    xs = range(len(TOKENS))
    w = 0.38
    b = [before[(t, 1)] for t in TOKENS]
    a = [after[(t, 1)] for t in TOKENS]
    ax1.bar([x - w / 2 - 0.01 for x in xs], b, w, color=COLORS[3], label="Before: one batch per cell")
    ax1.bar([x + w / 2 + 0.01 for x in xs], a, w, color=COLORS[0], label="After: window ≥ 5 s")
    for x, v in zip(xs, b):
        ax1.text(x - w / 2, v + 3, f"{v:.0f}", ha="center", va="bottom", fontsize=11, color=INK)
    for x, v in zip(xs, a):
        ax1.text(x + w / 2, v + 3, f"{v:.0f}", ha="center", va="bottom", fontsize=11, color=INK)
    ax1.set_xticks(list(xs), [str(t) for t in TOKENS])
    ax1.set_xlabel("Output tokens per request")
    ax1.set_ylabel("Efficiency (tok/s/W)")
    ax1.set_title("Concurrency 1", fontsize=15)
    ax1.grid(axis="x", visible=False)
    ax1.set_ylim(0, 225)
    ax1.legend(loc="upper right", fontsize=12)

    # Right: all 30 cells, overstatement ratio vs how long the original batch lasted.
    for i, t in enumerate(TOKENS):
        xs2 = [wall_first[(t, c)] for c in CONCURRENCY]
        ys2 = [before[(t, c)] / after[(t, c)] for c in CONCURRENCY]
        ax2.scatter(xs2, ys2, color=COLORS[i], marker=MARKERS[i], s=70, label=f"{t}",
                    edgecolors="white", linewidths=1, zorder=3)
    ax2.axhline(1, color=MUTED, linewidth=1.2, linestyle="--", zorder=2)
    ax2.axvline(1, color=GRID, linewidth=1.2, zorder=1)
    ax2.text(1.07, 11, "1 s = one\npowermetrics\nsample", fontsize=11, color=MUTED, va="top")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    plain_log_ticks(ax2, "x", [0.2, 0.5, 1, 2, 5, 10, 20, 50])
    plain_log_ticks(ax2, "y", [0.5, 1, 2, 5, 10])
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax2.set_xlabel("Original batch duration (s, log)")
    ax2.set_ylabel("tok/s/W before ÷ after (log)")
    ax2.set_title("All 30 cells", fontsize=15)
    ax2.legend(title="Tokens", loc="upper right", fontsize=11, title_fontsize=11, ncol=2,
               handletextpad=0.2, columnspacing=0.8)
    save(fig, "pilot-power-bias.png")


def ttft_gap() -> None:
    points: dict[str, dict[float, list[float]]] = {"gpu": defaultdict(list), "cpu": defaultdict(list)}
    with open(TTFT_GAP) as f:
        for r in csv.DictReader(f):
            if r["error"]:
                continue
            points[r["label"]][float(r["gap_s"])].append(float(r["ttft_s"]) * 1000)

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 6), layout="constrained")
    series = [("gpu", "MAX on M4 Pro GPU (bf16)", COLORS[0], "o"),
              ("cpu", "MAX on CPU (fp32, control)", COLORS[3], "s")]
    for key, label, color, marker in series:
        gaps = sorted(points[key])
        # Plot a 0 s gap at 0.02 s so it fits on a log axis; the tick is labelled "0".
        xpos = [g if g > 0 else 0.02 for g in gaps]
        for x, g in zip(xpos, gaps):
            ax.scatter([x] * len(points[key][g]), points[key][g], color=color, alpha=0.22, s=26,
                       marker=marker, linewidths=0, zorder=2)
        med = [statistics.median(points[key][g]) for g in gaps]
        ax.plot(xpos, med, color=color, marker=marker, label=label, zorder=3)
        ax.text(xpos[-1] * 1.12, med[-1], label.split(" (")[0].replace("MAX on ", ""), color=INK,
                va="center", fontsize=13)
    ax.axvspan(1, 2, color="#f2f2f2", zorder=0)
    ax.text(1.41, 285, "wake-up step\n(+~160 ms)", ha="center", va="bottom", fontsize=12, color=INK)
    ax.set_xscale("log")
    ticks = [0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v < 0.03 else f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0.015, 11)
    ax.set_ylim(0, 340)
    ax.set_xlabel("Idle gap before the request (s, log scale)")
    ax.set_ylabel("Time to first token (ms)")
    ax.set_title("The GPU pays ~160 ms to wake after 1–2 s idle; the CPU doesn't", fontsize=17)
    ax.legend(loc="upper left")
    ax.text(0.99, 0.02, "single requests, 8 rounds per gap · line = median, faint dots = rounds",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color=MUTED)
    save(fig, "pilot-ttft-gap.png")


if __name__ == "__main__":
    throughput()
    power_bias()
    ttft_gap()
