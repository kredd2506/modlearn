"""Print a cells CSV as the paper-style grid: output tokens (rows) x concurrency (columns), median over repeats.

    pixi run summarize results/pilot-mac-smollm2-135m-cells.csv [--metric tok_s_per_w]
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def grid(rows: list[dict], metric: str) -> str:
    vals: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in rows:
        if r.get(metric) not in ("", None):
            vals[int(r["output_tokens"]), int(r["concurrency"])].append(float(r[metric]))
    tokens = sorted({t for t, _ in vals})
    concs = sorted({c for _, c in vals})
    lines = [
        f"| {metric} (median) | " + " | ".join(f"conc {c}" for c in concs) + " |",
        "|---|" + "---:|" * len(concs),
    ]
    for t in tokens:
        cells = [f"{statistics.median(vals[t, c]):.3g}" if vals.get((t, c)) else "–" for c in concs]
        lines.append(f"| {t} tokens | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells_csv", type=Path)
    ap.add_argument("--metric", default="throughput_tok_s")
    args = ap.parse_args(argv)
    with open(args.cells_csv) as f:
        rows = list(csv.DictReader(f))
    print(grid(rows, args.metric))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
