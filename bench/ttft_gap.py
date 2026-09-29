"""TTFT as a function of idle time before the request.

Sends single requests separated by controlled idle gaps (gap order shuffled per round) and records TTFT,
to test whether a slow first token comes from the server going idle (e.g. GPU power-gating) rather than
from the prompt or from batching.

    pixi run python -m bench.ttft_gap --label gpu --out results/ttft-gap-mac.csv
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from bench.client import stream_chat
from bench.run import REPO, append_rows, load_prompts

GAPS_S = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--label", required=True, help="e.g. gpu or cpu; stored in each row")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    pid, text = load_prompts(REPO / "bench/prompts.jsonl")[0]  # fixed prompt: only the gap varies
    rng = random.Random(0)
    for _ in range(3):  # warm up
        stream_chat(args.base_url, args.model, pid, text, args.tokens, ignore_eos=True, batch_t0=time.perf_counter())

    for rnd in range(args.rounds):
        gaps = GAPS_S[:]
        rng.shuffle(gaps)
        rows = []
        for gap in gaps:
            # Keep the server busy right before the gap, so the gap is the only idle time.
            stream_chat(args.base_url, args.model, pid, text, args.tokens, ignore_eos=True, batch_t0=time.perf_counter())
            time.sleep(gap)
            r = stream_chat(args.base_url, args.model, pid, text, args.tokens, ignore_eos=True, batch_t0=time.perf_counter())
            rows.append({"label": args.label, "round": rnd, "gap_s": gap, "ttft_s": r.ttft_s, "latency_s": r.latency_s,
                         "itl_s": r.itl_s, "error": r.error})
        append_rows(args.out, rows)
        print(f"round {rnd}: " + "  ".join(f"{x['gap_s']}s→{x['ttft_s'] * 1000:.0f}ms" for x in sorted(rows, key=lambda x: x["gap_s"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
