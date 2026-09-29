# Results

One `*-cells.csv` per run: one row per (repeat, output_tokens, concurrency) cell, written by `bench/run.py`.
`*-meta.json` holds the run config, git commit, platform, and idle power with the model loaded.
Per-request rows live in `results/raw/` (not tracked).

Print a paper-style grid with `pixi run summarize results/<run>-cells.csv --metric <column>`.

## Known caveats

- **pilot-mac-smollm2-135m** (M1 pilot, MAX 26.6 on the M4 Pro GPU, harness `8a574dc`): the power figures for
  short cells aren't reliable. At 10 tokens a batch lasts ~0.2 s, but powermetrics samples cover 1 s, and the
  Apple GPU power-gates to ~0 W when idle. Each sample therefore averages the burst with idle time, which
  **understates power and overstates tok/s/W** for cells well under a few seconds. Throughput, TTFT, and ITL
  are unaffected. The same limitation applies to the paper's 1 s `nvidia-smi` sampling. It's fixed in later
  runs by repeating short batches until the measurement window is long enough.
- **pilot-mac-smollm2-135m-minwin5** (same setup, with `min_window_s = 5`): each cell repeats its batch until
  the power window lasts at least 5 s (≥5 samples). Short-cell tok/s/W fell by up to 12× (10 tok × 1: 201 → 16),
  while 2048-token cells, which were already long enough, match the first run within ~1%. The tok/s/W CV across
  repeats fell from a median of 9.5% (worst 141%) to 5.5% (worst 29%). The cells CSV was written by the code
  committed alongside it; the recorded git state is `7e1ef71-dirty`, where the diff is this commit's harness change.
  **Use this run, not the first one, for power.**

## Experiment: TTFT vs idle time (`ttft-gap-mac.csv`, from `bench/ttft_gap.py`)

The pilot's TTFT fell into two groups (~65 ms and ~250 ms). In the logged requests, the slow group was
almost entirely the **first batch of each cell** (89% slow at concurrency 1), and was unrelated to prompt or
prompt length. A controlled test sent single requests after idle gaps of 0–5 s, 8 rounds each:

| Server | 0 s | 0.25 s | 1 s | 2 s | 5 s |
|---|---:|---:|---:|---:|---:|
| MAX on M4 Pro **GPU** (bf16) | ~66 ms | ~90 ms | ~100 ms | **~260 ms** | **~260 ms** |
| MAX on **CPU** (fp32, control) | ~29 ms | ~30 ms | ~50 ms | ~50 ms | ~50 ms |

After 1–2 s of idle the **GPU** pays about 170 ms extra on the next request. The CPU shows no such step, which
points to the Apple GPU waking from power-gating (consistent with the ~0 W idle power). A small ~20 ms idle
penalty appears on both, from MAX scheduling or CPU power states. The runner now sends an unmeasured
`prewarm` request just before each cell (the paper measures steady state). In a check with 2.5 s idle between
cells, first-batch TTFT fell from a median of 217 ms to 67 ms. The two pilot runs above predate this; with
`min_window_s = 5` only the first batch of each cell is affected.

## Cross-check: our client vs `max benchmark` (`crosscheck/`)

The same MAX server (SmolLM2-135M, bf16, M4 Pro GPU), 200-token outputs. `max benchmark` used
`--dataset-name random --random-input-len 50 --random-output-len 200`. Its random dataset does **not** force
the output length (9–242 tokens), so its throughput is computed here as total output tokens / duration.

| Concurrency | Metric | Our client (minwin5 run, median) | `max benchmark` |
|---|---|---:|---:|
| 4 | throughput | 328 tok/s | 324 tok/s (6,161 tok / 19.0 s) |
| 4 | TTFT p50 / ITL mean | 71 ms / 11.5 ms | 72 ms / 11.9 ms |
| 32 | throughput | 1,990 tok/s | 1,968 tok/s (60,947 tok / 31.0 s) |
| 32 | TTFT p50 / ITL mean | 404 ms / 14.2 ms | 325 ms / 13.9 ms |

Throughput and ITL agree within 1–3%. TTFT at 32 differs by design: our client releases all 32 requests at
once, as the paper did, so they queue for prefill together. `max benchmark` keeps 32 requests in flight and
their arrivals stagger. Note that `max benchmark`'s per-request "Output throughput" row is **not** aggregate
server throughput.
