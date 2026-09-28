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
