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
