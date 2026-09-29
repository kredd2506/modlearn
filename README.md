# modlearn

How much better does the Qualcomm Cloud AI 100 Ultra get with a newer software stack?

This repo re-runs the LLM-serving benchmark from Sada et al.,
[*Serving LLMs in HPC Clusters: A Comparative Study of Qualcomm Cloud AI 100 Ultra and NVIDIA Data Center GPUs*](https://arxiv.org/abs/2507.00418)
(arXiv:2507.00418), with [Modular MAX](https://max.modular.com/) as the serving engine alongside vLLM, and uses
[Mojo](https://mojolang.org/) for custom kernels. The paper measured Qualcomm's 2025 stack (vLLM on both sides) and
found competitive energy efficiency, a per-token lead on 3 of 12 models, and much finer hardware allocation. Qualcomm
has since acquired Modular, and MAX now runs on the same accelerator. This project measures what that changes.

**Blog:** [kredd2506.github.io/modlearn](https://kredd2506.github.io/modlearn) has the write-ups, starting with
[why this paper](https://kredd2506.github.io/modlearn/2026/09/28/why-this-paper.html) and the
[pilot](https://kredd2506.github.io/modlearn/2026/09/28/benchmarking-before-benchmarking.html).

## Hypotheses (stated before any runs)

| | Hypothesis | Fails if |
|---|---|---|
| **H1** | On Qualcomm, MAX substantially raises throughput over the paper's numbers at similar power | Throughput stays in the paper's range |
| **H2** | On the A100, MAX matches or beats vLLM in throughput and tok/s/W | MAX is slower or less efficient in most cells |
| **H3** | A right-sized A100 (1 GPU for ≤8B models) raises the A100's tok/s/W | The A100's efficiency doesn't improve |
| **H4** | One Mojo kernel runs unchanged on Apple, NVIDIA and Qualcomm silicon | It needs per-chip rewrites |

Every result is reported, whichever way it goes, and every test setting is published, not a selection.

## Status

| Stage | Status |
|---|---|
| Benchmark harness (client, power sampling, resumable runner, tests) | ✅ Done |
| Pilot: MAX 26.6 on an Apple M4 Pro GPU, all 30 settings × 3 repeats | ✅ Done |
| Custom Mojo kernel (H4) | In progress |
| A100: vLLM vs MAX (H2, H3) | Not started |
| Qualcomm Cloud AI 100 Ultra on MAX (H1) | Looking for hardware access |

The pilot (SmolLM2-135M, bf16) reached 328 tok/s at 8.9 W (35.6 tok/s/W) in the paper's headline setting
(200 tokens × 4 parallel requests), and up to 1,990 tok/s at 32 parallel requests. Along the way it caught a
power-sampling bias that overstated efficiency by up to 12× on short runs, and a GPU wake-up delay in the latency
numbers. Our client matches `max benchmark` within 1–3%. The pilot is a harness check, not a result about any
datacenter chip.

## Protocol

Kept identical to the paper so the numbers are comparable:

- Output tokens {10, 50, 100, 200, 1024, 2048} × concurrent requests {1, 2, 4, 16, 32}
- 20 fixed prompts ([`bench/prompts.jsonl`](bench/prompts.jsonl)), temperature 0, max model length 8192
- Client: Python `requests` + `ThreadPoolExecutor` against the OpenAI-compatible endpoint, the same for every engine
- Throughput = total generated tokens ÷ the longest request's completion time
- Efficiency = tok/s ÷ average watts, also reported as joules per token

Where we differ from the paper, and why:

- **bf16 on every engine.** The paper used fp16. MAX offers bf16, not fp16, for most architectures.
- **Power windows of at least 5 s.** 1 s samples over sub-second runs understate power (found in the pilot).
- **Idle power with the model loaded** is recorded for every run.
- **3 repeats per setting**, and time to first token and inter-token latency are reported as extras, kept separate
  from the paper-comparable metrics.

## Repo layout

| Path | What's there |
|---|---|
| [`bench/`](bench/) | Client, power samplers (NVML on NVIDIA, `powermetrics` on Apple; `qaic-util` still to come), runner, summarizer |
| [`configs/`](configs/) | One TOML per run: server URL, model, engine, hardware, power source |
| [`results/`](results/) | Summarized per-setting CSVs and run metadata (tracked). Blog numbers come from these. See [`results/README.md`](results/README.md) |
| [`results/paper/`](results/paper/) | The paper's Table 2, transcribed |
| [`analysis/`](analysis/) | Scripts that turn `results/` into the blog's figures |
| [`tests/`](tests/) | Harness tests |
| [`blog/`](blog/) | The Jekyll blog, deployed to GitHub Pages on push to `main` |

## Reproduce the pilot

Needs [pixi](https://pixi.sh). The environment (MAX 26.6, Mojo 1.1) is locked for macOS on Apple silicon and
Linux x86-64.

```sh
pixi install
pixi run test

# Terminal 1: serve the model on the Apple GPU
pixi run max serve --model HuggingFaceTB/SmolLM2-135M-Instruct --quantization-encoding bfloat16 \
  --device-memory-utilization 0.5 --max-length 8192

# Terminal 2: sample GPU power (needs sudo)
sudo powermetrics --samplers gpu_power -i 1000 -f plist -o results/raw/powermetrics.plist

# Terminal 3: run the 30 settings, then print a paper-style grid
pixi run bench configs/pilot-mac-smollm2-135m-minwin5.toml
pixi run summarize results/pilot-mac-smollm2-135m-minwin5-cells.csv --metric throughput_tok_s
```

Runs resume: settings already in the output CSV are skipped. To run without power data (no sudo), use
[`configs/smoke-mac-nopower.toml`](configs/smoke-mac-nopower.toml).

On macOS, the Apple GPU needs full Xcode and the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`).
Without them, MAX fails with "Metal Compiler failed to compile metallib". The CPU works without either:
`--devices cpu --quantization-encoding float32`.

## Credits

The benchmark design, test grid and baseline numbers are from Sada et al. (UC San Diego), who did the first
independent measurement of this accelerator with power, on a shared production cluster. The blog uses the
[White Paper](https://github.com/vinitkumar/white-paper) Jekyll theme (MIT, see `blog/THEME-LICENSE`).
