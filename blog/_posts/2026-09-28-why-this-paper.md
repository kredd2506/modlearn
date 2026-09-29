---
layout: post
title: "Chip or software? Why I'm re-running the Qualcomm Cloud AI 100 vs A100 study on Modular MAX"
date: 2026-09-28
tags: [qualcomm, nvidia, max, mojo, modular, llm-inference, energy-efficiency]
excerpt: "A 2025 UCSD study found Qualcomm's Cloud AI 100 Ultra drew 7–35× less power than A100s for LLM serving, yet was more efficient per token on only 3 of 12 models. Both sides ran vLLM. Qualcomm now owns Modular, so here's what the paper shows, what it leaves open, and why it's the right yardstick for MAX and Mojo."
---

In July 2025, a team at UC San Diego published
[*Serving LLMs in HPC Clusters: A Comparative Study of Qualcomm Cloud AI 100 Ultra and NVIDIA Data Center GPUs*](https://arxiv.org/abs/2507.00418)
(Sada et al.; I'm reading v3, October 2025). A year later, Qualcomm
[completed its acquisition of Modular](https://www.prnewswire.com/news-releases/qualcomm-completes-acquisition-of-modular-302837286.html),
the company behind the MAX inference framework and the Mojo language. A month after that, Modular showed
[MAX running on the same Qualcomm accelerator](https://www.modular.com/blog/modcon-qualcomm).

Those three events set up a question the paper couldn't ask: **were its results about the chip, or about the
software running on it?** This post explains what the paper did, why I think it matters, what I read in its numbers,
and why it's the right yardstick for Modular's claims. The [first results post]({{ site.baseurl }}/2026/09/28/benchmarking-before-benchmarking.html)
covers how I built and validated the benchmark harness.

## The paper in one paragraph

The authors serve 12 open models, from GPT-2 (124M parameters) to three 70B models, on two kinds of hardware in the
[National Research Platform](https://nationalresearchplatform.org/) (NRP), a shared Kubernetes cluster used by more than
300 research groups. One side is a node with 8 **Qualcomm Cloud AI 100 Ultra** cards at the San Diego Supercomputer
Center. Each card contains four independent chips ("devices"). The other side is nodes with 4 or 8 **NVIDIA A100 80 GB** GPUs.
Both sides serve through **vLLM**: stock vLLM on the A100s, and a Qualcomm-provided vLLM fork on Cloud AI 100, where each
model is first exported to ONNX and compiled ahead of time into a Qualcomm Program Container (QPC). The benchmark
sweeps output length (10 to 2,048 tokens) × concurrent requests (1 to 32), and it measures throughput and accelerator
power, sampled once per second.

## Why this paper matters

**It's an independent measurement of non-NVIDIA inference hardware, with power, in production.** Vendor
benchmarks are easy to find. Third-party numbers that include watts, taken on a real multi-tenant cluster rather than a
tuned demo box, are rare. The work was funded under an NSF program whose title states the stakes: *Democratizing the
Accelerator Ecosystem for Science and Discovery*.

**It measures the right things for inference at scale.** Serving cost is driven by energy per token and by how
finely you can divide hardware among users, not by peak FLOPS. The paper's strongest finding is about
**granularity**. A single Cloud AI 100 device served granite-3.2-8b at 36 W, and Nemotron-70B ran on one card (four
devices) at 148 W. Compare that with the 4 and 8 A100s the authors used, which drew 1,246 W and 2,983 W. For a
cluster serving hundreds of groups, being able to hand out a 36 W slice instead of a whole 80 GB GPU is an operational
advantage in its own right.

**Its protocol is concrete enough to reproduce.** The paper spells out the test grid, the client (Python
`requests` plus a thread pool), how throughput is computed, and how power is sampled. That's what makes a re-run with a
different software stack a fair comparison rather than a new, unrelated benchmark.

## What the numbers actually say

The headline, repeated in the abstract and the conclusion, is power: Qualcomm uses 10–35× less (from the paper's
Table 2 I get 7–35×). That's true, but power is only half of efficiency. The other half is how much work you get for
that power. The paper reports one cell of its grid in detail (Table 2: 200 output tokens, 4 parallel requests), and
plotting both halves from that table tells a more mixed story:

![For each of 12 models, how many times less power Qualcomm draws (7 to 35×) and how many times less throughput it delivers (12 to 75×). Only granite-3.2-8b, GPT-2 and Nemotron-70B have a power reduction larger than their throughput reduction.]({{ site.baseurl }}/assets/modlearn/paper-power-vs-throughput.png)

When the throughput gap (orange) is wider than the power gap (blue), the A100 gets more tokens per joule. Divide one
by the other and you get the efficiency ratio, which the paper also reports in Table 3:

![Qualcomm tokens per second per watt divided by A100's, per model, on a log scale. Above parity: granite-3.2-8b 2.72×, GPT-2 2.65×, Nemotron-70B 1.16×. Below parity: nine models from 0.77× down to 0.13×.]({{ site.baseurl }}/assets/modlearn/paper-efficiency-ratio.png)

**Per token, Qualcomm is more efficient on 3 of 12 models.** On the other nine, the A100 setup is 1.3× to 8× more
efficient. The absolute throughput on Qualcomm is also low for most models: 6–25 tok/s in total across four
concurrent requests, which is a few tokens per second for each user.

The paper is candid that A100 delivers "higher absolute throughput," and it frames Qualcomm's case around power
budgets and granularity. That's fair. But the efficiency picture is the one a data-center operator pays for, and it's
the one I want to test.

## Reading the fine print

A careful re-read turned up six details that shape how far those numbers generalize. None of them is a criticism of
the effort; a first measurement in a production cluster is hard, and the authors document their choices. They are the
reasons to measure again.

1. **The batch size is fixed at compile time, and the A100 was capped to match.** On Qualcomm, each model is compiled
   ahead of time for a batch size of 1, 2, or 4, and the paper notes that "compilation can take hours." On A100, vLLM's
   `--max-num-seqs` was set "matching the batch size from QPC compilation." So neither system served more than four
   sequences at once, even in the 16- and 32-request tests. That's a sensible choice for a like-for-like comparison. It
   also switches off the thing GPUs are best at, which is batching dozens of requests together, and it explains why
   the paper's headline cell is 4 parallel requests.
2. **Small models ran on 4 A100s.** The authors say some of them "could potentially run on fewer A100 GPUs." An 8B model
   fits on one A100 80 GB, so the A100 power for those rows is roughly four GPUs' worth for one GPU's work.
3. **The precision may differ.** The A100s ran fp16. Qualcomm was compiled with "FP16 or INT8 quantization where
   applicable," and the paper doesn't say which models used which.
4. **One cell of 30 is reported.** The grid covers 6 output lengths × 5 concurrency levels, but the tables show only
   200 tokens × 4 requests. The prompts and code aren't published, and two rows (`deepseek-qwen-7b` and `DeepSeek-Qwen-7B`)
   report different numbers without explanation.
5. **Power is sampled at 1 s, for the accelerator only, and idle isn't measured.** In
   [my pilot]({{ site.baseurl }}/2026/09/28/benchmarking-before-benchmarking.html), 1 s sampling overstated
   efficiency by up to 12× on sub-second batches. That was on an Apple GPU that idles near 0 W, so the effect on an A100
   would be smaller, but the exposure exists for the 10- and 50-token cells. The paper also says Qualcomm's advantage
   grows at idle, but it doesn't report idle power.
6. **Table 3's rack-scale view assumes perfect scaling.** It extrapolates to 32 Qualcomm devices and 8 A100s by assuming
   N copies of a model deliver N× the throughput, which is an upper bound.

## Chip or software? The case for software

Here's the evidence inside the paper itself. The Qualcomm hardware serves **granite-3.2-8b on one device at 25 tok/s
and 36 W**. It serves **DeepSeek-R1-Distill-Llama-8B**, a model of the same size, on **four devices at 24 tok/s and
140 W**. That's four times the silicon and nearly four times the power for the same throughput. It also serves
**GPT-2 at 218 tok/s on one device**. If the chip's raw capability set the limit, four devices would do more than one.
Something between the model and the silicon is doing the limiting: how the model is exported, compiled, partitioned
across devices, and scheduled.

To be fair, the two 8B rows may also differ in precision or compiled batch size, since the paper doesn't say. But
those are software choices too.

Every step in that path is software:
- the ONNX export
- the ahead-of-time QPC compile, fixed to one batch size and context length
- the device-group partitioning
- a vLLM fork maintained separately from upstream vLLM

The paper's own workflow description (sections 2.3 and 2.5) reads like a list of places where performance can be won
or lost.

I can't prove from the outside that the software is the bottleneck. That's exactly why it's worth testing.

## Why this matters for Modular, Mojo, and MAX

On July 29, 2026, Qualcomm completed its acquisition of Modular and said that "Mojo, MAX, and Modular Cloud will
continue as products and brands." Modular's pitch has always been portability without a performance penalty: one
serving stack (MAX), one kernel language (Mojo), many kinds of hardware. At ModCon in August, the Modular team wrote
that "the same code that runs on NVIDIA and AMD GPUs can now target Qualcomm Technologies' silicon." They showed GPT-2
reaching "6.7× the baseline" after about three weeks of optimization, and Gemma 4 31B served across a card's four
devices with tensor parallelism.

One detail matters for comparisons: that 6.7× is measured against a *PyTorch implementation* on Cloud AI 100, not
against the paper's vLLM setup. The two numbers can't be put side by side. That gap is exactly what an independent
re-run can fill.

That's why this paper is the right yardstick:

- **It's the best public "before" picture.** It's an independent, publicly available measurement of Qualcomm's inference
  stack before MAX, with power, on the exact accelerator Modular now targets. If MAX unlocks the hardware, it should
  show up here as more throughput at similar watts, and the 3-of-12 count should move.
- **Portability has to be measured on both sides.** Running MAX on the A100 against vLLM asks whether a portable
  stack costs performance on the incumbent's hardware. Running MAX on Qualcomm against the paper asks whether it
  helps the new hardware. Both halves matter; winning on one while losing on the other isn't portability.
- **The compile-time batch limit is a software question with a measurable answer.** If MAX can serve Cloud AI 100 at
  16 or 32 concurrent requests without a fixed-batch compile, that changes what the chip is good for.
- **"Same code" is testable.** Writing one Mojo kernel and running it unchanged on an Apple GPU, an NVIDIA GPU, and,
  with luck, a Qualcomm device is the most direct check of the portability claim I can do from outside.
- **Independent numbers matter more after an acquisition.** Qualcomm now owns both the chip and the software stack. A
  third-party measurement with published code, data, and every cell of the grid is useful to everyone, Modular
  included.

## What I'm changing, and why

The plan is to keep everything the paper specifies and fix what limits it:

| | Paper | This re-run |
|---|---|---|
| Test grid, client, throughput formula | 6 lengths × 5 concurrency, requests + thread pool | **Same** |
| Engines | vLLM on both sides | **vLLM and MAX** on A100; **MAX** on Qualcomm |
| Batch cap | Both capped at the QPC batch (1, 2, or 4) | **Both**: a paper-faithful capped run *and* each system at its best |
| A100 count for ≤8B models | 4 | **1** (and 4 for a few models, to line up with the paper) |
| Precision | A100 fp16; Qualcomm fp16 or INT8 | **bf16 on every engine**, stated per row (MAX offers bf16, not fp16, for most models) |
| Power | 1 s sampling, accelerator only | 1 s sampling **plus** NVML's energy counter, windows of **≥5 s**, **idle measured** |
| Latency | End-to-end only | Also time to first token and inter-token latency, measured warm |
| Reporting | 1 of 30 cells; prompts and code unpublished | **All cells**, prompts, code, and raw data [on GitHub](https://github.com/kredd2506/modlearn) |

## The hypotheses, stated before the runs

- **H1, Qualcomm:** MAX serves the paper's models on Cloud AI 100 Ultra at substantially higher throughput than the
  paper's vLLM numbers, at similar power. *This fails if the throughput stays in the same range.*
- **H2, A100:** MAX matches or beats vLLM in throughput and tok/s/W on the same GPU. *This fails if MAX is slower
  or less efficient in most cells.*
- **H3, a fair A100:** running ≤8B models on 1 A100 instead of 4 shrinks Qualcomm's apparent efficiency advantage on
  those models. *This fails if Qualcomm's lead survives.*
- **H4, Mojo:** one Mojo kernel runs unchanged on Apple, NVIDIA, and Qualcomm silicon. *This fails if it needs
  per-chip rewrites.*

I'll report the results that go against the story as prominently as the ones that support it. The biggest risk is H1:
I don't yet have access to Qualcomm hardware. If that doesn't change, H1 will be reported as unanswered, and H2–H4 will
still stand on their own.

## What's next

The harness is built and validated. The
[pilot post]({{ site.baseurl }}/2026/09/28/benchmarking-before-benchmarking.html) covers the three measurement
problems it caught on a MacBook before any GPU money was spent. Next come A100 runs, vLLM against MAX, and the search
for Qualcomm access.

The figures in this post come from the paper's Table 2, transcribed to
[`results/paper/sada2025-table2.csv`](https://github.com/kredd2506/modlearn/blob/main/results/paper/sada2025-table2.csv)
and plotted by [`analysis/plot_paper.py`](https://github.com/kredd2506/modlearn/blob/main/analysis/plot_paper.py).
