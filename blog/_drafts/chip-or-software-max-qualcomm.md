---
layout: post
title: "Was it the chip or the software? Re-running the Qualcomm Cloud AI 100 vs A100 study on MAX"
comments: true
categories: [ml-systems, benchmarking]
tags: [mojo, max, modular, qualcomm, llm-inference]
excerpt: "A UCSD study found Qualcomm's Cloud AI 100 Ultra competitive with A100s on energy efficiency for LLM serving, ahead per token on 3 of 12 models, on 2025 software. We re-run it with Modular's MAX to measure how much a newer stack adds."
---

<!-- Draft (Jekyll doesn't build _drafts). To publish: move to blog/_posts/YYYY-MM-DD-chip-or-software-max-qualcomm.md,
     add `date:` to the front matter, and push to main. The Pages workflow deploys it.
     Each hypothesis section reports yes / no / unanswered from results/. -->

## TL;DR
_TBD once results are in._

## The paper, and what its numbers actually say
- Sada et al. ([arXiv:2507.00418](https://arxiv.org/abs/2507.00418)) served 12 LLMs with vLLM on the Qualcomm Cloud
  AI 100 Ultra and on 4×/8× A100.
- It found competitive energy efficiency, a per-token lead on 3 of 12 models, and much finer hardware allocation.
  Most models served at ~6–25 tok/s, and similar models needed very different device counts: headroom for software.
- Since then, Qualcomm acquired Modular, and [ModCon 2026](https://www.modular.com/blog/modcon-qualcomm) showed MAX
  running on Cloud AI 100 Ultra.

## Hypotheses (stated before running anything)
- **H1:** on Qualcomm, MAX substantially raises throughput over the paper's numbers at similar power
- **H2:** on the A100, MAX ≥ vLLM in throughput and tok/s/W
- **H3:** a right-sized A100 (1 GPU for ≤8B models) raises the A100's tokens per watt
- **H4:** one Mojo kernel runs unchanged on Apple, NVIDIA and Qualcomm silicon

## Method: what we kept, what we changed
## Pilot
<!-- Covered in its own post: blog/_posts/2026-09-28-benchmarking-before-benchmarking.md. Link to it here
     ({{ site.baseurl }}/2026/09/28/benchmarking-before-benchmarking.html) instead of repeating it. -->
## H2 + H3: A100, vLLM vs MAX
## H1: Qualcomm Cloud AI 100 Ultra on MAX
## H4: one Mojo kernel, several chips
## Scoreboard: the paper's efficiency ratios, recomputed
## What I learned about Mojo and MAX
## Limitations
## Reproduce it
