---
layout: post
title: "Was it the chip or the software? Re-running the Qualcomm Cloud AI 100 vs A100 study on MAX"
comments: true
categories: [ml-systems, benchmarking]
tags: [mojo, max, modular, qualcomm, llm-inference]
excerpt: "A UCSD study found Qualcomm's Cloud AI 100 Ultra drew 10–35× less power than A100s for LLM serving, yet was more efficient per token on only 3 of 12 models. Both sides ran vLLM. We re-run it with Modular's MAX to ask whether that was the chip or the software."
---

<!-- Draft (Jekyll doesn't build _drafts). To publish: move to blog/_posts/YYYY-MM-DD-chip-or-software-max-qualcomm.md,
     add `date:` to the front matter, and push to main. The Pages workflow deploys it.
     Each hypothesis section reports yes / no / unanswered from results/. -->

## TL;DR
_TBD once results are in._

## The paper, and what its numbers actually say
- Sada et al. ([arXiv:2507.00418](https://arxiv.org/abs/2507.00418)) served 12 LLMs with vLLM on the Qualcomm Cloud
  AI 100 Ultra and on 4×/8× A100.
- The headline is 10–35× lower power. Per token, though, Qualcomm is more efficient on 3 of 12 models, and its
  throughput is ~6–25 tok/s for most models.
- Since then, Qualcomm acquired Modular, and [ModCon 2026](https://www.modular.com/blog/modcon-qualcomm) showed MAX
  running on Cloud AI 100 Ultra.

## Hypotheses (stated before running anything)
- **H1:** on Qualcomm, MAX substantially outperforms the paper's vLLM numbers
- **H2:** on the A100, MAX ≥ vLLM in throughput and tok/s/W
- **H3:** a fair A100 configuration (1 GPU for ≤8B models) shrinks Qualcomm's apparent advantage
- **H4:** one Mojo kernel runs unchanged on Apple, NVIDIA and Qualcomm silicon

## Method: what we kept, what we changed
## Pilot: the benchmark code against Modular Cloud
## H2 + H3: A100, vLLM vs MAX
## H1: Qualcomm Cloud AI 100 Ultra on MAX
## H4: one Mojo kernel, several chips
## Scoreboard: the paper's efficiency ratios, recomputed
## What I learned about Mojo and MAX
## Limitations
## Reproduce it
