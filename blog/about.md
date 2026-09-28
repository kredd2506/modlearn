---
layout: default
title: About
---
## About {{ site.name }}

<img class="user-avatar" src="{{ site.owner.avatar }}">

This site documents a re-run of an LLM-serving benchmark, *Serving LLMs in HPC Clusters:
Qualcomm Cloud AI 100 Ultra vs NVIDIA A100* ([arXiv:2507.00418](https://arxiv.org/abs/2507.00418)).
The re-run uses **Modular's MAX** serving engine and **Mojo** kernels in place of vLLM. It keeps the paper's test settings,
metrics, and power methodology so the numbers are directly comparable. It publishes every test setting,
not a selected few, along with the raw data.

It is also a learning log: I'm learning Mojo and MAX by doing this.

The benchmark code, results, and analysis are open:
[github.com/kredd2506/modlearn](https://github.com/kredd2506/modlearn).

— Manish Reddy

<div class="pagination">
  {% if site.owner.email %}
    <a href="mailto:{{ site.owner.email }}" class="social-media-icons"><i class="fa fa-2x fa-envelope-square" aria-hidden="true"></i></a>
  {% endif %}
  {% if site.owner.github %}
    <a href="{{ site.owner.github }}" class="social-media-icons"><i class="fa-brands fa-2x fa-square-github" aria-hidden="true"></i></a>
  {% endif %}
</div>
