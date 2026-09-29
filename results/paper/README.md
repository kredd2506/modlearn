# The paper's published numbers

`sada2025-table2.csv` is transcribed from Table 2 of Sada et al., *Serving LLMs in HPC Clusters: A Comparative Study
of Qualcomm Cloud AI 100 Ultra and NVIDIA Data Center GPUs*, arXiv:2507.00418v3 (Oct 2025): "Performance and Power
Consumption Using Actual Required Configurations (200 tokens, 4 parallel requests)". Model names are kept exactly
as printed. The paper lists both `deepseek-qwen-7b` and `DeepSeek-Qwen-7B` with different numbers and doesn't
explain the difference.

Setup details from the paper that matter when reading these numbers:
- Both sides use vLLM: stock CUDA vLLM on A100 SXM4-80GB, and Qualcomm's vLLM fork on Cloud AI 100 Ultra (SDK 1.20).
- On Qualcomm, each model is compiled ahead of time to a QPC for a fixed batch size (1, 2, or 4), context length 8192, and
  device group. On A100, vLLM's `--max-num-seqs` is set to that same batch size.
- The A100 runs fp16. Qualcomm uses "FP16 or INT8 quantization where applicable"; the paper doesn't say which models used which.
- Power: `nvidia-smi` at 1 s and `qaic-util -q`, accelerator power only.
- The paper reports only this one cell of its 30-cell matrix.
