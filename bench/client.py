"""Load generator for OpenAI-compatible chat servers (MAX, vLLM, ...).

Mirrors the paper's client (Sada et al., §3.2): `requests` + `ThreadPoolExecutor`, one thread per
concurrent request, temperature 0. Throughput is total generated tokens divided by the *maximum*
request completion time in the batch. Streaming is used so we also get TTFT and inter-token latency.
"""

from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests


@dataclass
class RequestResult:
    prompt_id: str
    t_start: float  # epoch seconds, used to align with power samples
    ttft_s: float | None  # dispatch -> first content token
    latency_s: float  # dispatch -> end of stream
    completion_s: float  # batch start -> end of stream (the paper's "completion time")
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    http_status: int
    error: str = ""

    @property
    def itl_s(self) -> float | None:
        """Mean inter-token latency after the first token."""
        if self.ttft_s is None or self.completion_tokens < 2:
            return None
        return (self.latency_s - self.ttft_s) / (self.completion_tokens - 1)


def stream_chat(
    base_url: str,
    model: str,
    prompt_id: str,
    prompt: str,
    max_tokens: int,
    *,
    ignore_eos: bool,
    batch_t0: float,
    timeout: float = 900.0,
) -> RequestResult:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        body["ignore_eos"] = True

    t_start = time.time()
    t0 = time.perf_counter()
    ttft = None
    usage: dict = {}
    finish = None
    status = 0
    error = ""
    try:
        with requests.post(f"{base_url}/chat/completions", json=body, stream=True, timeout=timeout) as r:
            status = r.status_code
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith(b"data: "):
                    continue
                data = line[len(b"data: ") :]
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                for choice in chunk.get("choices") or []:
                    if ttft is None and (choice.get("delta") or {}).get("content"):
                        ttft = time.perf_counter() - t0
                    finish = choice.get("finish_reason") or finish
                if chunk.get("usage"):
                    usage = chunk["usage"]
    except Exception as e:  # recorded per request; a failed request must not kill the sweep
        error = f"{type(e).__name__}: {e}"
    t1 = time.perf_counter()

    if not error and not usage:
        error = "no usage in stream"
    return RequestResult(
        prompt_id=prompt_id,
        t_start=t_start,
        ttft_s=ttft,
        latency_s=t1 - t0,
        completion_s=t1 - batch_t0,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        finish_reason=finish,
        http_status=status,
        error=error,
    )


def run_batch(
    base_url: str,
    model: str,
    prompts: list[tuple[str, str]],
    max_tokens: int,
    *,
    ignore_eos: bool,
    timeout: float = 900.0,
) -> list[RequestResult]:
    """Fire len(prompts) requests at once, one thread each, and wait for all of them."""
    batch_t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [
            pool.submit(
                stream_chat,
                base_url,
                model,
                pid,
                text,
                max_tokens,
                ignore_eos=ignore_eos,
                batch_t0=batch_t0,
                timeout=timeout,
            )
            for pid, text in prompts
        ]
        return [f.result() for f in futures]


def summarize_batch(results: list[RequestResult]) -> dict:
    """Per-cell metrics. `throughput_tok_s` is the paper's definition."""
    ok = [r for r in results if not r.error]
    tokens = sum(r.completion_tokens for r in ok)
    wall = max((r.completion_s for r in results), default=0.0)
    ttfts = sorted(r.ttft_s for r in ok if r.ttft_s is not None)
    itls = [r.itl_s for r in ok if r.itl_s is not None]
    return {
        "n_requests": len(results),
        "n_errors": len(results) - len(ok),
        "n_finish_length": sum(r.finish_reason == "length" for r in ok),
        "generated_tokens": tokens,
        "wall_s": wall,
        "throughput_tok_s": tokens / wall if wall > 0 else 0.0,
        "ttft_p50_s": statistics.median(ttfts) if ttfts else None,
        "ttft_max_s": ttfts[-1] if ttfts else None,
        "itl_mean_s": statistics.fmean(itls) if itls else None,
    }
