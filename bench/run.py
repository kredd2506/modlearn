"""Sweep the paper's test matrix against a running OpenAI-compatible server.

    pixi run bench configs/pilot-mac-smollm2-135m.toml

Writes per-request rows to results/raw/<name>/requests.csv (ignored) and one row per cell to
results/<name>-cells.csv (tracked). Cells already in the cells CSV are skipped, so an interrupted
sweep resumes where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import platform
import random
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import requests

from bench.client import run_batch, summarize_batches
from bench.power import make_sampler

REPO = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "output_tokens": [10, 50, 100, 200, 1024, 2048],
    "concurrency": [1, 2, 4, 16, 32],
    "repeats": 3,
    "ignore_eos": True,
    "seed": 0,
    "power": "none",
    "idle_seconds": 10,
    # Repeat a cell's batch until the measurement window lasts at least this long, so 1 s power samples
    # aren't averaged with idle time on short cells (a 10-token batch can finish in 0.2 s).
    "min_window_s": 5.0,
    # Send one tiny unmeasured request right before each cell. The runner idles between cells (waiting for
    # power samples), and the M4 Pro GPU pays ~160 ms to wake after >1-2 s idle (results/ttft-gap-mac.csv);
    # the paper reports steady-state serving, so we measure warm.
    "prewarm": True,
    "prompts": "bench/prompts.jsonl",
    "results_dir": "results",
    "request_timeout_s": 900,
}


def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        cfg = {**DEFAULTS, **tomllib.load(f)["run"]}
    for key in ("name", "base_url", "model", "engine", "engine_version", "hardware", "devices", "dtype"):
        if key not in cfg:
            raise SystemExit(f"{path}: [run] is missing '{key}'")
    return cfg


def load_prompts(path: Path) -> list[tuple[str, str]]:
    with open(path) as f:
        return [(p["id"], p["text"]) for p in map(json.loads, f) if p]


def pick_prompts(prompts: list[tuple[str, str]], n: int, offset: int) -> list[tuple[str, str]]:
    """n prompts, cycling through the fixed set from a per-cell offset."""
    return [prompts[(offset + i) % len(prompts)] for i in range(n)]


def plan_cells(cfg: dict) -> list[tuple[int, int, int]]:
    """(repeat, output_tokens, concurrency), shuffled within each repeat so drift doesn't align with a cell."""
    cells = []
    for rep in range(cfg["repeats"]):
        grid = [(rep, t, c) for t in cfg["output_tokens"] for c in cfg["concurrency"]]
        random.Random(cfg["seed"] + rep).shuffle(grid)
        cells += grid
    return cells


def done_cells(path: Path) -> set[tuple[int, int, int]]:
    if not path.exists():
        return set()
    with open(path) as f:
        return {(int(r["repeat"]), int(r["output_tokens"]), int(r["concurrency"])) for r in csv.DictReader(f)}


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def git_state() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=REPO) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def wait_for_server(base_url: str, model: str, timeout_s: float = 600) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            ids = [m["id"] for m in requests.get(f"{base_url}/models", timeout=5).json()["data"]]
            if model in ids:
                return
            raise SystemExit(f"server at {base_url} serves {ids}, not {model}")
        except requests.RequestException:
            time.sleep(2)
    raise SystemExit(f"server at {base_url} not ready after {timeout_s}s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--max-cells", type=int, default=None, help="stop after N new cells (smoke tests)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    prompts = load_prompts(REPO / cfg["prompts"])
    results_dir = REPO / cfg["results_dir"]
    raw_dir = results_dir / "raw" / cfg["name"]
    cells_csv = results_dir / f"{cfg['name']}-cells.csv"
    requests_csv = raw_dir / "requests.csv"
    raw_dir.mkdir(parents=True, exist_ok=True)

    wait_for_server(cfg["base_url"], cfg["model"])
    sampler = make_sampler(cfg["power"], **cfg)
    ident = {k: cfg[k] for k in ("engine", "engine_version", "hardware", "devices", "dtype", "model")}

    meta_path = raw_dir / "meta.json"
    if not meta_path.exists():
        # Warm up (first request can include compilation), then measure idle power with the model loaded.
        run_batch(cfg["base_url"], cfg["model"], prompts[:1], 16, ignore_eos=True)
        time.sleep(2)
        sampler.begin()
        t0 = time.time()
        time.sleep(cfg["idle_seconds"])
        idle = sampler.end(t0, time.time())
        meta = {
            "config": cfg,
            "git": git_state(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "idle_power": dataclasses.asdict(idle),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"idle power: {idle.avg_w} W ({idle.n_samples} samples)")

    done = done_cells(cells_csv)
    todo = [c for c in plan_cells(cfg) if c not in done]
    print(f"{cfg['name']}: {len(done)} cells done, {len(todo)} to go")

    for i, (rep, n_tok, conc) in enumerate(todo[: args.max_cells]):
        offset = (rep * 7 + cfg["output_tokens"].index(n_tok) * 3 + conc) % len(prompts)
        key = {"repeat": rep, "output_tokens": n_tok, "concurrency": conc}
        batches = []
        if cfg["prewarm"]:
            run_batch(cfg["base_url"], cfg["model"], prompts[:1], 4, ignore_eos=True)
        sampler.begin()
        t0 = time.time()
        while not batches or time.time() - t0 < cfg["min_window_s"]:
            batch = pick_prompts(prompts, conc, offset + len(batches) * conc)
            results = run_batch(
                cfg["base_url"], cfg["model"], batch, n_tok,
                ignore_eos=cfg["ignore_eos"], timeout=cfg["request_timeout_s"],
            )
            append_rows(
                requests_csv,
                [{**key, "batch": len(batches), **ident, **dataclasses.asdict(r), "itl_s": r.itl_s} for r in results],
            )
            batches.append(results)
        t1 = time.time()
        power = sampler.end(t0, t1)

        s = summarize_batches(batches)
        tput = s["throughput_tok_s"]
        tokens = s["generated_tokens"]
        cell = {
            **key, **ident, **s,
            "t_start": t0, "t_end": t1,
            "min_window_s": cfg["min_window_s"],
            "avg_power_w": power.avg_w,
            "energy_j": power.energy_j,
            "power_samples": power.n_samples,
            "power_source": power.source,
            # tokens per joule == tok/s per W, computed over the whole window (including the small gaps
            # between back-to-back batches), so it stays consistent with energy_j.
            "tok_s_per_w": tokens / power.energy_j if power.energy_j else None,
            "j_per_token": power.energy_j / tokens if power.energy_j and tokens else None,
            "git": git_state(),
        }
        append_rows(cells_csv, [cell])
        w = f"{power.avg_w:.1f} W" if power.avg_w else "no power"
        print(
            f"[{i + 1}/{len(todo)}] rep={rep} tokens={n_tok:>4} conc={conc:>2}: "
            f"{tput:8.1f} tok/s  {w}  batches={s['n_batches']}  errors={s['n_errors']}",
            flush=True,
        )
    sampler.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
