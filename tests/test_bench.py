import csv
import datetime as dt
import json
import plistlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bench import run as runner
from bench.client import run_batch, stream_chat, summarize_batch
from bench.power import PowermetricsFileSampler, integrate, parse_powermetrics_plist

MODEL = "mock/model"
NATURAL_STOP = 5  # tokens the mock emits before "EOS" when ignore_eos is off
TOKEN_DELAY_S = 0.002


class MockOpenAI(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": MODEL}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = req["max_tokens"] if req.get("ignore_eos") else min(req["max_tokens"], NATURAL_STOP)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        for i in range(n):
            time.sleep(TOKEN_DELAY_S)
            choice = {"delta": {"content": f"t{i} "}, "index": 0}
            if i == n - 1:
                choice["finish_reason"] = "length" if n == req["max_tokens"] else "stop"
            send({"choices": [choice]})
        send({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": n}})
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_ignore_eos_forces_exact_length(server):
    r = stream_chat(server, MODEL, "p", "hi", 40, ignore_eos=True, batch_t0=time.perf_counter())
    assert (r.completion_tokens, r.finish_reason, r.error) == (40, "length", "")
    assert r.ttft_s is not None and 0 < r.ttft_s < r.latency_s
    assert r.itl_s == pytest.approx((r.latency_s - r.ttft_s) / 39)


def test_without_ignore_eos_model_can_stop_early(server):
    r = stream_chat(server, MODEL, "p", "hi", 40, ignore_eos=False, batch_t0=time.perf_counter())
    assert (r.completion_tokens, r.finish_reason) == (NATURAL_STOP, "stop")


def test_batch_throughput_is_tokens_over_max_completion_time(server):
    results = run_batch(server, MODEL, [("a", "x"), ("b", "y"), ("c", "z")], 30, ignore_eos=True)
    s = summarize_batch(results)
    assert s["generated_tokens"] == 90 and s["n_errors"] == 0 and s["n_finish_length"] == 3
    assert s["wall_s"] == max(r.completion_s for r in results)
    assert s["throughput_tok_s"] == pytest.approx(90 / s["wall_s"])


def test_connection_error_is_recorded_not_raised():
    r = stream_chat("http://127.0.0.1:9/v1", MODEL, "p", "hi", 5, ignore_eos=True, batch_t0=time.perf_counter(), timeout=2)
    assert r.error and r.completion_tokens == 0


def _write_config(tmp_path, base_url):
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        f"""[run]
name = "mocktest"
base_url = "{base_url}"
model = "{MODEL}"
engine = "mock"
engine_version = "0"
hardware = "test"
devices = 1
dtype = "none"
output_tokens = [4, 8]
concurrency = [1, 3]
repeats = 2
idle_seconds = 0
results_dir = "{tmp_path / 'results'}"
"""
    )
    return cfg


def test_runner_writes_cells_and_resumes(server, tmp_path):
    cfg = _write_config(tmp_path, server)
    cells_csv = tmp_path / "results" / "mocktest-cells.csv"

    runner.main([str(cfg), "--max-cells", "3"])
    with open(cells_csv) as f:
        first = list(csv.DictReader(f))
    assert len(first) == 3

    runner.main([str(cfg)])  # resumes: only the remaining 5 of 2x2x2 cells
    with open(cells_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    assert len({(r["repeat"], r["output_tokens"], r["concurrency"]) for r in rows}) == 8
    for r in rows:
        assert int(r["generated_tokens"]) == int(r["output_tokens"]) * int(r["concurrency"])
        assert r["n_errors"] == "0"

    requests_csv = tmp_path / "results" / "raw" / "mocktest" / "requests.csv"
    with open(requests_csv) as f:
        assert len(list(csv.DictReader(f))) == 2 * 2 * (1 + 3)  # repeats x output lengths x sum(concurrency)

    runner.main([str(cfg)])  # nothing left to do
    with open(cells_csv) as f:
        assert len(list(csv.DictReader(f))) == 8


def _plist_stream(samples):
    """samples: list of (end_epoch, elapsed_s, gpu_mJ)"""
    docs = []
    for end, elapsed, mj in samples:
        docs.append(
            plistlib.dumps(
                {
                    "timestamp": dt.datetime.fromtimestamp(end, dt.timezone.utc).replace(tzinfo=None),
                    "elapsed_ns": int(elapsed * 1e9),
                    "gpu": {"gpu_energy": mj},
                }
            )
        )
    return b"\0".join(docs) + b"\0"


def test_powermetrics_parse_and_overlap_weighting(tmp_path):
    # Two 1 s samples: 10 W over [100, 101], 20 W over [101, 102].
    raw = _plist_stream([(101.0, 1.0, 10_000), (102.0, 1.0, 20_000)])
    intervals = parse_powermetrics_plist(raw)
    assert [(i.start, i.end, i.power_w) for i in intervals] == [(100.0, 101.0, 10.0), (101.0, 102.0, 20.0)]

    # A window straddling both samples equally averages to 15 W.
    r = integrate(intervals, 100.5, 101.5)
    assert r.avg_w == pytest.approx(15.0) and r.energy_j == pytest.approx(15.0) and r.n_samples == 2
    # A window shorter than one sample still gets that sample's power.
    assert integrate(intervals, 101.2, 101.4).avg_w == pytest.approx(20.0)
    # No overlap -> no reading.
    assert integrate(intervals, 200, 201).avg_w is None

    f = tmp_path / "pm.plist"
    f.write_bytes(raw + b"<?xml half-written")  # trailing partial doc is ignored
    s = PowermetricsFileSampler(f, wait_s=0)
    s.begin()
    assert s.end(100.0, 102.0).avg_w == pytest.approx(15.0)


def test_powermetrics_sampler_requires_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="sudo powermetrics"):
        PowermetricsFileSampler(tmp_path / "missing.plist").begin()
