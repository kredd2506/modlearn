"""Power measurement backends.

Every sampler exposes `begin()` before a batch and `end(t0, t1)` after it, where t0/t1 are the batch's
epoch-second bounds, and returns a PowerReading for that window.

- nvml:          the paper's method (sample board power every 1 s and average) plus the NVML energy counter,
                 which gives exact joules with no sampling gaps.
- powermetrics:  macOS. powermetrics needs root, so the user runs it separately, writing a plist stream:
                   sudo powermetrics --samplers gpu_power -i 1000 -f plist -o results/raw/powermetrics.plist
                 and we integrate the samples that overlap each batch window, weighted by overlap.
- none:          no power data (e.g. hosted endpoints).
"""

from __future__ import annotations

import datetime as dt
import plistlib
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PowerReading:
    avg_w: float | None  # average power over the window
    energy_j: float | None  # energy over the window
    n_samples: int
    source: str


class NullSampler:
    name = "none"

    def begin(self) -> None:
        pass

    def end(self, t0: float, t1: float) -> PowerReading:
        return PowerReading(None, None, 0, self.name)

    def close(self) -> None:
        pass


@dataclass
class _Interval:
    start: float  # epoch s
    end: float  # epoch s
    power_w: float


def parse_powermetrics_plist(raw: bytes) -> list[_Interval]:
    """powermetrics -f plist writes one plist document per sample, separated by NUL bytes."""
    out = []
    for doc in raw.split(b"\0"):
        doc = doc.strip()
        if not doc:
            continue
        try:
            d = plistlib.loads(doc)
        except Exception:
            continue  # last document may still be half-written
        ts = d.get("timestamp")
        elapsed_s = d.get("elapsed_ns", 0) / 1e9
        if not isinstance(ts, dt.datetime) or elapsed_s <= 0:
            continue
        if ts.tzinfo is None:  # plistlib returns naive UTC datetimes
            ts = ts.replace(tzinfo=dt.timezone.utc)
        gpu = d.get("gpu") or {}
        proc = d.get("processor") or {}
        if "gpu_energy" in gpu:  # mJ over the sample interval
            watts = gpu["gpu_energy"] / 1000 / elapsed_s
        elif "gpu_power" in proc:  # mW
            watts = proc["gpu_power"] / 1000
        elif "gpu_energy" in proc:
            watts = proc["gpu_energy"] / 1000 / elapsed_s
        else:
            continue
        end = ts.timestamp()
        out.append(_Interval(end - elapsed_s, end, watts))
    return out


def integrate(intervals: list[_Interval], t0: float, t1: float) -> PowerReading:
    """Energy of the overlap between each sample interval and [t0, t1]."""
    energy = 0.0
    covered = 0.0
    n = 0
    for iv in intervals:
        overlap = min(iv.end, t1) - max(iv.start, t0)
        if overlap > 0:
            energy += iv.power_w * overlap
            covered += overlap
            n += 1
    if covered <= 0:
        return PowerReading(None, None, 0, "powermetrics")
    avg = energy / covered
    return PowerReading(avg, avg * (t1 - t0), n, "powermetrics")


class PowermetricsFileSampler:
    name = "powermetrics"

    def __init__(self, path: str | Path, wait_s: float = 3.0):
        self.path = Path(path)
        self.wait_s = wait_s  # powermetrics writes a sample only after its interval ends

    def begin(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Start it first:\n"
                f"  sudo powermetrics --samplers gpu_power -i 1000 -f plist -o {self.path}"
            )

    def end(self, t0: float, t1: float) -> PowerReading:
        deadline = time.time() + self.wait_s
        while True:
            intervals = parse_powermetrics_plist(self.path.read_bytes())
            if (intervals and intervals[-1].end >= t1) or time.time() > deadline:
                return integrate(intervals, t0, t1)
            time.sleep(0.25)

    def close(self) -> None:
        pass


class NvmlSampler:
    """Sums over the given GPU indices (all visible GPUs by default)."""

    name = "nvml"

    def __init__(self, indices: list[int] | None = None, interval_s: float = 1.0):
        import pynvml  # linux-64 only (nvidia-ml-py)

        self.nvml = pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in (indices or range(count))]
        self.interval_s = interval_s
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._e0 = 0.0

    def _energy_mj(self) -> float:
        return float(sum(self.nvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in self.handles))

    def _power_w(self) -> float:
        return sum(self.nvml.nvmlDeviceGetPowerUsage(h) for h in self.handles) / 1000

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._samples.append(self._power_w())
            self._stop.wait(self.interval_s)

    def begin(self) -> None:
        self._samples = []
        self._stop.clear()
        self._e0 = self._energy_mj()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def end(self, t0: float, t1: float) -> PowerReading:
        self._stop.set()
        if self._thread:
            self._thread.join()
        energy_j = (self._energy_mj() - self._e0) / 1000
        # avg_w follows the paper (mean of 1 s samples); energy_j comes from the hardware counter.
        avg = statistics.fmean(self._samples) if self._samples else energy_j / max(t1 - t0, 1e-9)
        return PowerReading(avg, energy_j, len(self._samples), self.name)

    def close(self) -> None:
        self.nvml.nvmlShutdown()


def make_sampler(kind: str, **kwargs):
    if kind == "none":
        return NullSampler()
    if kind == "powermetrics":
        return PowermetricsFileSampler(kwargs.get("powermetrics_file", "results/raw/powermetrics.plist"))
    if kind == "nvml":
        return NvmlSampler(kwargs.get("gpu_indices"))
    raise ValueError(f"unknown power sampler: {kind}")
