#!/usr/bin/env python3
"""GPU metric sampling helpers for OmniEdge E2E runs."""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VramSample:
    timestamp_s: float
    used_mib: int
    total_mib: int


@dataclass
class VramSnapshot:
    available: bool
    baseline_mib: Optional[int]
    peak_mib: Optional[int]
    total_mib: Optional[int]
    sample_count: int
    warning: Optional[str] = None
    samples: List[VramSample] = field(default_factory=list)


class NvidiaSmiSampler:
    """Samples VRAM usage once per second using nvidia-smi."""

    def __init__(self, poll_interval_s: float = 1.0) -> None:
        self._poll_interval_s = poll_interval_s
        self._samples: List[VramSample] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._warning: Optional[str] = None
        self._available = True

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> VramSnapshot:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        with self._lock:
            if not self._samples:
                return VramSnapshot(
                    available=self._available,
                    baseline_mib=None,
                    peak_mib=None,
                    total_mib=None,
                    sample_count=0,
                    warning=self._warning,
                    samples=[],
                )

            baseline = self._samples[0].used_mib
            peak = max(s.used_mib for s in self._samples)
            total = self._samples[0].total_mib
            return VramSnapshot(
                available=self._available,
                baseline_mib=baseline,
                peak_mib=peak,
                total_mib=total,
                sample_count=len(self._samples),
                warning=self._warning,
                samples=list(self._samples),
            )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = self._sample_once()
            if sample is not None:
                with self._lock:
                    self._samples.append(sample)
            time.sleep(self._poll_interval_s)

    def _sample_once(self) -> Optional[VramSample]:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            self._available = False
            self._warning = "nvidia-smi not found"
            return None

        if proc.returncode != 0:
            self._available = False
            self._warning = (proc.stderr or proc.stdout or "nvidia-smi failed").strip()
            return None

        line = (proc.stdout.strip().splitlines() or [""])[0].strip()
        if not line:
            self._available = False
            self._warning = "nvidia-smi returned empty output"
            return None

        try:
            used_raw, total_raw = [part.strip() for part in line.split(",")[:2]]
            used_mib = int(used_raw)
            total_mib = int(total_raw)
        except (ValueError, IndexError):
            self._available = False
            self._warning = f"unparseable nvidia-smi output: {line}"
            return None

        self._available = True
        self._warning = None
        return VramSample(
            timestamp_s=time.monotonic(),
            used_mib=used_mib,
            total_mib=total_mib,
        )
