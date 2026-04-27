"""Process compute usage (wall time, RAM, CPU time, CUDA peak) for audit reports."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class ComputeUsageSnapshot:
    """Resource use over a code block (typically one inference or training call)."""

    wall_time_ms: float
    process_cpu_time_ms: Optional[float] = None
    process_ram_start_mb: Optional[float] = None
    process_ram_end_mb: Optional[float] = None
    process_ram_delta_mb: Optional[float] = None
    gpu_peak_alloc_mb: Optional[float] = None
    gpu_device: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ComputeUsageRecorder:
    """Context manager: measure wall time, optional psutil RAM/CPU, optional PyTorch CUDA peak."""

    __slots__ = ("_cpu0", "_proc", "_rss0", "_t0", "usage")

    def __init__(self) -> None:
        self._t0: float = 0.0
        self._cpu0: Any = None
        self._rss0: Optional[int] = None
        self._proc: Any = None
        self.usage: Optional[Dict[str, Any]] = None

    def __enter__(self) -> ComputeUsageRecorder:
        self._t0 = time.perf_counter()
        self._cpu0 = None
        self._rss0 = None
        self._proc = None
        try:
            import psutil  # type: ignore

            self._proc = psutil.Process()
            self._cpu0 = self._proc.cpu_times()
            self._rss0 = self._proc.memory_info().rss
        except Exception:
            pass

        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        return self

    def __exit__(self, *exc: Any) -> None:
        t1 = time.perf_counter()
        wall_ms = (t1 - self._t0) * 1000.0

        cpu_ms: Optional[float] = None
        rss_start_mb: Optional[float] = None
        rss_end_mb: Optional[float] = None
        rss_delta_mb: Optional[float] = None

        if self._proc is not None and self._cpu0 is not None:
            try:
                cpu1 = self._proc.cpu_times()
                cpu_ms = (cpu1.user - self._cpu0.user + cpu1.system - self._cpu0.system) * 1000.0
                rss_end = self._proc.memory_info().rss
                rss_end_mb = rss_end / (1024.0**2)
                if self._rss0 is not None:
                    rss_start_mb = self._rss0 / (1024.0**2)
                    rss_delta_mb = (rss_end - self._rss0) / (1024.0**2)
            except Exception:
                pass

        gpu_peak: Optional[float] = None
        gpu_name: Optional[str] = None
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                gpu_peak = torch.cuda.max_memory_allocated(0) / (1024.0**2)
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

        snap = ComputeUsageSnapshot(
            wall_time_ms=round(wall_ms, 2),
            process_cpu_time_ms=round(cpu_ms, 2) if cpu_ms is not None else None,
            process_ram_start_mb=round(rss_start_mb, 2) if rss_start_mb is not None else None,
            process_ram_end_mb=round(rss_end_mb, 2) if rss_end_mb is not None else None,
            process_ram_delta_mb=round(rss_delta_mb, 3) if rss_delta_mb is not None else None,
            gpu_peak_alloc_mb=round(gpu_peak, 2) if gpu_peak is not None else None,
            gpu_device=gpu_name,
        )
        self.usage = snap.as_dict()
