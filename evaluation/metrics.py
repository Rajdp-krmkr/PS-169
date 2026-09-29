"""
Quantitative performance metrics calculation for FSOC Coarse PAT benchmarking.

All tracking error metrics are computed strictly against known Ground Truth.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class FrameTelemetrySample:
    sim_time: float
    ground_truth_az_deg: float
    ground_truth_el_deg: float
    camera_pan_deg: float
    camera_tilt_deg: float
    pointing_error_deg: float
    is_in_fov: bool
    is_locked: bool
    pat_state: str
    detector_latency_ms: float
    fps: float
    detected_az_deg: Optional[float] = None
    detected_el_deg: Optional[float] = None
    predicted_az_deg: Optional[float] = None
    predicted_el_deg: Optional[float] = None


@dataclass
class PerformanceSummary:
    total_frames: int
    duration_sec: float
    mean_fps: float
    min_fps: float

    # Pointing Error Statistics (deg)
    initial_error_deg: float
    final_error_deg: float
    mean_error_deg: float
    std_error_deg: float
    max_error_deg: float
    rmse_deg: float
    error_p50_deg: float
    error_p95_deg: float
    error_p99_deg: float

    # Acquisition & State Timing
    acquisition_time_sec: float
    lock_retention_rate_pct: float
    total_lost_events: int
    mean_reacquisition_time_sec: float

    # System Latency (ms)
    mean_detector_latency_ms: float
    max_detector_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "duration_sec": round(self.duration_sec, 2),
            "mean_fps": round(self.mean_fps, 1),
            "min_fps": round(self.min_fps, 1),
            "initial_error_deg": round(self.initial_error_deg, 4),
            "final_error_deg": round(self.final_error_deg, 4),
            "mean_error_deg": round(self.mean_error_deg, 4),
            "std_error_deg": round(self.std_error_deg, 4),
            "max_error_deg": round(self.max_error_deg, 4),
            "rmse_deg": round(self.rmse_deg, 4),
            "error_p50_deg": round(self.error_p50_deg, 4),
            "error_p95_deg": round(self.error_p95_deg, 4),
            "error_p99_deg": round(self.error_p99_deg, 4),
            "acquisition_time_sec": round(self.acquisition_time_sec, 3) if self.acquisition_time_sec >= 0 else None,
            "lock_retention_rate_pct": round(self.lock_retention_rate_pct, 2),
            "total_lost_events": self.total_lost_events,
            "mean_reacquisition_time_sec": round(self.mean_reacquisition_time_sec, 3) if self.mean_reacquisition_time_sec >= 0 else None,
            "mean_detector_latency_ms": round(self.mean_detector_latency_ms, 2),
            "max_detector_latency_ms": round(self.max_detector_latency_ms, 2),
        }


class PerformanceMetricsCalculator:
    """
    Accumulates telemetry across a simulation run and computes ground-truth PAT metrics.
    """

    def __init__(self, lock_threshold_deg: float = 0.08) -> None:
        self.lock_thresh = lock_threshold_deg
        self.samples: List[FrameTelemetrySample] = []

        self.first_acquisition_time: Optional[float] = None
        self.reacquisition_latencies: List[float] = []
        self.lost_event_count: int = 0

        self._in_loss_state = False
        self._loss_start_time: Optional[float] = None

    def record_sample(self, sample: FrameTelemetrySample) -> None:
        """Record telemetry for a single frame."""
        self.samples.append(sample)

        # Check first acquisition
        if sample.is_locked and self.first_acquisition_time is None:
            self.first_acquisition_time = sample.sim_time

        # Monitor loss and reacquisition events
        if sample.pat_state in ("LOST", "UNCERTAIN"):
            if not self._in_loss_state:
                self._in_loss_state = True
                self._loss_start_time = sample.sim_time
                if sample.pat_state == "LOST":
                    self.lost_event_count += 1
        elif sample.pat_state == "TRACKING" and self._in_loss_state:
            self._in_loss_state = False
            if self._loss_start_time is not None:
                reacq_dt = sample.sim_time - self._loss_start_time
                self.reacquisition_latencies.append(reacq_dt)
                self._loss_start_time = None

    def compute_summary(self) -> PerformanceSummary:
        """Compute full statistical summary over all recorded frames."""
        if not self.samples:
            return PerformanceSummary(
                total_frames=0, duration_sec=0.0, mean_fps=0.0, min_fps=0.0,
                initial_error_deg=0.0, final_error_deg=0.0, mean_error_deg=0.0,
                std_error_deg=0.0, max_error_deg=0.0, rmse_deg=0.0,
                error_p50_deg=0.0, error_p95_deg=0.0, error_p99_deg=0.0,
                acquisition_time_sec=-1.0, lock_retention_rate_pct=0.0,
                total_lost_events=0, mean_reacquisition_time_sec=-1.0,
                mean_detector_latency_ms=0.0, max_detector_latency_ms=0.0,
            )

        errors = np.array([s.pointing_error_deg for s in self.samples], dtype=np.float64)
        fps_vals = np.array([s.fps for s in self.samples if s.fps > 0], dtype=np.float64)
        latencies = np.array([s.detector_latency_ms for s in self.samples], dtype=np.float64)

        duration = self.samples[-1].sim_time - self.samples[0].sim_time
        locked_count = sum(1 for s in self.samples if s.is_locked)
        lock_rate = (locked_count / len(self.samples)) * 100.0

        rmse = float(np.sqrt(np.mean(np.square(errors))))
        mean_reacq = float(np.mean(self.reacquisition_latencies)) if self.reacquisition_latencies else -1.0

        return PerformanceSummary(
            total_frames=len(self.samples),
            duration_sec=duration,
            mean_fps=float(np.mean(fps_vals)) if fps_vals.size > 0 else 0.0,
            min_fps=float(np.min(fps_vals)) if fps_vals.size > 0 else 0.0,
            initial_error_deg=float(errors[0]),
            final_error_deg=float(errors[-1]),
            mean_error_deg=float(np.mean(errors)),
            std_error_deg=float(np.std(errors)),
            max_error_deg=float(np.max(errors)),
            rmse_deg=rmse,
            error_p50_deg=float(np.percentile(errors, 50)),
            error_p95_deg=float(np.percentile(errors, 95)),
            error_p99_deg=float(np.percentile(errors, 99)),
            acquisition_time_sec=self.first_acquisition_time if self.first_acquisition_time is not None else -1.0,
            lock_retention_rate_pct=lock_rate,
            total_lost_events=self.lost_event_count,
            mean_reacquisition_time_sec=mean_reacq,
            mean_detector_latency_ms=float(np.mean(latencies)) if latencies.size > 0 else 0.0,
            max_detector_latency_ms=float(np.max(latencies)) if latencies.size > 0 else 0.0,
        )
