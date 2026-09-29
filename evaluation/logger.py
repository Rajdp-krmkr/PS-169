"""
High-rate structured telemetry stream logger (CSV and JSON).

Records per-frame ground truth, sensor detections, Kalman estimates,
controller commands, and execution latencies.
"""

from __future__ import annotations
import csv
import json
import os
import time
from typing import Dict, Any, List, Optional


class TelemetryLogger:
    """
    Records and persists simulation telemetry streams to CSV and JSON formats.
    """

    def __init__(
        self,
        scenario_name: str = "simulation",
        output_dir: str = "reports",
        auto_timestamp: bool = True,
    ) -> None:
        self.scenario_name = scenario_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        t_str = time.strftime("%Y%m%d_%H%M%S") if auto_timestamp else "latest"
        self.csv_path = os.path.join(output_dir, f"telemetry_{scenario_name}_{t_str}.csv")
        self.json_path = os.path.join(output_dir, f"telemetry_{scenario_name}_{t_str}.json")

        self.records: List[Dict[str, Any]] = []

        self.csv_fields = [
            "sim_time",
            "frame_idx",
            "ground_truth_az_deg",
            "ground_truth_el_deg",
            "camera_pan_deg",
            "camera_tilt_deg",
            "pointing_error_deg",
            "is_locked",
            "pat_state",
            "fps",
            "detector_latency_ms",
            "detected_u",
            "detected_v",
            "predicted_az_deg",
            "predicted_el_deg",
            "detection_confidence",
        ]

        self._init_csv()

    def _init_csv(self) -> None:
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fields)
            writer.writeheader()

    def log_frame(self, data: Dict[str, Any]) -> None:
        """Log telemetry for a single simulation frame."""
        record = {
            "sim_time": round(float(data.get("sim_time", 0.0)), 4),
            "frame_idx": int(data.get("frame_idx", len(self.records))),
            "ground_truth_az_deg": round(float(data.get("gt_az_deg", 0.0)), 4),
            "ground_truth_el_deg": round(float(data.get("gt_el_deg", 0.0)), 4),
            "camera_pan_deg": round(float(data.get("pan_deg", 0.0)), 4),
            "camera_tilt_deg": round(float(data.get("tilt_deg", 0.0)), 4),
            "pointing_error_deg": round(float(data.get("error_deg", 0.0)), 4),
            "is_locked": int(bool(data.get("is_locked", False))),
            "pat_state": str(data.get("state", "UNKNOWN")),
            "fps": round(float(data.get("fps", 0.0)), 1),
            "detector_latency_ms": round(float(data.get("det_latency_ms", 0.0)), 2),
            "detected_u": round(float(data.get("detected_u", -1.0)), 1) if data.get("detected_u") is not None else -1.0,
            "detected_v": round(float(data.get("detected_v", -1.0)), 1) if data.get("detected_v") is not None else -1.0,
            "predicted_az_deg": round(float(data.get("pred_az_deg", 0.0)), 4) if data.get("pred_az_deg") is not None else None,
            "predicted_el_deg": round(float(data.get("pred_el_deg", 0.0)), 4) if data.get("pred_el_deg") is not None else None,
            "detection_confidence": round(float(data.get("confidence", 0.0)), 3),
        }

        self.records.append(record)

        # Append to CSV
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fields)
            writer.writerow(record)

    def finalize(self, summary_metrics: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """Save full telemetry and summary metrics to JSON."""
        payload = {
            "scenario": self.scenario_name,
            "total_records": len(self.records),
            "summary_metrics": summary_metrics or {},
            "telemetry_stream": self.records,
        }

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return self.csv_path, self.json_path
