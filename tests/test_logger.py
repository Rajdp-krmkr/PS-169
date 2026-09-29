"""
Unit tests for telemetry logger (CSV and JSON).
"""

import os
import shutil
import csv
import json
import pytest
from evaluation.logger import TelemetryLogger


def test_telemetry_logger():
    test_dir = "reports/test_log_dir"
    logger = TelemetryLogger(scenario_name="test_run", output_dir=test_dir, auto_timestamp=False)

    try:
        # Log 3 frames
        for i in range(3):
            logger.log_frame({
                "sim_time": i * 0.033,
                "frame_idx": i,
                "gt_az_deg": 1.2,
                "gt_el_deg": 0.4,
                "pan_deg": 1.0,
                "tilt_deg": 0.3,
                "error_deg": 0.22,
                "is_locked": True,
                "state": "TRACKING",
                "fps": 30.0,
                "det_latency_ms": 4.5,
            })

        csv_p, json_p = logger.finalize(summary_metrics={"test_metric": 123.45})

        assert os.path.exists(csv_p)
        assert os.path.exists(json_p)

        # Check CSV
        with open(csv_p, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 3
            assert float(reader[0]["pointing_error_deg"]) == 0.22

        # Check JSON
        with open(json_p, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["total_records"] == 3
            assert data["summary_metrics"]["test_metric"] == 123.45

    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
