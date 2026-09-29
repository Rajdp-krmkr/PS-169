"""
Control Delay Queue for Realistic Transport and Actuation Latency Modeling.

In real-world FSOC coarse PAT systems, the end-to-end feedback loop experiences
latency across several stages:
1. Camera sensor exposure & readout time (10-33 ms)
2. Image preprocessing, AI inference & centroiding (5-20 ms)
3. State estimation & PID/Kalman compute (1-3 ms)
4. Motor drive communication bus & actuator lag (10-30 ms)

ControlDelayQueue implements a discrete FIFO buffer for gimbal command rate/position
deltas to accurately model this phase lag.
"""

from __future__ import annotations
from collections import deque
from typing import Tuple, Optional


class ControlDelayQueue:
    """
    FIFO queue introducing integer frame delay into gimbal control actuation.
    """

    def __init__(self, delay_frames: int = 0) -> None:
        self.delay_frames = max(0, int(delay_frames))
        self.queue: deque[Tuple[float, float]] = deque()
        self.reset()

    def reset(self) -> None:
        """Clear queue and pre-fill with zero commands."""
        self.queue.clear()
        for _ in range(self.delay_frames):
            self.queue.append((0.0, 0.0))

    def set_delay_frames(self, delay_frames: int) -> None:
        """Adjust delay dynamically."""
        self.delay_frames = max(0, int(delay_frames))
        self.reset()

    def step(self, cmd_pan: float, cmd_tilt: float) -> Tuple[float, float]:
        """
        Buffer input command and return the delayed command.
        If delay_frames is 0, passes commands immediately through.
        """
        if self.delay_frames <= 0:
            return float(cmd_pan), float(cmd_tilt)

        self.queue.append((float(cmd_pan), float(cmd_tilt)))
        delayed_pan, delayed_tilt = self.queue.popleft()
        return delayed_pan, delayed_tilt
