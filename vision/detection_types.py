"""
Data models and types for optical beacon detection and tracking.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> Tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return float(self.w * self.h)


@dataclass
class Detection:
    center_u: float
    center_v: float
    bbox: BoundingBox
    confidence: float
    area: float
    peak_intensity: float
    circularity: float
    detector_name: str
    timestamp: float
    class_id: int = 1
    class_name: str = "optical_beacon"

    @property
    def center(self) -> Tuple[float, float]:
        return self.center_u, self.center_v


@dataclass
class DetectionResult:
    primary: Optional[Detection]
    candidates: List[Detection] = field(default_factory=list)
    latency_ms: float = 0.0
    success: bool = False
