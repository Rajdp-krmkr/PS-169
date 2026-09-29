"""
Unit tests for coordinate conversions and optical projection models.
"""

import math
import pytest
from simulation.coordinate_system import (
    deg2rad,
    rad2deg,
    normalize_angle_deg,
    normalize_angle_rad,
    OpticalProjection,
)


def test_unit_conversions():
    assert math.isclose(deg2rad(180.0), math.pi, rel_tol=1e-6)
    assert math.isclose(rad2deg(math.pi / 2.0), 90.0, rel_tol=1e-6)
    assert math.isclose(normalize_angle_deg(370.0), 10.0, abs_tol=1e-6)
    assert math.isclose(normalize_angle_deg(-190.0), 170.0, abs_tol=1e-6)
    assert math.isclose(normalize_angle_rad(3.0 * math.pi), math.pi, abs_tol=1e-6)


def test_optical_projection_center():
    proj = OpticalProjection(
        resolution=(1280, 720),
        fov_horizontal_rad=deg2rad(20.0),
        fov_vertical_rad=deg2rad(11.25),
    )

    # Center target has zero angular error
    u, v = proj.angles_to_pixels(0.0, 0.0)
    assert math.isclose(u, 640.0, abs_tol=1e-5)
    assert math.isclose(v, 360.0, abs_tol=1e-5)

    # Inverse projection at center recovers zero angular error
    d_az, d_el = proj.pixels_to_angular_error(640.0, 360.0)
    assert math.isclose(d_az, 0.0, abs_tol=1e-7)
    assert math.isclose(d_el, 0.0, abs_tol=1e-7)


def test_optical_projection_roundtrip():
    proj = OpticalProjection(
        resolution=(1920, 1080),
        fov_horizontal_rad=deg2rad(15.0),
        fov_vertical_rad=deg2rad(8.4375),
    )

    test_angles = [
        (deg2rad(2.5), deg2rad(1.2)),
        (deg2rad(-4.0), deg2rad(2.0)),
        (deg2rad(5.0), deg2rad(-3.0)),
    ]

    for az_in, el_in in test_angles:
        u, v = proj.angles_to_pixels(az_in, el_in, exact=True)
        az_out, el_out = proj.pixels_to_angular_error(u, v, exact=True)
        assert math.isclose(az_in, az_out, rel_tol=1e-5, abs_tol=1e-6)
        assert math.isclose(el_in, el_out, rel_tol=1e-5, abs_tol=1e-6)


def test_fov_containment():
    proj = OpticalProjection(
        resolution=(1280, 720),
        fov_horizontal_rad=deg2rad(20.0),
        fov_vertical_rad=deg2rad(10.0),
    )

    # Inside FOV
    assert proj.is_in_fov(deg2rad(8.0), deg2rad(4.0)) is True
    # Outside FOV horizontally
    assert proj.is_in_fov(deg2rad(11.0), deg2rad(0.0)) is False
    # Outside FOV vertically
    assert proj.is_in_fov(deg2rad(0.0), deg2rad(6.0)) is False
