import math

import torch

from ogcde.distance.targets import (
    direct_distance_target,
    geometry_residual_target,
)
from ogcde.geometry.camera import CameraIntrinsics


def test_direct_distance_target_uses_distance_not_depth():
    x = 3.0
    y = 4.0
    z = 12.0
    dist = math.sqrt(x * x + y * y + z * z)
    target = direct_distance_target(torch.tensor([dist], dtype=torch.float32))
    assert torch.isclose(target, torch.tensor([math.log(dist)], dtype=torch.float32)).all()


def test_geometry_residual_target_matches_definition():
    z_gt = torch.tensor([12.0], dtype=torch.float32)
    dist_gt = torch.tensor([13.0], dtype=torch.float32)
    intr = CameraIntrinsics(fx=400.0, fy=400.0, cx=320.0, cy=240.0)
    center = torch.tensor([[320.0, 240.0]], dtype=torch.float32)

    target = geometry_residual_target(z_gt, dist_gt, intrinsics=intr, box_center=center)
    s_gt = dist_gt / z_gt
    s_geo = intr.scale_from_pixel(320.0, 240.0)
    expected = torch.stack([torch.log(z_gt), torch.log(s_gt / s_geo)], dim=-1)
    assert torch.allclose(target, expected)


def test_hflip_intrinsics_use_width_minus_cx_convention():
    intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=301.25, cy=240.0)
    flipped = intr.hflip(640.0)
    assert abs(flipped.cx - (640.0 - 301.25)) < 1e-6
    assert abs(flipped.cy - 240.0) < 1e-6


def test_camera_ray_equivalence_for_resize():
    intr = CameraIntrinsics(fx=500.0, fy=600.0, cx=320.0, cy=240.0)
    resized = intr.resize(1.5, 1.5)
    u = 320.0
    v = 240.0
    u_new = 1.5 * u
    v_new = 1.5 * v
    ray_original = (u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy
    ray_new = (u_new - resized.cx) / resized.fx, (v_new - resized.cy) / resized.fy
    assert abs(ray_original[0] - ray_new[0]) < 1e-6
    assert abs(ray_original[1] - ray_new[1]) < 1e-6


def test_letterbox_intrinsics_are_scaled_once():
    intr = CameraIntrinsics(fx=700.0, fy=700.0, cx=320.0, cy=240.0)
    letterboxed = intr.letterbox(0.5, (16.0, 8.0))
    assert abs(letterboxed.fx - 350.0) < 1e-6
    assert abs(letterboxed.fy - 350.0) < 1e-6
    assert abs(letterboxed.cx - 176.0) < 1e-6
    assert abs(letterboxed.cy - 128.0) < 1e-6


def test_hflip_then_letterbox_preserves_both_transforms():
    intr = CameraIntrinsics(fx=500.0, fy=600.0, cx=320.0, cy=240.0)
    flipped = intr.hflip(640.0)
    letterboxed = flipped.letterbox(0.5, (32.0, 16.0))
    expected = CameraIntrinsics(fx=250.0, fy=300.0, cx=640.0 - 320.0, cy=240.0)
    expected = CameraIntrinsics(fx=250.0, fy=300.0, cx=expected.cx * 0.5 + 32.0, cy=expected.cy * 0.5 + 16.0)
    assert abs(letterboxed.fx - 250.0) < 1e-6
    assert abs(letterboxed.fy - 300.0) < 1e-6
    assert abs(letterboxed.cx - (320.0 * 0.5 + 32.0)) < 1e-6
    assert abs(letterboxed.cy - (240.0 * 0.5 + 16.0)) < 1e-6
