#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ogcde.datasets import KITTIDistanceDataset, FCOSTargetAdapter, YOLOTargetAdapter
from ogcde.detectors import build_detector
from ogcde.distance import build_distance_method


def create_synthetic_kitti(root: Path):
    (root / "image_2").mkdir(parents=True, exist_ok=True)
    (root / "label_2").mkdir(parents=True, exist_ok=True)
    (root / "calib").mkdir(parents=True, exist_ok=True)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    Image.fromarray(image).save(root / "image_2" / "000000.png")
    (root / "label_2" / "000000.txt").write_text(
        "Car 0.00 0 0.00 4.0 4.0 10.0 12.0 5.0 1.8 1.6 3.0 1.0 2.0 10.0 0.0\n",
        encoding="utf-8",
    )
    (root / "calib" / "000000.txt").write_text(
        "P2: 500 0 16 0 0 600 16 0 0 0 1 0\n",
        encoding="utf-8",
    )
    (root / "split.txt").write_text("000000\n", encoding="utf-8")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        create_synthetic_kitti(root)

        for detector_name in ["yolov8_custom", "fcos"]:
            for method_name in ["direct_distance", "learned_scale", "analytic_geometry", "geometry_residual"]:
                detector = build_detector(detector_name, nc=3, backbone_size="n")
                method = build_distance_method(method_name)
                dataset = KITTIDistanceDataset(str(root), str(root / "split.txt"), img_size=64, augment=False, hflip=False)
                image, target, meta = dataset[0]
                assert set(target.keys()) >= {"boxes", "labels", "z_gt", "distance_gt", "loc3d", "object_ids"}
                YOLOTargetAdapter()(target)
                FCOSTargetAdapter()(target)
                print(f"PASS {detector_name} + {method_name}")


if __name__ == "__main__":
    main()
