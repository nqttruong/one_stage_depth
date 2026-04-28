"""KITTI dataset for OGCDE one-stage training.

KITTI label format (15 columns):
    type  truncated  occluded  alpha
    bbox_left  bbox_top  bbox_right  bbox_bottom
    h  w  l                # 3D dimensions (meters)
    x  y  z                # 3D location in camera coord (meters).
                           # Note: (x, y, z) is BOTTOM CENTER of the 3D box.
    rotation_y             # around Y axis in camera coord

Calibration file: P2 is the 3x4 projection matrix for the left color camera.

Contact point: we project the 3D bottom center (x, y, z) to 2D via P2.
    [u, v, w] = P2 @ [x, y, z, 1];  u /= w;  v /= w.

Augmentation policy (this file):
  - Default: color jitter only (P2-safe).
  - Optional `hflip=True`: horizontal flip is also geometry-safe under our
    formulation because (a) ||xyz|| is invariant to x-sign flip, (b) z is
    invariant to x-sign flip, so distance/depth/scale targets are unchanged;
    only the 2D bbox + 2D contact point need their u-coordinate mirrored.
    P2 itself is not used after GT extraction, so we don't track it.
"""

from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from .utils import letterbox


# --------------------------- KITTI file parsing -----------------------------

def parse_kitti_calib(path):
    calib = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            calib[key.strip()] = np.array([float(x) for x in val.split()], dtype=np.float64)
    P2 = calib["P2"].reshape(3, 4)
    return {"P2": P2}


def parse_kitti_label(path):
    """Return list of object dicts."""
    if not os.path.exists(path):
        return []
    objects = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            objects.append({
                "type": parts[0],
                "truncated": float(parts[1]),
                "occluded": int(parts[2]),
                "alpha": float(parts[3]),
                "bbox": [float(x) for x in parts[4:8]],      # l, t, r, b
                "dims": [float(x) for x in parts[8:11]],     # h, w, l
                "loc":  [float(x) for x in parts[11:14]],    # x, y, z (bottom center)
                "rot_y": float(parts[14]),
            })
    return objects


def project_3d_to_2d(pts_3d: np.ndarray, P: np.ndarray) -> np.ndarray:
    """(N,3) -> (N,2) image pixels."""
    n = pts_3d.shape[0]
    pts_h = np.hstack([pts_3d, np.ones((n, 1))])
    p = pts_h @ P.T
    return p[:, :2] / p[:, 2:3]


def compute_contact_point_2d(obj, P2):
    """Bottom center of 3D bbox -> 2D. KITTI `loc` is already the bottom center."""
    xyz = np.array(obj["loc"], dtype=np.float64).reshape(1, 3)
    return project_3d_to_2d(xyz, P2)[0]  # (u, v)


# ------------------------------ dataset -------------------------------------

class KITTIOGCDEDataset(Dataset):
    """KITTI for OGCDE one-stage.

    Args:
        root: KITTI root with subfolders image_2/, label_2/, calib/.
        split_file: path to a text file with one image id per line
                    (e.g. "000000"). Use standard Eigen train/val splits.
        img_size: letterbox target.
        distance_mode: 'euclidean' (||x,y,z||) or 'depth' (z only).
                       The scale factor s = distance / depth depends on this.
        depth_mode: 'bottom_z' (z of bottom center, same as loc[2]) or
                    'center_z' (z of 3D bbox center = loc[2] - h/2).
    """

    CLASS_MAP = {
        "Car":            0,
        "Van":            0,
        "Truck":          0,
        "Pedestrian":     1,
        "Person_sitting": 1,
        "Cyclist":        2,
    }

    def __init__(self, root, split_file, img_size=640, augment=False,
                 distance_mode="euclidean", depth_mode="bottom_z",
                 filter_difficult=True, hflip=False,
                 depth_source=None):
        """
        Args:
            root, split_file, img_size: standard.
            augment: master switch for any augmentation.
            distance_mode: 'euclidean' (||x,y,z||) or 'depth' (z only).
            depth_mode: 'bottom_z' or 'center_z'. Used only when
                depth_source is None (default fast pipeline).
            hflip: if True AND augment, randomly horizontal-flip with prob 0.5.
                Geometry-safe under our setup (see module docstring).
            depth_source: optional path to a JSON or npy mapping
                {image_id: list of per-object depths}. Use this for
                "stage 2" training with median-LiDAR-Z per object.
                If None, falls back to depth_mode.
        """
        self.root = root
        self.image_dir = os.path.join(root, "image_2")
        self.label_dir = os.path.join(root, "label_2")
        self.calib_dir = os.path.join(root, "calib")
        with open(split_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]
        self.img_size = img_size
        self.augment = augment
        self.distance_mode = distance_mode
        self.depth_mode = depth_mode
        self.filter_difficult = filter_difficult
        self.hflip = hflip

        self._depth_lookup = None
        if depth_source is not None:
            self._depth_lookup = self._load_depth_source(depth_source)

    def __len__(self):
        return len(self.ids)

    # --------------------------------------------------------------------
    @staticmethod
    def _load_depth_source(path):
        """Load per-object depth overrides.

        Supports .json mapping {image_id: [d1, d2, ...]} where the list is
        aligned with the kept-object order produced by `_build_targets`
        (post-filtering). Use prepare_lidar_depth.py to build a compatible
        file. If a key is missing, falls back to depth_mode for that image.
        """
        import json
        if path.endswith(".json"):
            with open(path) as f:
                return json.load(f)
        elif path.endswith(".npy"):
            return np.load(path, allow_pickle=True).item()
        raise ValueError(f"Unsupported depth_source extension: {path}")

    @staticmethod
    def _color_jitter(img, b=0.2, c=0.2, s=0.2):
        """Light photometric augmentation in HSV+RGB space.

        Applies random brightness/contrast/saturation perturbations.
        Geometry-safe by construction (no spatial change).
        """
        img = img.astype(np.float32)
        # brightness (additive)
        if b > 0:
            img += np.random.uniform(-b, b) * 255.0
        # contrast (mean-preserving scale)
        if c > 0:
            mean = img.mean()
            img = (img - mean) * (1 + np.random.uniform(-c, c)) + mean
        # saturation (HSV S channel)
        if s > 0:
            hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                               cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] *= 1 + np.random.uniform(-s, s)
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)

    # --------------------------------------------------------------------
    def _build_targets(self, objects, P2, image_id):
        boxes, labels, dists, depths, contacts = [], [], [], [], []
        for obj in objects:
            if obj["type"] not in self.CLASS_MAP:
                continue
            if self.filter_difficult and (obj["truncated"] > 0.5 or obj["occluded"] > 2):
                continue

            l, t, r, b = obj["bbox"]
            w = r - l
            h_ = b - t
            if w < 5 or h_ < 5:
                continue

            cx = (l + r) / 2.0
            cy = (t + b) / 2.0
            boxes.append([cx, cy, w, h_])
            labels.append(self.CLASS_MAP[obj["type"]])

            x, y, z = obj["loc"]          # bottom center in cam coord
            if self.depth_mode == "bottom_z":
                d = z
            else:  # center_z
                d = z - obj["dims"][0] / 2.0

            if self.distance_mode == "euclidean":
                dist = float(np.sqrt(x * x + y * y + z * z))
            else:  # depth
                dist = float(z)

            dists.append(dist)
            depths.append(float(d))

            cp = compute_contact_point_2d(obj, P2)
            contacts.append([float(cp[0]), float(cp[1])])

        # depth_source override (stage-2 training with median LiDAR Z)
        if self._depth_lookup is not None and image_id in self._depth_lookup:
            override = self._depth_lookup[image_id]
            if len(override) == len(depths):
                depths = list(override)

        return (
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(labels, dtype=np.int64),
            np.asarray(dists, dtype=np.float32),
            np.asarray(depths, dtype=np.float32),
            np.asarray(contacts, dtype=np.float32).reshape(-1, 2),
        )

    # --------------------------------------------------------------------
    def __getitem__(self, idx):
        fid = self.ids[idx]
        img_path = os.path.join(self.image_dir, f"{fid}.png")
        label_path = os.path.join(self.label_dir, f"{fid}.txt")
        calib_path = os.path.join(self.calib_dir, f"{fid}.txt")

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        calib = parse_kitti_calib(calib_path)
        P2 = calib["P2"]

        objects = parse_kitti_label(label_path)
        boxes, labels, dists, depths, contacts = self._build_targets(objects, P2, fid)

        # photometric aug (default ON when augment=True; P2-safe).
        if self.augment:
            img = self._color_jitter(img)

        # letterbox
        img_lb, ratio, (px, py) = letterbox(img, self.img_size)
        if len(boxes) > 0:
            boxes[:, 0] = boxes[:, 0] * ratio + px
            boxes[:, 1] = boxes[:, 1] * ratio + py
            boxes[:, 2] *= ratio
            boxes[:, 3] *= ratio
            contacts[:, 0] = contacts[:, 0] * ratio + px
            contacts[:, 1] = contacts[:, 1] * ratio + py

        # optional H-flip (opt-in via self.hflip).
        # Geometry-safe: distance/depth/scale invariant to x-sign flip;
        # only need to mirror 2D coordinates (boxes + contact points).
        if self.augment and self.hflip and np.random.rand() < 0.5:
            img_lb = img_lb[:, ::-1, :].copy()
            W = img_lb.shape[1]
            if len(boxes) > 0:
                boxes[:, 0] = W - boxes[:, 0]
                contacts[:, 0] = W - contacts[:, 0]

        img_t = torch.from_numpy(img_lb.transpose(2, 0, 1)).contiguous().float() / 255.0

        return img_t, {
            "boxes":   torch.from_numpy(boxes),
            "labels":  torch.from_numpy(labels),
            "dist":    torch.from_numpy(dists),
            "depth":   torch.from_numpy(depths),
            "contact": torch.from_numpy(contacts),
            "image_id": fid,
            "orig_shape": img.shape[:2],   # (H, W) before letterbox
            "ratio": ratio,
            "pad": (px, py),
        }


# ------------------------------ collate -------------------------------------

def collate_ogcde(batch):
    """Flatten per-image targets across a batch with a batch_idx vector."""
    imgs = torch.stack([b[0] for b in batch], 0)
    meta = [b[1] for b in batch]

    boxes_l, labels_l, dist_l, depth_l, contact_l, batch_l = [], [], [], [], [], []
    for i, t in enumerate(meta):
        n = len(t["boxes"])
        if n == 0:
            continue
        boxes_l.append(t["boxes"])
        labels_l.append(t["labels"])
        dist_l.append(t["dist"])
        depth_l.append(t["depth"])
        contact_l.append(t["contact"])
        batch_l.append(torch.full((n,), i, dtype=torch.long))

    if not boxes_l:
        targets = {
            "boxes": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.long),
            "dist": torch.zeros(0),
            "depth": torch.zeros(0),
            "contact": torch.zeros(0, 2),
            "batch_idx": torch.zeros(0, dtype=torch.long),
        }
    else:
        targets = {
            "boxes":     torch.cat(boxes_l, 0),
            "labels":    torch.cat(labels_l, 0),
            "dist":      torch.cat(dist_l, 0),
            "depth":     torch.cat(depth_l, 0),
            "contact":   torch.cat(contact_l, 0),
            "batch_idx": torch.cat(batch_l, 0),
        }

    return imgs, targets, meta
