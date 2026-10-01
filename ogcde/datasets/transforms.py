from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ogcde.geometry.camera import CameraIntrinsics


def apply_crop(image: np.ndarray, boxes: np.ndarray, intrinsics: CameraIntrinsics, x0: int, y0: int):
    h, w = image.shape[:2]
    img = image[y0 : y0 + h, x0 : x0 + w] if False else image
    boxes = boxes.copy()
    if boxes.size:
        boxes[:, 0] -= x0
        boxes[:, 2] -= x0
        boxes[:, 1] -= y0
        boxes[:, 3] -= y0
    return img, boxes, intrinsics.crop(float(x0), float(y0))


def apply_resize(image: np.ndarray, boxes: np.ndarray, intrinsics: CameraIntrinsics, sx: float, sy: float):
    if boxes.size:
        boxes[:, 0] *= sx
        boxes[:, 2] *= sx
        boxes[:, 1] *= sy
        boxes[:, 3] *= sy
    return image, boxes, intrinsics.resize(float(sx), float(sy))


def letterbox_image(image: np.ndarray, target_size: int, boxes: np.ndarray, intrinsics: CameraIntrinsics):
    h, w = image.shape[:2]
    scale = min(target_size / float(h), target_size / float(w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, image.shape[2]), 128, dtype=image.dtype)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    if boxes.size:
        boxes = boxes.copy()
        boxes[:, 0] *= scale
        boxes[:, 2] *= scale
        boxes[:, 1] *= scale
        boxes[:, 3] *= scale
        boxes[:, 0] += pad_x
        boxes[:, 2] += pad_x
        boxes[:, 1] += pad_y
        boxes[:, 3] += pad_y
    new_intr = intrinsics.resize(float(scale), float(scale)).letterbox(float(scale), (float(pad_x), float(pad_y)))
    return canvas, boxes, new_intr, (float(scale), float(scale)), (float(pad_x), float(pad_y))


def apply_hflip(image: np.ndarray, boxes: np.ndarray, intrinsics: CameraIntrinsics):
    width = image.shape[1]
    flipped = image[:, ::-1, :].copy()
    if boxes.size:
        boxes = boxes.copy()
        boxes[:, 0] = width - boxes[:, 2]
        boxes[:, 2] = width - boxes[:, 0]
    return flipped, boxes, intrinsics.hflip(float(width))
