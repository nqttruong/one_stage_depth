from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ogcde.geometry.camera import CameraIntrinsics


def apply_crop(image: np.ndarray, boxes: np.ndarray, intrinsics: CameraIntrinsics, x0: int, y0: int, crop_w: int, crop_h: int):
    h, w = image.shape[:2]
    x0 = int(max(0, min(x0, w)))
    y0 = int(max(0, min(y0, h)))
    crop_w = int(max(1, min(crop_w, w - x0)))
    crop_h = int(max(1, min(crop_h, h - y0)))
    cropped = image[y0 : y0 + crop_h, x0 : x0 + crop_w].copy()

    boxes_out = boxes.copy() if boxes.size else boxes.copy()
    if boxes_out.size:
        boxes_out[:, 0] = np.clip(boxes_out[:, 0] - x0, 0.0, float(crop_w))
        boxes_out[:, 2] = np.clip(boxes_out[:, 2] - x0, 0.0, float(crop_w))
        boxes_out[:, 1] = np.clip(boxes_out[:, 1] - y0, 0.0, float(crop_h))
        boxes_out[:, 3] = np.clip(boxes_out[:, 3] - y0, 0.0, float(crop_h))
        keep = (boxes_out[:, 2] > boxes_out[:, 0]) & (boxes_out[:, 3] > boxes_out[:, 1])
        boxes_out = boxes_out[keep]
    return cropped, boxes_out, intrinsics.crop(float(x0), float(y0))


def apply_resize(image: np.ndarray, boxes: np.ndarray, intrinsics: CameraIntrinsics, sx: float, sy: float):
    if image.ndim == 2:
        resized = cv2.resize(image, (0, 0), fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)
    else:
        resized = cv2.resize(image, (0, 0), fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)
    boxes_out = boxes.copy() if boxes.size else boxes.copy()
    if boxes_out.size:
        boxes_out[:, 0] *= sx
        boxes_out[:, 2] *= sx
        boxes_out[:, 1] *= sy
        boxes_out[:, 3] *= sy
    return resized, boxes_out, intrinsics.resize(float(sx), float(sy))


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
    boxes_out = boxes.copy() if boxes.size else boxes.copy()
    if boxes_out.size:
        boxes_out = boxes_out.copy()
        boxes_out[:, 0] *= scale
        boxes_out[:, 2] *= scale
        boxes_out[:, 1] *= scale
        boxes_out[:, 3] *= scale
        boxes_out[:, 0] += pad_x
        boxes_out[:, 2] += pad_x
        boxes_out[:, 1] += pad_y
        boxes_out[:, 3] += pad_y
    new_intr = intrinsics.letterbox(float(scale), (float(pad_x), float(pad_y)))
    return canvas, boxes_out, new_intr, (float(scale), float(scale)), (float(pad_x), float(pad_y))


def apply_hflip(image: np.ndarray, boxes: np.ndarray, intrinsics: CameraIntrinsics):
    width = image.shape[1]
    flipped = image[:, ::-1, :].copy()
    boxes_out = boxes.copy() if boxes.size else boxes.copy()
    if boxes_out.size:
        old_x1 = boxes_out[:, 0].copy()
        old_x2 = boxes_out[:, 2].copy()
        boxes_out[:, 0] = width - old_x2
        boxes_out[:, 2] = width - old_x1
    return flipped, boxes_out, intrinsics.hflip(float(width))
