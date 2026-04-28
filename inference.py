"""Inference: run OGCDE on a video or an image folder.

For a video, also computes stability:
    - runs a simple IoU tracker
    - for each track with >=2 frames, collects distance predictions
    - reports mean per-track variance

Usage:
    python inference.py --ckpt runs/ogcde/best.pt --source my_video.mp4 --save-video
    python inference.py --ckpt runs/ogcde/best.pt --source image_folder
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import cv2
import numpy as np
import torch

from ogcde.model import OGCDENet
from ogcde.utils import decode_predictions, letterbox, unletterbox_boxes, unletterbox_points
from ogcde.metrics import stability, bbox_iou_xyxy, xywh_to_xyxy


CLASSES = ["Car", "Pedestrian", "Cyclist"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", required=True, help="video file or image folder")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--obj-thr", type=float, default=0.3)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--out", default="runs/infer")
    return ap.parse_args()


def preprocess(frame, size):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lb, ratio, pad = letterbox(rgb, size)
    t = torch.from_numpy(lb.transpose(2, 0, 1)).contiguous().float() / 255.0
    return t, ratio, pad


def draw(frame, dets):
    for b, s, c, d, dist, cp in zip(
        dets["boxes"], dets["scores"], dets["classes"],
        dets["depth"], dets["distance"], dets["contact"],
    ):
        x, y, w, h = b
        x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (int(cp[0]), int(cp[1])), 4, (0, 0, 255), -1)
        label = f"{CLASSES[int(c)]} {dist:.1f}m  d={float(d):.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame


class SimpleIoUTracker:
    """One-step greedy IoU association — not SOTA, but enough for stability eval."""

    def __init__(self, iou_thr=0.3, max_lost=10):
        self.next_id = 0
        self.tracks = {}  # id -> dict{last_xyxy, lost}
        self.iou_thr = iou_thr
        self.max_lost = max_lost

    def update(self, boxes_xywh):
        assignments = [-1] * len(boxes_xywh)
        if len(self.tracks) and len(boxes_xywh):
            tids = list(self.tracks.keys())
            track_boxes = np.array([self.tracks[t]["xyxy"] for t in tids])
            det_xyxy = xywh_to_xyxy(boxes_xywh)
            ious = bbox_iou_xyxy(det_xyxy, track_boxes)  # (N, T)
            while True:
                if ious.size == 0 or ious.max() < self.iou_thr:
                    break
                i, j = np.unravel_index(ious.argmax(), ious.shape)
                assignments[i] = tids[j]
                ious[i, :] = -1
                ious[:, j] = -1

        # increment lost
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["lost"] += 1

        # update matched / create new
        for i, tid in enumerate(assignments):
            xyxy = xywh_to_xyxy(np.asarray(boxes_xywh[i:i+1]))[0]
            if tid == -1:
                tid = self.next_id
                self.next_id += 1
                assignments[i] = tid
            self.tracks[tid] = {"xyxy": xyxy, "lost": 0}

        # drop stale
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["lost"] > self.max_lost:
                del self.tracks[tid]

        return assignments


def iter_frames(source):
    if os.path.isdir(source):
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = sorted([os.path.join(source, f) for f in os.listdir(source)
                        if f.lower().endswith(exts)])
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                yield img, f
    else:
        cap = cv2.VideoCapture(source)
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f, f"frame_{i:06d}"
            i += 1
        cap.release()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    model = OGCDENet(nc=args.num_classes).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tracker = SimpleIoUTracker()
    track_dists = defaultdict(list)  # id -> list of distances
    writer = None

    for frame, name in iter_frames(args.source):
        t, ratio, pad = preprocess(frame, args.img_size)
        t = t.unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(t)
            dets = decode_predictions(
                preds, args.num_classes,
                obj_thr=args.obj_thr, iou_thr=args.iou_thr,
            )[0]

        if len(dets["boxes"]) > 0:
            boxes = unletterbox_boxes(dets["boxes"].cpu().numpy(), ratio, pad)
            contacts = unletterbox_points(dets["contact"].cpu().numpy(), ratio, pad)
            dets_orig = {
                "boxes": boxes,
                "scores": dets["scores"].cpu().numpy(),
                "classes": dets["classes"].cpu().numpy(),
                "depth": dets["depth"].cpu().numpy(),
                "distance": dets["distance"].cpu().numpy(),
                "contact": contacts,
            }
            ids = tracker.update(boxes)
            for tid, dist in zip(ids, dets_orig["distance"]):
                track_dists[tid].append(float(dist))
            frame = draw(frame, dets_orig)

        if args.save_video:
            if writer is None:
                h, w = frame.shape[:2]
                path = os.path.join(args.out, "out.mp4")
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
            writer.write(frame)

    if writer:
        writer.release()

    stab = stability(track_dists)
    print(f"Stability (mean per-track variance of distance): {stab:.4f}")


if __name__ == "__main__":
    main()
