"""Inference: run OGCDE on a video, image folder, or live camera.

For video/camera, also computes stability:
    - runs a simple IoU tracker
    - for each track with >=2 frames, collects distance predictions
    - reports mean per-track variance

Usage:
    python inference.py --ckpt runs/ogcde/best.pt --source my_video.mp4 --save-video
    python inference.py --ckpt runs/ogcde/best.pt --source image_folder
    python inference.py --ckpt runs/ogcde/best.pt --camera          # webcam 0
    python inference.py --ckpt runs/ogcde/best.pt --camera 1        # webcam 1
"""

from __future__ import annotations

import argparse
import os
import time
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
    ap.add_argument("--source", default=None, help="video file or image folder")
    ap.add_argument("--camera", type=int, nargs="?", const=0, default=None,
                    help="camera index for live inference (default 0)")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--obj-thr", type=float, default=0.65)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--min-box", type=int, default=30,
                    help="Min width AND height (px) to keep a detection (default 30)")
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--out", default="runs/infer")
    ap.add_argument("--display-scale", type=float, default=0.5,
                    help="Scale factor for display window (default 0.5 = half size)")
    return ap.parse_args()


def preprocess(frame, size):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lb, ratio, pad = letterbox(rgb, size)
    t = torch.from_numpy(lb.transpose(2, 0, 1)).contiguous().float() / 255.0
    return t, ratio, pad


def draw(frame, dets, min_box=0):
    for b, s, c, d, dist, cp in zip(
        dets["boxes"], dets["scores"], dets["classes"],
        dets["depth"], dets["distance"], dets["contact"],
    ):
        x, y, w, h = b
        if w < min_box or h < min_box:
            continue
        x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (int(cp[0]), int(cp[1])), 4, (0, 0, 255), -1)
        label = f"{CLASSES[int(c)]} {s:.2f} {dist:.1f}m"
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


def run_video(args, model, device):
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {args.source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    tracker = SimpleIoUTracker()
    track_dists = defaultdict(list)
    writer = None
    prev_t = time.time()
    paused = False

    print(f"Video: {args.source}  ({src_fps:.1f} fps). Nhấn SPACE tạm dừng, Q/ESC thoát.")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break

            t, ratio, pad = preprocess(frame, args.img_size)
            with torch.no_grad():
                preds = model(t.unsqueeze(0).to(device))
                dets = decode_predictions(
                    preds, args.num_classes,
                    obj_thr=args.obj_thr, iou_thr=args.iou_thr,
                )[0]

            if len(dets["boxes"]) > 0:
                boxes    = unletterbox_boxes(dets["boxes"].cpu().numpy(), ratio, pad)
                contacts = unletterbox_points(dets["contact"].cpu().numpy(), ratio, pad)
                dets_orig = {
                    "boxes":    boxes,
                    "scores":   dets["scores"].cpu().numpy(),
                    "classes":  dets["classes"].cpu().numpy(),
                    "depth":    dets["depth"].cpu().numpy(),
                    "distance": dets["distance"].cpu().numpy(),
                    "contact":  contacts,
                }
                ids = tracker.update(boxes)
                for tid, dist in zip(ids, dets_orig["distance"]):
                    track_dists[tid].append(float(dist))
                frame = draw(frame, dets_orig, min_box=args.min_box)

            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            cv2.putText(frame, f"FPS {fps:.1f}", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            if args.save_video:
                if writer is None:
                    h, w = frame.shape[:2]
                    os.makedirs(args.out, exist_ok=True)
                    out_name = os.path.splitext(os.path.basename(args.source))[0] + "_out.mp4"
                    path = os.path.join(args.out, out_name)
                    writer = cv2.VideoWriter(
                        path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
                    print(f"Saving to {path}")
                writer.write(frame)

            disp = cv2.resize(frame, (0, 0), fx=args.display_scale, fy=args.display_scale) \
                if args.display_scale != 1.0 else frame
            cv2.imshow("OGCDE", disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key == ord(" "):
            paused = not paused

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    stab = stability(track_dists)
    print(f"Stability (mean per-track variance): {stab:.4f}")


def run_camera(args, model, device):
    cam_id = args.camera if args.camera is not None else 0
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được camera {cam_id}")

    tracker = SimpleIoUTracker()
    track_dists = defaultdict(list)
    writer = None
    prev_t = time.time()

    print(f"Camera {cam_id} đang chạy. Nhấn Q hoặc ESC để thoát.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Mất tín hiệu camera.")
            break

        t, ratio, pad = preprocess(frame, args.img_size)
        with torch.no_grad():
            preds = model(t.unsqueeze(0).to(device))
            dets = decode_predictions(
                preds, args.num_classes,
                obj_thr=args.obj_thr, iou_thr=args.iou_thr,
            )[0]

        if len(dets["boxes"]) > 0:
            boxes    = unletterbox_boxes(dets["boxes"].cpu().numpy(), ratio, pad)
            contacts = unletterbox_points(dets["contact"].cpu().numpy(), ratio, pad)
            dets_orig = {
                "boxes":    boxes,
                "scores":   dets["scores"].cpu().numpy(),
                "classes":  dets["classes"].cpu().numpy(),
                "depth":    dets["depth"].cpu().numpy(),
                "distance": dets["distance"].cpu().numpy(),
                "contact":  contacts,
            }
            ids = tracker.update(boxes)
            for tid, dist in zip(ids, dets_orig["distance"]):
                track_dists[tid].append(float(dist))
            frame = draw(frame, dets_orig, min_box=args.min_box)

        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        cv2.putText(frame, f"FPS {fps:.1f}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        if args.save_video:
            if writer is None:
                h, w = frame.shape[:2]
                os.makedirs(args.out, exist_ok=True)
                path = os.path.join(args.out, "camera_out.mp4")
                writer = cv2.VideoWriter(
                    path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
            writer.write(frame)

        disp = cv2.resize(frame, (0, 0), fx=args.display_scale, fy=args.display_scale) \
            if args.display_scale != 1.0 else frame
        cv2.imshow("OGCDE", disp)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    stab = stability(track_dists)
    print(f"Stability (mean per-track variance): {stab:.4f}")


def main():
    args = parse_args()

    if args.camera is None and args.source is None:
        raise ValueError("Cần chỉ định --source (video/folder) hoặc --camera [index]")

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    backbone_size = ckpt.get("args", {}).get("backbone_size", "n")
    model = OGCDENet(nc=args.num_classes, backbone_size=backbone_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {args.ckpt}  (epoch {ckpt.get('epoch','?')})")

    if args.camera is not None:
        run_camera(args, model, device)
        return

    source = args.source
    is_video = not os.path.isdir(source) and source.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"))
    if is_video:
        run_video(args, model, device)
        return

    # image folder — batch inference, no live window
    os.makedirs(args.out, exist_ok=True)
    tracker = SimpleIoUTracker()
    track_dists = defaultdict(list)

    for frame, name in iter_frames(source):
        t, ratio, pad = preprocess(frame, args.img_size)
        with torch.no_grad():
            preds = model(t.unsqueeze(0).to(device))
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
            frame = draw(frame, dets_orig, min_box=args.min_box)

        out_path = os.path.join(args.out, os.path.basename(name))
        cv2.imwrite(out_path, frame)

    stab = stability(track_dists)
    print(f"Stability (mean per-track variance of distance): {stab:.4f}")


if __name__ == "__main__":
    main()
