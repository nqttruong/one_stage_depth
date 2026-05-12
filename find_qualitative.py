"""Scan KITTI val split and select 5 qualitative figure candidates.

Scenarios:
  1. easy       — 2-6 cars, all 10-30m, straight road, no peds/cyclists
  2. far_range  — at least 1 car at 50-70m, model must predict it
  3. oblique    — car at image edge with large lateral angle (|x/z| > 0.45)
  4. crowded    — ≥2 classes present, ≥5 total objects
  5. failure    — object with occlusion=2 (largely occluded) → model likely misses

Output: prints image IDs + GT stats for review.

Usage:
    python find_qualitative.py \\
        --kitti-root /data/kitti/training \\
        --split splits/distformer_val.txt
"""

import argparse
import math
import os

from ogcde.dataset import parse_kitti_label

CLASS_MAP = {"Car": 0, "Van": 0, "Truck": 0,
             "Pedestrian": 1, "Person_sitting": 1, "Cyclist": 2}


def score_image(objects: list) -> dict:
    cars = [o for o in objects if o["type"] in {"Car", "Van", "Truck"}]
    peds = [o for o in objects if o["type"] in {"Pedestrian", "Person_sitting"}]
    cycs = [o for o in objects if o["type"] == "Cyclist"]
    all_obj = cars + peds + cycs

    # distances
    dists = [math.sqrt(sum(x**2 for x in o["loc"])) for o in all_obj]
    car_dists = [math.sqrt(sum(x**2 for x in o["loc"])) for o in cars]
    # lateral angles for cars: arctan(|x|/z)
    car_angles = [abs(math.atan2(abs(o["loc"][0]), max(o["loc"][2], 1e-3)))
                  for o in cars]

    n_classes = sum([len(cars) > 0, len(peds) > 0, len(cycs) > 0])

    # scenario scores
    scores = {}

    # 1. Easy
    if (2 <= len(cars) <= 8 and len(peds) == 0 and len(cycs) == 0
            and car_dists and all(10 <= d <= 35 for d in car_dists)):
        scores["easy"] = 10 - abs(len(cars) - 4)  # prefer ~4 cars
    else:
        scores["easy"] = -1

    # 2. Far-range: car at 50-70m
    far_cars = [d for d in car_dists if 45 <= d <= 80]
    scores["far_range"] = max(far_cars) if far_cars else -1

    # 3. Oblique: car at large angle
    scores["oblique"] = max(car_angles) if car_angles else -1

    # 4. Crowded: mix of classes
    if n_classes >= 2 and len(all_obj) >= 5:
        scores["crowded"] = len(all_obj) + n_classes * 2
    else:
        scores["crowded"] = -1

    # 5. Failure: occluded=2 (largely occluded) objects
    n_heavy_occ = sum(1 for o in all_obj if o.get("occluded", 0) >= 2)
    scores["failure"] = n_heavy_occ if n_heavy_occ > 0 else -1

    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--split",      required=True)
    ap.add_argument("--top",        type=int, default=5,
                    help="Print top-N candidates per scenario")
    args = ap.parse_args()

    label_dir = os.path.join(args.kitti_root, "label_2")
    ids = [l.strip() for l in open(args.split) if l.strip()]

    buckets = {s: [] for s in ["easy", "far_range", "oblique", "crowded", "failure"]}

    for img_id in ids:
        path = os.path.join(label_dir, f"{img_id}.txt")
        objects = [o for o in parse_kitti_label(path) if o["type"] != "DontCare"]
        if not objects:
            continue
        s = score_image(objects)
        for scenario, score in s.items():
            if score > 0:
                cars   = [o for o in objects if o["type"] in {"Car","Van","Truck"}]
                peds   = [o for o in objects if o["type"] in {"Pedestrian","Person_sitting"}]
                cycs   = [o for o in objects if o["type"] == "Cyclist"]
                all_o  = cars + peds + cycs
                dists  = [math.sqrt(sum(x**2 for x in o["loc"])) for o in all_o]
                occ2   = sum(1 for o in all_o if o.get("occluded",0) >= 2)
                buckets[scenario].append({
                    "id": img_id, "score": score,
                    "n_car": len(cars), "n_ped": len(peds), "n_cyc": len(cycs),
                    "dists": [round(d, 1) for d in sorted(dists)],
                    "occ2": occ2,
                })

    for scenario, items in buckets.items():
        top = sorted(items, key=lambda x: -x["score"])[:args.top]
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario.upper()}")
        print(f"{'='*60}")
        for x in top:
            print(f"  ID={x['id']}  score={x['score']:.2f}  "
                  f"car={x['n_car']} ped={x['n_ped']} cyc={x['n_cyc']}  "
                  f"occ2={x['occ2']}  dists={x['dists']}")


if __name__ == "__main__":
    main()
