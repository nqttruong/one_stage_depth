from __future__ import annotations

from typing import Any, Dict, List

import torch


def collate_distance_batch(batch: List[Any]):
    images = []
    targets = []
    metas = []
    for image, target, meta in batch:
        if isinstance(image, torch.Tensor):
            images.append(image)
        else:
            images.append(torch.from_numpy(image).permute(2, 0, 1).float())
        targets.append(target)
        metas.append(meta)
    if images:
        images = torch.stack(images, dim=0)
    else:
        images = torch.empty((0, 0, 0, 0), dtype=torch.float32)
    return {"images": images, "targets": targets, "metas": metas}
