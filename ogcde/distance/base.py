from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn


class DistanceMethod(ABC):
    """A detector-agnostic distance formulation."""

    method_name: str = "base"
    output_dim: int = 1

    @abstractmethod
    def decode(self, logits: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @abstractmethod
    def target_from_gt(self, z_gt: torch.Tensor, dist_gt: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, logits: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class DistanceHead(nn.Module):
    """Reusable dense distance head for P3/P4/P5 feature pyramids."""

    def __init__(self, in_channels: Sequence[int], method: DistanceMethod, hidden_channels: int = 128):
        super().__init__()
        self.method = method
        self.heads = nn.ModuleList()
        for ch in in_channels:
            self.heads.append(
                nn.Sequential(
                    nn.Conv2d(ch, hidden_channels, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(hidden_channels, method.output_dim, kernel_size=1),
                )
            )

    def forward(self, feats: Sequence[torch.Tensor]):
        out = []
        for feat, h in zip(feats, self.heads):
            out.append(h(feat))
        return out


class DistanceDetectionModel(nn.Module):
    """Composes a detector adapter with a distance estimator in the new refactored pipeline."""

    def __init__(self, detector: nn.Module, distance_method: DistanceMethod):
        super().__init__()
        self.detector = detector
        self.distance_method = distance_method
        self.distance_head = DistanceHead(
            tuple(int(ch) for ch in detector.feature_channels),
            method=distance_method,
        )

    def forward(self, x: torch.Tensor):
        feats = self.detector.forward_features(x)
        return self.distance_head(feats)

    def compute_loss(self, logits, targets, metas):
        if not isinstance(targets, list):
            targets = [targets]
        if not isinstance(metas, list):
            metas = [metas]

        total_loss = 0.0
        valid_count = 0
        for batch_idx, target in enumerate(targets):
            if not isinstance(target, dict) or "boxes" not in target:
                continue
            boxes = target["boxes"]
            if boxes.numel() == 0:
                continue
            if batch_idx >= len(metas):
                raise ValueError("Missing metadata for target batch entry.")
            meta = metas[batch_idx]
            intrinsics = meta.get("intrinsics_network")
            if intrinsics is None:
                raise ValueError("DistanceDetectionModel requires intrinsics_network in metadata.")

            centers = (boxes[:, :2] + boxes[:, 2:4]) / 2.0
            z_gt = target["z_gt"].to(boxes.device)
            dist_gt = target["distance_gt"].to(boxes.device)
            if z_gt.numel() != centers.shape[0] or dist_gt.numel() != centers.shape[0]:
                raise ValueError("Target length mismatch for distance supervision.")

            target_tensor = self.distance_method.target_from_gt(
                z_gt,
                dist_gt,
                intrinsics=intrinsics,
                box_center=centers,
            )
            flat_logits = []
            for feat in logits:
                flat_logits.append(feat[batch_idx].permute(1, 2, 0).reshape(-1, self.distance_method.output_dim))
            if not flat_logits:
                continue
            flat_logits = torch.cat(flat_logits, dim=0)[:max(1, target_tensor.shape[0])]
            if flat_logits.shape[-1] != target_tensor.shape[-1]:
                if flat_logits.shape[-1] == 1 and target_tensor.shape[-1] == 1:
                    flat_logits = flat_logits.reshape(-1, 1)
                else:
                    flat_logits = flat_logits[:, : target_tensor.shape[-1]]
            if flat_logits.shape[0] < target_tensor.shape[0]:
                pad = target_tensor.shape[0] - flat_logits.shape[0]
                flat_logits = torch.cat([flat_logits, flat_logits.new_zeros((pad, flat_logits.shape[-1]))], dim=0)
            loss = self.distance_method.loss(flat_logits[: target_tensor.shape[0]], target_tensor)
            total_loss = total_loss + loss
            valid_count += 1

        if valid_count == 0:
            return torch.zeros((), device=next(self.parameters()).device, requires_grad=True)
        return total_loss / max(1, valid_count)
