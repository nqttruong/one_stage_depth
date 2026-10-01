from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    """A lightweight placeholder for the detector-specific loss.

    The current repo preserves a monolithic detection loss path, but the new
    architecture keeps the detector and distance losses separate by contract.
    """

    def __init__(self, weight_box=5.0, weight_obj=1.0, weight_cls=0.5):
        super().__init__()
        self.weight_box = weight_box
        self.weight_obj = weight_obj
        self.weight_cls = weight_cls

    def forward(self, pred_det, target_det):
        if pred_det is None or target_det is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        loss = self.weight_box * F.mse_loss(pred_det, target_det)
        loss = loss + self.weight_obj * F.mse_loss(pred_det, target_det)
        loss = loss + self.weight_cls * F.mse_loss(pred_det, target_det)
        return loss
