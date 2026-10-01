from __future__ import annotations

from typing import Any, Dict, List

import torch


class FCOSTargetAdapter:
    """Convert the canonical physical target to FCOS-style per-image targets."""

    def __call__(self, targets):
        if isinstance(targets, dict):
            targets = [targets]
        converted = []
        for target in targets:
            entry = {
                "boxes": target["boxes"],
                "labels": target["labels"],
            }
            converted.append(entry)
        return converted
