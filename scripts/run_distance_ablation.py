#!/usr/bin/env python3
"""Print the exact eight experiment commands for the new detector-agnostic distance matrix.

This does not fabricate results; it is simply a launch helper to keep the
experiment flow consistent and reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-root", default="configs/experiments")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = Path(args.config_root)
    configs = sorted(base.glob("*.yaml"))
    if not configs:
        raise FileNotFoundError(f"No YAML configs found in {base}")

    for cfg in configs:
        cmd = f"python train.py --config {cfg.as_posix()}"
        print(cmd)


if __name__ == "__main__":
    main()
