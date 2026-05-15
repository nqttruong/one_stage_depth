"""Fig. 5 — λ_geo warmup schedule justification.

Compares V3 (no warmup, λ_geo=2.0 from start) vs V4 (warmup over 5 epochs)
on detection loss and geometry loss curves.

Both variants share identical config except --geo-warmup-epochs (0 vs 5).
Reads training logs:
    logs/v3_sec_theta.log
    logs/v4_warmup.log

Usage:
    python scripts/make_warmup_figure.py --out figs/warmup.pdf
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     9,
    "pdf.fonttype":  42,
    "ps.fonttype":   42,
    "axes.linewidth": 0.6,
})

EPOCH_RE = re.compile(
    r"\[epoch (\d+)\] λ_geo=([\d.]+) train [\d.]+s \| "
    r"total ([\d.]+) box ([\d.]+) obj ([\d.]+) cls ([\d.]+) "
    r"d ([\d.]+) s ([\d.]+) cp ([\d.]+) geo ([\d.]+)"
)
VAL_RE = re.compile(r"\[epoch (\d+)\] val total=([\d.]+)")


def parse_log(path):
    """Return dict: epoch -> {'lambda_geo', 'box', 'obj', 'cls', 'd', 's', 'cp', 'geo'}"""
    data = {}
    if not os.path.exists(path):
        return data
    with open(path) as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if not m:
                continue
            ep = int(m.group(1))
            data[ep] = {
                "lambda_geo": float(m.group(2)),
                "total":      float(m.group(3)),
                "box":        float(m.group(4)),
                "obj":        float(m.group(5)),
                "cls":        float(m.group(6)),
                "d":          float(m.group(7)),
                "s":          float(m.group(8)),
                "cp":         float(m.group(9)),
                "geo":        float(m.group(10)),
            }
    return data


def merge_logs(*paths):
    out = {}
    for p in paths:
        out.update(parse_log(p))
    return out


def to_arrays(data, max_epoch=100):
    eps  = sorted([e for e in data.keys() if e <= max_epoch])
    return {
        "epoch":      np.array(eps),
        "lambda_geo": np.array([data[e]["lambda_geo"] for e in eps]),
        "det":        np.array([data[e]["box"] + data[e]["obj"] + data[e]["cls"] for e in eps]),
        "geo":        np.array([data[e]["geo"] for e in eps]),
        "depth":      np.array([data[e]["d"] + data[e]["s"] for e in eps]),
        "contact":    np.array([data[e]["cp"] for e in eps]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3-log",   default="logs/v3_sec_theta.log")
    ap.add_argument("--v4-log",   default="logs/v4_warmup.log")
    ap.add_argument("--max-epoch", type=int, default=100,
                    help="Plot up to this epoch (warmup effect is in first 30)")
    ap.add_argument("--out",      default="figs/warmup.pdf")
    args = ap.parse_args()

    v3 = to_arrays(parse_log(args.v3_log), args.max_epoch)
    v4 = to_arrays(parse_log(args.v4_log), args.max_epoch)

    if len(v3["epoch"]) == 0 or len(v4["epoch"]) == 0:
        raise SystemExit("Empty logs — check --v3-log and --v4-log paths")

    # ── figure: 2 subplots (det loss / geo loss) + λ schedule overlay ──────
    fig, (ax_det, ax_geo) = plt.subplots(
        1, 2, figsize=(7.16, 2.7), constrained_layout=True
    )

    # ─ (a) Detection loss ─
    ax_det.plot(v3["epoch"], v3["det"], "-", color="#D55E00", lw=1.2,
                label=r"V3: no warmup ($\lambda_\mathrm{geo}=2.0$)")
    ax_det.plot(v4["epoch"], v4["det"], "-", color="#0072B2", lw=1.2,
                label=r"V4: warmup (5 ep, 0.1$\to$2.0)")
    ax_det.axvspan(0, 5, color="#0072B2", alpha=0.10, zorder=0)
    ax_det.text(5.5, ax_det.get_ylim()[1] if False else 1.5,
                "warmup\nphase", fontsize=6.5, color="#0072B2",
                ha="left", va="top", style="italic")
    ax_det.set_xlabel("Epoch")
    ax_det.set_ylabel("Detection loss (box + obj + cls)")
    ax_det.set_title("(a) Detection loss", fontsize=9, pad=3)
    ax_det.set_xlim(0, args.max_epoch)
    ax_det.legend(loc="upper right", fontsize=7, frameon=False)
    ax_det.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    for sp in ("top", "right"):
        ax_det.spines[sp].set_visible(False)

    # ─ (b) Geometry loss + λ_geo schedule (twin axis) ─
    ax_geo.plot(v3["epoch"], v3["geo"], "-", color="#D55E00", lw=1.2,
                label="V3: no warmup")
    ax_geo.plot(v4["epoch"], v4["geo"], "-", color="#0072B2", lw=1.2,
                label="V4: warmup")
    ax_geo.axvspan(0, 5, color="#0072B2", alpha=0.10, zorder=0)
    ax_geo.set_xlabel("Epoch")
    ax_geo.set_ylabel("Geometry loss $L_\\mathrm{geo}$ (meters)", color="black")
    ax_geo.set_title(r"(b) $L_\mathrm{geo}$ and warmup schedule", fontsize=9, pad=3)
    ax_geo.set_xlim(0, args.max_epoch)
    for sp in ("top",):
        ax_geo.spines[sp].set_visible(False)

    # secondary axis: λ_geo schedule for V4
    ax_lambda = ax_geo.twinx()
    ax_lambda.plot(v4["epoch"], v4["lambda_geo"], "--", color="#666666",
                   lw=1.0, label=r"$\lambda_\mathrm{geo}$ (V4)")
    ax_lambda.axhline(2.0, color="#D55E00", lw=0.8, linestyle=":", alpha=0.5)
    ax_lambda.set_ylabel(r"$\lambda_\mathrm{geo}$", color="#666666")
    ax_lambda.set_ylim(0, 2.5)
    ax_lambda.tick_params(axis="y", labelcolor="#666666")
    for sp in ("top",):
        ax_lambda.spines[sp].set_visible(False)

    # combined legend
    lines_geo,    labels_geo    = ax_geo.get_legend_handles_labels()
    lines_lambda, labels_lambda = ax_lambda.get_legend_handles_labels()
    ax_geo.legend(lines_geo + lines_lambda, labels_geo + labels_lambda,
                  loc="upper right", fontsize=7, frameon=False)
    ax_geo.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    # ── save ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", format="pdf")
    png = args.out.replace(".pdf", ".png")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Saved → {args.out}  ({size_mb:.2f} MB)")
    print(f"Preview → {png}")

    # ── numeric summary for caption text ─────────────────────────────────
    e1_v3 = v3["det"][v3["epoch"] == 1][0] if 1 in v3["epoch"] else None
    e1_v4 = v4["det"][v4["epoch"] == 1][0] if 1 in v4["epoch"] else None
    e1_v3_geo = v3["geo"][v3["epoch"] == 1][0] if 1 in v3["epoch"] else None
    e1_v4_geo = v4["geo"][v4["epoch"] == 1][0] if 1 in v4["epoch"] else None
    print(f"\nEpoch 1:")
    print(f"  V3 det={e1_v3:.3f}  geo={e1_v3_geo:.3f}")
    print(f"  V4 det={e1_v4:.3f}  geo={e1_v4_geo:.3f}")


if __name__ == "__main__":
    main()
