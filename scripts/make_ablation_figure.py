"""Fig. 9 — Ablation visualization.

Horizontal bar chart of AbsRel + δ<1.25 across all ablation variants,
including the stage-2 LiDAR refinement at the bottom (largest single
improvement). Supports Table I (ablation table) visually.

Variants (incremental):
  V1: baseline (det + direct depth, no CP, no sec θ, no warmup, no class-w)
  V2: + contact point
  V3: + sec(θ) head
  V4: + λ_geo warmup
  V5: + per-class weights = full OGCDE (Phase 1, annotation GT)
  V5*: + stage-2 LiDAR refinement (Phase 2 fine-tune on LiDAR GT)

Usage:
    python scripts/make_ablation_figure.py --out figs/ablation.pdf
"""

import argparse
import os

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


# (label, AbsRel %, δ<1.25 %, group, delta_from_prev_str)
VARIANTS = [
    ("V1: baseline",                       13.09, 86.96, "base",   ""),
    ("V2: + contact point",                13.58, 85.72, "comp",   "+0.49"),
    ("V3: + $\\sec\\theta$ head",          12.92, 88.27, "comp",   "−0.66"),
    ("V4: + $\\lambda_\\mathrm{geo}$ warmup", 13.87, 86.98, "comp",   "+0.95"),
    ("V5: + per-class weights (full)",     12.62, 89.72, "full",   "−1.25"),
    ("V5*: + stage-2 LiDAR refinement",     7.54, 96.74, "stage2", "−5.08"),
]
GROUP_COLOR = {
    "base":   "#888888",
    "comp":   "#0072B2",
    "full":   "#009E73",
    "stage2": "#D55E00",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/ablation.pdf")
    args = ap.parse_args()

    labels  = [v[0] for v in VARIANTS]
    absrels = [v[1] for v in VARIANTS]
    deltas  = [v[2] for v in VARIANTS]   # δ<1.25
    groups  = [v[3] for v in VARIANTS]
    dstr    = [v[4] for v in VARIANTS]
    colors  = [GROUP_COLOR[g] for g in groups]

    y = np.arange(len(VARIANTS))[::-1]   # top→bottom = V1→V5*

    fig, (ax_a, ax_d) = plt.subplots(
        1, 2, figsize=(7.16, 3.0), constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1]},
    )

    # ── (a) AbsRel ──────────────────────────────────────────────────────
    bars_a = ax_a.barh(y, absrels, color=colors,
                       edgecolor="black", linewidth=0.4, height=0.65)
    for yi, val in zip(y, absrels):
        ax_a.text(val + 0.2, yi, f"{val:.2f}%",
                  va="center", ha="left", fontsize=7.5, fontweight="bold")

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels, fontsize=8)
    ax_a.set_xlabel("AbsRel (%, lower better)", fontsize=8.5)
    ax_a.set_title("(a) Distance estimation error", fontsize=9, pad=3)
    ax_a.set_xlim(0, max(absrels) * 1.25)
    ax_a.grid(True, axis="x", linestyle=":", linewidth=0.4, alpha=0.5)
    ax_a.invert_xaxis()    # lower-better, so left is good
    # actually keep normal direction for readability
    ax_a.invert_xaxis()    # cancel — left=0, right=high
    for sp in ("top", "right"):
        ax_a.spines[sp].set_visible(False)

    # ── (b) δ<1.25 ──────────────────────────────────────────────────────
    bars_d = ax_d.barh(y, deltas, color=colors,
                       edgecolor="black", linewidth=0.4, height=0.65)
    for yi, val in zip(y, deltas):
        ax_d.text(val + 0.4, yi, f"{val:.2f}%",
                  va="center", ha="left", fontsize=7.5, fontweight="bold")

    ax_d.set_yticks(y)
    ax_d.set_yticklabels([])    # share with left panel
    ax_d.set_xlabel("$\\delta{<}1.25$ (%, higher better)", fontsize=8.5)
    ax_d.set_title("(b) Threshold accuracy", fontsize=9, pad=3)
    ax_d.set_xlim(82, 100)
    ax_d.grid(True, axis="x", linestyle=":", linewidth=0.4, alpha=0.5)
    for sp in ("top", "right"):
        ax_d.spines[sp].set_visible(False)

    # ── legend (group colour key) ───────────────────────────────────────
    handles = [
        plt.Rectangle((0,0), 1, 1, color=GROUP_COLOR["base"],   label="Baseline"),
        plt.Rectangle((0,0), 1, 1, color=GROUP_COLOR["comp"],   label="Component"),
        plt.Rectangle((0,0), 1, 1, color=GROUP_COLOR["full"],   label="Full Phase 1"),
        plt.Rectangle((0,0), 1, 1, color=GROUP_COLOR["stage2"], label="+ Stage-2 LiDAR"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.12),
               handlelength=1.2, handletextpad=0.4, columnspacing=1.5)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", format="pdf")
    png = args.out.replace(".pdf", ".png")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.out}  ({os.path.getsize(args.out)/1e6:.2f} MB)")
    print(f"Preview → {png}")


if __name__ == "__main__":
    main()
