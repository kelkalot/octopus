"""Figure 2 — Gemma matrix-level replication.

Two panels:
- (a) Joint-condition damage curve. Control regex-degen rate vs steering coef
  for joint {#3997, #13700, #11444} vs single #3997 (matched scalar coefficient).
  Shows the §4.2-on-Gemma finding: at moderate amplification (c=+200) joint
  damages controls 58% where single leaves them at 0%.
- (b) Matched-geometry CI separation. Bar chart with Wilson 95% CIs for joint
  c=+200 (perturbation magnitude 345 via joint sum-norm 1.724) vs random
  unit-direction at c=+345 (matched magnitude). The §4.3-on-Gemma test.

Output: figures/figure2_gemma_replication.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from src.detectors import is_degenerate as is_degen_canonical
    from src.detectors import wilson_ci
except ImportError:  # invoked as `python src/plot_figure2.py`
    from detectors import is_degenerate as is_degen_canonical
    from detectors import wilson_ci

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
    "grid.linewidth": 0.4,
    "grid.color": "#dddddd",
    "axes.edgecolor": "#444444",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "figure.dpi": 160,
})

C_JOINT = "#9c2b3d"   # joint suppression / amplification (deep red)
C_SINGLE = "#1f6f8b"  # single-feature comparator (steel blue)
C_RAND = "#5f7d4f"    # random direction (green)
C_GREY = "#888888"

DATA = Path("data/interventions")

CONTROLS = {"Explain how a car engine works.",
            "Write a recipe for tomato soup.",
            "Describe the steps to change a flat tyre."}


def panel_a_joint_damage(ax) -> None:
    """Per-coef control degen rate for joint vs single on Gemma."""
    joint = json.loads((DATA / "gemma_joint_3997_13700_11444.json").read_text())
    single = json.loads((DATA / "gemma_feat3997_narrow.json").read_text())

    coefs = sorted(set(r["coefficient"] for r in joint))

    def rate(records, c):
        bucket = [r for r in records if r["coefficient"] == c and r["prompt"] in CONTROLS]
        if not bucket: return 0.0
        return sum(is_degen_canonical(r["completion"]) for r in bucket) / len(bucket)

    joint_rates = [rate(joint, c) * 100 for c in coefs]
    single_rates = [rate(single, c) * 100 for c in coefs]

    ax.plot(coefs, single_rates, "o-", color=C_SINGLE, lw=1.6, ms=5,
            label=r"single \#3997")
    ax.plot(coefs, joint_rates, "s-", color=C_JOINT, lw=1.6, ms=5,
            label=r"joint $\{$\#3997, \#13700, \#11444$\}$")

    ax.set_xlabel("steering coefficient $c$")
    ax.set_ylabel("control-prompt regex-degen rate (%)")
    ax.set_title("(a) joint vs single on Gemma controls",
                 loc="left", pad=6)
    ax.set_ylim(-5, 110)
    ax.set_xlim(-450, 450)
    ax.axvline(0, color="#bbbbbb", lw=0.5)
    ax.grid(True, axis="y", alpha=0.6)
    ax.legend(loc="upper center", frameon=False, bbox_to_anchor=(0.5, -0.22),
              ncol=1, handlelength=2)

    diag_x, diag_y = 200, 58.3
    ax.annotate(f"joint $c{{=}}{{+}}200$:\n$58\\%$ damage\n(single: $0\\%$)",
                xy=(diag_x, diag_y), xytext=(60, 90),
                fontsize=7, color=C_JOINT,
                arrowprops=dict(arrowstyle="->", color=C_JOINT, lw=0.5))


def panel_b_matched_geometry(ax) -> None:
    """Bar chart: joint c=+200 vs random c=+345 with Wilson 95% CIs."""
    joint = json.loads((DATA / "gemma_joint_3997_13700_11444.json").read_text())
    rd = json.loads((DATA / "gemma_random_direction.json").read_text())

    joint_pos = [r for r in joint if r["coefficient"] == 200.0
                 and r["prompt"] in CONTROLS]
    rand_pos = [r for r in rd if r["coefficient"] == 345.0
                and r["prompt"] in CONTROLS]

    j_k = sum(is_degen_canonical(r["completion"]) for r in joint_pos)
    r_k = sum(is_degen_canonical(r["completion"]) for r in rand_pos)
    j_n, r_n = len(joint_pos), len(rand_pos)
    j_p = j_k / j_n
    r_p = r_k / r_n
    j_lo, j_hi = wilson_ci(j_k, j_n)
    r_lo, r_hi = wilson_ci(r_k, r_n)

    labels = ["random direction\n$c{=}+345$\n($K{=}5$, $n{=}120$)",
             "joint amplification\n$c{=}+200$\n($n{=}36$)"]
    means = [r_p * 100, j_p * 100]
    err_lo = [(r_p - r_lo) * 100, (j_p - j_lo) * 100]
    err_hi = [(r_hi - r_p) * 100, (j_hi - j_p) * 100]
    colors = [C_RAND, C_JOINT]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=[err_lo, err_hi], color=colors,
                  width=0.45, capsize=6, alpha=0.92,
                  error_kw={"elinewidth": 1.2, "ecolor": "#444444"})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("control-prompt regex-degen rate (%)")
    ax.set_title("(b) matched-geometry control on Gemma",
                 loc="left", pad=6)
    ax.set_ylim(0, 85)
    ax.grid(True, axis="y", alpha=0.6)

    # Value labels on top of each bar/CI
    for xi, m, hi in zip(x, means, err_hi):
        ax.text(xi, m + hi + 2, f"{m:.1f}%",
                ha="center", va="bottom", fontsize=8, color="#222222")

    # Annotate the CI gap
    gap_y = (j_lo * 100 + r_hi * 100) / 2
    ax.annotate("", xy=(0, r_hi * 100), xytext=(0, j_lo * 100),
                arrowprops=dict(arrowstyle="-", color=C_GREY, lw=0.6, ls="--"))
    ax.text(0.5, gap_y, "$95\\%$ CIs separated\n(gap $\\approx\\!10\\times$)",
            ha="center", va="center", fontsize=7.5, color=C_GREY,
            rotation=0,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#cccccc", lw=0.5))


def main() -> None:
    fig = plt.figure(figsize=(9.5, 4.0))
    gs = fig.add_gridspec(1, 2, wspace=0.35,
                          left=0.07, right=0.98,
                          top=0.90, bottom=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    panel_a_joint_damage(ax_a)
    panel_b_matched_geometry(ax_b)

    out_pdf = Path("paper/figures/figure2_gemma_replication.pdf")
    out_png = Path("paper/figures/figure2_gemma_replication.png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    print(f"[plot] {out_pdf}")
    print(f"[plot] {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
