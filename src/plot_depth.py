"""Build the depth-claim figure for the layer sweep."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAYERS = [12, 16, 20, 24]
DATA_DIR = Path("data/activations")


def top_cluster_specific(layer: int, c_threshold: float = 1.0,
                         ab_ratio: float = 0.5) -> dict:
    """Return the highest-meanA feature that is Class-1 cluster-specific:
    meanC < c_threshold AND meanB < ab_ratio * meanA (so A is not just
    "general introspective register" — it discriminates A from B)."""
    d = json.loads((DATA_DIR / f"sae_layer{layer}_top_features.json").read_text())
    candidates = [c for c in d["top_features"]
                  if c["mean_C"] < c_threshold
                  and c["mean_B"] < ab_ratio * c["mean_A"]]
    if not candidates:
        return None
    return max(candidates, key=lambda c: c["mean_A"])


def main():
    rows = []
    for L in LAYERS:
        f = top_cluster_specific(L)
        if f is None:
            continue
        rows.append((L, f))
    print(f"{'layer':>5} {'feat':>6} {'meanA':>7} {'meanB':>7} {'meanC':>7} {'A-C':>7}")
    for L, f in rows:
        print(f"{L:>5} #{f['feature_idx']:>5} {f['mean_A']:>7.2f} {f['mean_B']:>7.2f} {f['mean_C']:>7.2f} {f['diff_AC']:>7.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=160)

    # Panel 1: magnitude growth with depth
    layers_arr = [r[0] for r in rows]
    meanA = [r[1]["mean_A"] for r in rows]
    meanB = [r[1]["mean_B"] for r in rows]
    meanC = [r[1]["mean_C"] for r in rows]
    feat_labels = [f"#{r[1]['feature_idx']}" for r in rows]

    x = np.arange(len(layers_arr))
    w = 0.27
    ax1.bar(x - w, meanA, width=w, color="#9b1d20", label="Pool A (intro w/ cluster)", zorder=2)
    ax1.bar(x,     meanB, width=w, color="#d39e35", label="Pool B (intro w/o cluster)", zorder=2)
    ax1.bar(x + w, meanC, width=w, color="#1f6f8b", label="Pool C (controls)", zorder=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"L{L}\n{lbl}" for L, lbl in zip(layers_arr, feat_labels)])
    ax1.set_ylabel("mean SAE feature activation")
    ax1.set_title("Top cluster-specific feature at each layer (meanC < 1.0)\n"
                  "Magnitude of the cluster-specific signal grows monotonically with depth.",
                  loc="left")
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(True, color="#eee", linewidth=0.7, zorder=0)
    ax1.set_axisbelow(True)
    for i, (a, c) in enumerate(zip(meanA, meanC)):
        ax1.text(i - w, a + max(meanA) * 0.02, f"{a:.1f}", ha="center", fontsize=9)
    # annotation: A-C label
    for i, r in enumerate(rows):
        ax1.text(i + w + 0.05, max(meanA) * 0.7,
                 f"A−C={r[1]['diff_AC']:.1f}",
                 ha="left", va="center", fontsize=8, color="#444")

    # Panel 2: count of cluster-specific features per layer (with thresholds)
    counts_strict = []
    counts_loose  = []
    for L in LAYERS:
        d = json.loads((DATA_DIR / f"sae_layer{L}_top_features.json").read_text())
        # Class-1 (cluster-specific): meanC small AND meanB < 0.5 * meanA
        strict = sum(1 for c in d["top_features"]
                     if c["mean_A"] >= 5 and c["mean_C"] < 0.5
                     and c["mean_B"] < 0.5 * c["mean_A"])
        loose  = sum(1 for c in d["top_features"]
                     if c["mean_A"] >= 1 and c["mean_C"] < 0.5
                     and c["mean_B"] < 0.5 * c["mean_A"])
        counts_strict.append(strict)
        counts_loose.append(loose)

    x = np.arange(len(LAYERS))
    w = 0.35
    ax2.bar(x - w/2, counts_loose, width=w, color="#aaaaaa",
            label="meanA ≥ 1, meanC < 0.5  (any cluster-specific)", zorder=2)
    ax2.bar(x + w/2, counts_strict, width=w, color="#9b1d20",
            label="meanA ≥ 5, meanC < 0.5  (strong cluster-specific)", zorder=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"L{L}" for L in LAYERS])
    ax2.set_ylabel("count of features in top-50 ranking")
    ax2.set_title("Cluster-specific feature density vs depth\n"
                  "Both 'any' and 'strong' cluster-specific feature counts increase with depth.",
                  loc="left")
    ax2.legend(loc="upper left", frameon=False)
    ax2.grid(True, color="#eee", linewidth=0.7, zorder=0)
    ax2.set_axisbelow(True)
    for i, (s, l) in enumerate(zip(counts_strict, counts_loose)):
        ax2.text(i - w/2, l + 0.5, f"{l}", ha="center", fontsize=9)
        ax2.text(i + w/2, s + 0.5, f"{s}", ha="center", fontsize=9, color="#9b1d20")

    fig.suptitle("Depth claim — where does the philosophy-of-mind register live in Qwen3-1.7B?",
                 x=0.02, ha="left", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    out = Path("figures/phase3_depth_claim.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
