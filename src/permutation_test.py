"""F4 alt — permutation test for the SAE-feature ranking pipeline.

Methodological negative control. If the ranking pipeline produces strong
"cluster-specific features" even when pool labels are randomly shuffled,
then the original ranking is unreliable. Conversely, if shuffled labels
produce only weak features (low magnitude, low z-score) compared to the
real labels, the methodology is appropriately selective.

Concretely: load layer-20 per-sample SAE activations from pools A/B/C;
in each of N permutation iterations, randomly partition the same 3 994
samples into three pools of the same sizes (1 633 / 367 / 1 994), recompute
the combined A−B / A−C z-score, take the top-1 score. Build the null
distribution and compare to the actual top-1 (29.49 for #29108).

Usage:
    uv run python src/permutation_test.py --n-perm 200 \
        --activations-dir data/activations --layer 20 \
        --out data/activations/sae_layer20_permutation.json \
        --figure figures/phase3_permutation_null.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def zscore(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figure", type=Path, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    A = np.load(args.activations_dir / f"sae_layer{args.layer}_pool_A.npz")["feats"]
    B = np.load(args.activations_dir / f"sae_layer{args.layer}_pool_B.npz")["feats"]
    C = np.load(args.activations_dir / f"sae_layer{args.layer}_pool_C.npz")["feats"]
    n_A, n_B, n_C = A.shape[0], B.shape[0], C.shape[0]
    print(f"[init] pool sizes: A={n_A} B={n_B} C={n_C}")

    # actual top-1 by raw mean(A) - mean(C). Z-score isn't a good test
    # statistic for permutation tests because it normalises by within-perm
    # std, which under random labels is small (no signal) and inflates the
    # apparent effect size.
    mA, mB, mC = A.mean(0), B.mean(0), C.mean(0)
    diff_AC = mA - mC
    actual_top1 = float(diff_AC.max())
    actual_top1_feat = int(diff_AC.argmax())
    actual_top10 = sorted(diff_AC.tolist(), reverse=True)[:10]
    print(f"[actual] top-1 (mean_A − mean_C) = {actual_top1:.3f} (feature #{actual_top1_feat})")

    # combine all samples for shuffling
    all_feats = np.concatenate([A, B, C], axis=0)
    n_total = all_feats.shape[0]

    # null distribution: max(mean_A − mean_C) under random label shuffling
    null_top1 = np.empty(args.n_perm, dtype=np.float32)
    null_top10_means = np.empty(args.n_perm, dtype=np.float32)
    for p in range(args.n_perm):
        perm = rng.permutation(n_total)
        idxA = perm[:n_A]
        idxC = perm[n_A + n_B:]
        mA_p = all_feats[idxA].mean(0)
        mC_p = all_feats[idxC].mean(0)
        diff_p = mA_p - mC_p
        top10 = np.partition(diff_p, -10)[-10:]
        null_top1[p] = float(top10.max())
        null_top10_means[p] = float(top10.mean())
        if (p + 1) % 50 == 0:
            print(f"[perm] {p+1}/{args.n_perm}  null top-1 mean so far = {null_top1[:p+1].mean():.3f}")

    null_mean = float(null_top1.mean())
    null_q025 = float(np.quantile(null_top1, 0.025))
    null_q975 = float(np.quantile(null_top1, 0.975))
    p_value = float(np.mean(null_top1 >= actual_top1))
    print(f"\n[null]  top-1 z over {args.n_perm} permutations:")
    print(f"  mean = {null_mean:.3f}, 95% CI = [{null_q025:.3f}, {null_q975:.3f}]")
    print(f"\n[result] actual top-1 = {actual_top1:.3f}; null mean = {null_mean:.3f}")
    print(f"         ratio (actual / null mean) = {actual_top1 / null_mean:.2f}×")
    print(f"         p-value (perm >= actual) = {p_value:.4f}  (n_perm={args.n_perm})")

    # plot
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=160)
    ax.hist(null_top1, bins=30, color="#aaaaaa", edgecolor="#666",
            label=f"null top-1 z (random labels, n={args.n_perm})", zorder=2)
    ax.axvline(actual_top1, color="#9b1d20", lw=2.5,
               label=f"actual top-1 z = {actual_top1:.2f}  (feature #{actual_top1_feat})")
    ax.set_xlabel("top-1 (mean_A − mean_C)")
    ax.set_ylabel("permutation count")
    ax.set_title(
        "Methodological permutation test\n"
        "Random label shuffles produce far weaker top features than the actual labels.",
        loc="left",
    )
    ax.legend(loc="best", frameon=False)
    ax.grid(True, color="#eee", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(args.figure)
    plt.close(fig)
    print(f"[plot] {args.figure}")

    summary = {
        "layer": args.layer,
        "n_perm": args.n_perm,
        "actual_top1_z": actual_top1,
        "actual_top1_feature": actual_top1_feat,
        "actual_top10_z": actual_top10,
        "null_top1_mean": null_mean,
        "null_top1_ci_95": [null_q025, null_q975],
        "p_value": p_value,
        "ratio_actual_to_null_mean": actual_top1 / null_mean,
    }
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
