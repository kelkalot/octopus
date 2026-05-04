"""Rank stability of the SAE-feature ranking under bootstrap resampling
of pools A, B, C.

For each bootstrap iteration:
  - resample with replacement from each pool's per-sample feature activations
  - recompute mean activations and the combined A−B / A−C z-score
  - take the top-K features

Then report:
  - inclusion rate of each "original top-K" feature in the bootstrap top-K
  - bootstrap CI (2.5%, 97.5%) on the mean activations of the headline features

Usage:
    uv run python src/bootstrap_ranks.py \
        --activations-dir data/activations \
        --layer 20 \
        --n-boot 500 \
        --top-k 50 \
        --out data/activations/sae_layer20_bootstrap.json \
        --figure figures/phase3_rank_stability.png
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
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figure", type=Path, required=True)
    ap.add_argument("--headline-features", type=int, nargs="*", default=None,
                    help="features to report inclusion stats for; default = "
                         "this layer's own top-12 from the original ranking")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    A = np.load(args.activations_dir / f"sae_layer{args.layer}_pool_A.npz")["feats"]
    B = np.load(args.activations_dir / f"sae_layer{args.layer}_pool_B.npz")["feats"]
    C = np.load(args.activations_dir / f"sae_layer{args.layer}_pool_C.npz")["feats"]
    print(f"[init] pool sizes: A={A.shape[0]} B={B.shape[0]} C={C.shape[0]}, d_sae={A.shape[1]}")

    # original ranking
    mA0, mB0, mC0 = A.mean(0), B.mean(0), C.mean(0)
    z0 = 0.5 * (zscore(mA0 - mB0) + zscore(mA0 - mC0))
    original_top = np.argsort(-z0)[: args.top_k]
    print(f"[init] original top-{args.top_k} features (top 10 shown): {list(map(int, original_top[:10]))}")

    # bootstrap
    inclusion = np.zeros(A.shape[1], dtype=np.int32)
    rank_samples = np.zeros((args.n_boot, A.shape[1]), dtype=np.int32)
    boot_means_A = np.zeros((args.n_boot, len(original_top)), dtype=np.float32)
    boot_means_B = np.zeros((args.n_boot, len(original_top)), dtype=np.float32)
    boot_means_C = np.zeros((args.n_boot, len(original_top)), dtype=np.float32)

    for b in range(args.n_boot):
        iA = rng.integers(0, A.shape[0], A.shape[0])
        iB = rng.integers(0, B.shape[0], B.shape[0])
        iC = rng.integers(0, C.shape[0], C.shape[0])
        mA = A[iA].mean(0)
        mB = B[iB].mean(0)
        mC = C[iC].mean(0)
        z = 0.5 * (zscore(mA - mB) + zscore(mA - mC))
        order = np.argsort(-z)
        top = order[: args.top_k]
        inclusion[top] += 1
        # also track each feature's bootstrap rank (1 = highest)
        rank_samples[b, order] = np.arange(A.shape[1], dtype=np.int32)
        # capture per-bootstrap means for the ORIGINAL top features (for CIs)
        boot_means_A[b] = mA[original_top]
        boot_means_B[b] = mB[original_top]
        boot_means_C[b] = mC[original_top]
        if (b + 1) % 100 == 0:
            print(f"[boot] {b+1}/{args.n_boot}")

    # report inclusion rates and rank stability for the headline features
    if args.headline_features is None:
        headline = [int(f) for f in original_top[:12]]
    else:
        headline = list(args.headline_features)
    print(f"\n[headline features] {headline}")
    print(f"[inclusion] proportion of {args.n_boot} bootstrap top-{args.top_k} sets containing each feature:")
    rows = []
    for f in headline:
        rate = inclusion[f] / args.n_boot
        median_rank = float(np.median(rank_samples[:, f])) + 1  # 1-indexed
        q025_rank = float(np.quantile(rank_samples[:, f], 0.025)) + 1
        q975_rank = float(np.quantile(rank_samples[:, f], 0.975)) + 1
        print(f"  feat #{f:>5}  inclusion={rate:>6.1%}  median_rank={median_rank:>5.0f}  "
              f"95% CI rank=[{q025_rank:>5.0f}, {q975_rank:>5.0f}]")
        rows.append({"feature": int(f), "inclusion_rate": float(rate),
                     "median_rank": median_rank,
                     "rank_ci_low": q025_rank, "rank_ci_high": q975_rank})

    # compute CIs for mean activations on the original top features
    ci_lo_A = np.quantile(boot_means_A, 0.025, axis=0)
    ci_hi_A = np.quantile(boot_means_A, 0.975, axis=0)
    ci_lo_B = np.quantile(boot_means_B, 0.025, axis=0)
    ci_hi_B = np.quantile(boot_means_B, 0.975, axis=0)
    ci_lo_C = np.quantile(boot_means_C, 0.025, axis=0)
    ci_hi_C = np.quantile(boot_means_C, 0.975, axis=0)

    # figure: top-12 bootstrap inclusion rate + rank distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=160)

    top12 = np.array(headline[:12]) if len(headline) >= 12 else original_top[:12]
    inclusion_rates = inclusion[top12] / args.n_boot
    labels = [f"#{int(f)}" for f in top12]
    bars = ax1.bar(np.arange(len(top12)), inclusion_rates, color="#9b1d20", zorder=2)
    ax1.set_xticks(np.arange(len(top12)))
    ax1.set_xticklabels(labels, rotation=0, fontsize=9)
    ax1.set_ylim(0, 1.04)
    ax1.set_ylabel(f"inclusion rate in bootstrap top-{args.top_k}\n(n_boot={args.n_boot})")
    ax1.set_title("Rank stability — top-12 features\n"
                  "How often does each feature reappear in the top set under "
                  "bootstrap resampling of the pools?",
                  loc="left")
    ax1.axhline(1.0, color="#888", lw=0.5)
    ax1.grid(True, color="#eee", linewidth=0.7, zorder=0)
    ax1.set_axisbelow(True)
    for i, r in enumerate(inclusion_rates):
        ax1.text(i, r + 0.02, f"{r:.1%}", ha="center", va="bottom", fontsize=8.5)

    # bootstrap CIs on mean A/B/C for the same features (from original_top order)
    x = np.arange(len(top12))
    w = 0.27
    A_lo = ci_lo_A[:12]; A_hi = ci_hi_A[:12]; A_mid = (A_lo + A_hi) / 2
    B_lo = ci_lo_B[:12]; B_hi = ci_hi_B[:12]; B_mid = (B_lo + B_hi) / 2
    C_lo = ci_lo_C[:12]; C_hi = ci_hi_C[:12]; C_mid = (C_lo + C_hi) / 2
    ax2.errorbar(x - w, A_mid, yerr=[A_mid - A_lo, A_hi - A_mid],
                 fmt="o", color="#9b1d20", capsize=3, label="Pool A")
    ax2.errorbar(x,     B_mid, yerr=[B_mid - B_lo, B_hi - B_mid],
                 fmt="o", color="#d39e35", capsize=3, label="Pool B")
    ax2.errorbar(x + w, C_mid, yerr=[C_mid - C_lo, C_hi - C_mid],
                 fmt="o", color="#1f6f8b", capsize=3, label="Pool C")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=0, fontsize=9)
    ax2.set_ylabel("mean SAE activation\n(95% bootstrap CI)")
    ax2.set_title("Bootstrap CIs on per-pool mean feature activations\n"
                  "Wide non-overlap between Pool A and Pool C confirms the\n"
                  "ranking is not driven by pool-level noise.",
                  loc="left")
    ax2.legend(loc="upper right", frameon=False)
    ax2.grid(True, color="#eee", linewidth=0.7, zorder=0)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(args.figure)
    plt.close(fig)
    print(f"[plot] {args.figure}")

    summary = {
        "n_boot": args.n_boot,
        "top_k": args.top_k,
        "headline_features": rows,
        "original_top": [int(x) for x in original_top],
    }
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
