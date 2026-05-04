"""Phase 3b: interpret top SAE features by their maximally-activating samples.

For each of the top-K candidate features, look up the samples in Pool A
(plus optionally B and C) whose mean SAE activation on that feature is
highest, and show the prompt + completion. If the top activating samples
share a coherent theme, the feature is interpretable from internal data
alone — no need for an external corpus pass.

Usage:
    uv run python -m src.interpret \
        --activations-dir data/activations \
        --pools-dir data/pools \
        --layer 20 \
        --top-features 12 \
        --top-samples 5 \
        --out data/activations/sae_layer20_interpretations.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_pool(pools_dir: Path, name: str) -> list[dict]:
    return json.loads((pools_dir / f"pool_{name}.json").read_text(encoding="utf-8"))


def load_feats(activations_dir: Path, prefix: str, name: str) -> np.ndarray:
    npz = np.load(activations_dir / f"{prefix}_pool_{name}.npz")
    return npz["feats"]


def render_completion(text: str, max_chars: int) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations-dir", type=Path, required=True)
    ap.add_argument("--pools-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--name", default=None,
                    help="filename prefix; default 'sae_layer{L}'")
    ap.add_argument("--top-features", type=int, default=12)
    ap.add_argument("--top-samples", type=int, default=5)
    ap.add_argument("--snippet-chars", type=int, default=420)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    prefix = args.name if args.name else f"sae_layer{args.layer}"
    summary = json.loads(
        (args.activations_dir / f"{prefix}_top_features.json")
        .read_text(encoding="utf-8")
    )
    top = summary["top_features"][: args.top_features]

    pools = {n: load_pool(args.pools_dir, n) for n in ("A", "B", "C")}
    feats = {n: load_feats(args.activations_dir, prefix, n) for n in ("A", "B", "C")}

    lines: list[str] = []
    lines.append(f"# SAE feature interpretations — layer {args.layer}")
    lines.append("")
    lines.append(f"Source: `{args.activations_dir}/sae_layer{args.layer}_top_features.json`")
    lines.append(f"Pools: A={len(pools['A'])} (intro w/ cluster), "
                 f"B={len(pools['B'])} (intro w/o cluster), "
                 f"C={len(pools['C'])} (control).")
    lines.append("")

    for cand in top:
        idx = cand["feature_idx"]
        rank = cand["rank"]
        lines.append(f"## rank {rank}: feature #{idx}")
        lines.append("")
        lines.append(
            f"mean_A={cand['mean_A']:.2f}, mean_B={cand['mean_B']:.2f}, "
            f"mean_C={cand['mean_C']:.2f}, A-B={cand['diff_AB']:.2f}, "
            f"A-C={cand['diff_AC']:.2f}, z={cand['combined_z']:.2f}"
        )
        lines.append("")

        # top activating samples in each pool
        for pool_name in ("A", "B", "C"):
            arr = feats[pool_name][:, idx]
            order = np.argsort(-arr)[: args.top_samples]
            lines.append(f"**Top {args.top_samples} activating samples in pool {pool_name}** "
                         f"(per-sample mean activation on feature #{idx}):")
            lines.append("")
            for j, sample_i in enumerate(order):
                v = float(arr[sample_i])
                if v <= 0:
                    lines.append(f"{j+1}. _(activation={v:.3f}, no signal)_")
                    continue
                rec = pools[pool_name][int(sample_i)]
                snippet = render_completion(rec["completion"], args.snippet_chars)
                lines.append(
                    f"{j+1}. _act={v:.3f}_ — **prompt:** {rec['prompt']}\n\n"
                    f"   {snippet}"
                )
            lines.append("")
        lines.append("---")
        lines.append("")

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote {args.out}")
    print(f"[note] open in your editor or markdown viewer to skim the {len(top)} feature interpretations")


if __name__ == "__main__":
    main()
