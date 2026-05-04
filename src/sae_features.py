"""Phase 3: rank SAE features that distinguish Pool A from Pools B and C.

For each sample in pools A/B/C, forward the chat-formatted prompt+completion
through the instruct model with a hook on the layer-N residual stream,
encode through the Qwen-Scope SAE, and store the mean activation over
completion tokens (32k-dim vector per sample).

Then rank features by per-pool mean differences:
    score_AB[i] = mean_A[i] - mean_B[i]   (cluster vs no-cluster intro)
    score_AC[i] = mean_A[i] - mean_C[i]   (cluster vs control)
A feature is "introspective-cluster-specific" if it ranks high on both.

Caveat: the SAE was trained on the BASE model. Applying to instruct-model
activations is a known approximation. Layer-20 reconstruction MSE/total ~20%
in the smoke test — adequate for ranking, not for full reconstruction.

Usage:
    uv run python -m src.sae_features \
        --pools-dir data/pools \
        --out-dir data/activations \
        --model Qwen/Qwen3-1.7B \
        --release qwen-scope-3-1.7b-base-w32k-l50 \
        --layer 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sae_lens import SAE
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_for_forward(tokenizer, prompt: str, completion: str) -> tuple[str, int]:
    """Return (full_text, prompt_len_tokens). Activations at positions
    [prompt_len_tokens:] correspond to the completion."""
    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full = prompt_only + completion
    prompt_ids = tokenizer(prompt_only, return_tensors="pt").input_ids
    return full, prompt_ids.shape[1]


def process_pool(
    name: str,
    samples: list[dict],
    model,
    tokenizer,
    sae: SAE,
    layer: int,
    device: str,
    out_dir: Path,
    max_length: int,
) -> np.ndarray:
    """Forward each sample, capture layer activations, encode via SAE, store
    completion-mean per sample. Returns (n_samples, d_sae) numpy array."""
    n = len(samples)
    d_sae = sae.cfg.d_sae
    feats = np.zeros((n, d_sae), dtype=np.float32)
    completion_token_counts = np.zeros(n, dtype=np.int32)

    captured: dict[str, torch.Tensor] = {}
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["x"] = h.detach()

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        t0 = time.time()
        for i, rec in enumerate(tqdm(samples, desc=f"pool {name}")):
            full, prompt_len = format_for_forward(tokenizer, rec["prompt"], rec["completion"])
            inp = tokenizer(full, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            total_len = inp["input_ids"].shape[1]
            if total_len <= prompt_len:
                continue  # nothing to measure
            with torch.no_grad():
                _ = model(**inp)
            x = captured["x"].squeeze(0)[prompt_len:total_len].to(torch.float32)
            with torch.no_grad():
                f = sae.encode(x.to(sae.dtype))
            feats[i] = f.mean(0).to(torch.float32).cpu().numpy()
            completion_token_counts[i] = total_len - prompt_len
        dt = time.time() - t0
        print(f"[pool {name}] {n} samples in {dt:.1f}s ({dt/n*1000:.0f} ms/sample)")
    finally:
        handle.remove()

    suffix = process_pool.suffix if hasattr(process_pool, "suffix") else f"sae_layer{layer}"
    out = out_dir / f"{suffix}_pool_{name}.npz"
    np.savez_compressed(
        out,
        feats=feats,
        completion_token_counts=completion_token_counts,
        sample_indices=np.array([s.get("sample_idx", i) for i, s in enumerate(samples)], dtype=np.int32),
    )
    print(f"[pool {name}] saved -> {out}")
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--release", default="qwen-scope-3-1.7b-base-w32k-l50")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--sae-id", default=None,
                    help="override sae_id (default: f'layer{args.layer}'). "
                         "Useful for Gemma Scope: 'layer_20/width_16k/canonical'.")
    ap.add_argument("--out-name", default=None,
                    help="filename suffix; default uses 'sae_layer{L}'")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=50,
                    help="number of top features to save in the ranking JSON")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap samples per pool — for quick runs")
    args = ap.parse_args()

    device = pick_device()
    sae_id = args.sae_id if args.sae_id else f"layer{args.layer}"
    print(f"[init] device={device} model={args.model} release={args.release} sae_id={sae_id}")

    print(f"[init] loading SAE")
    sae = SAE.from_pretrained(release=args.release, sae_id=sae_id, device=device)
    print(f"[init] SAE d_in={sae.cfg.d_in} d_sae={sae.cfg.d_sae} dtype={sae.cfg.dtype}")

    print(f"[init] loading tokenizer + model")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, trust_remote_code=True
    ).to(device)
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_suffix = args.out_name if args.out_name else f"sae_layer{args.layer}"
    process_pool.suffix = out_suffix

    pool_feats = {}
    for name in ("A", "B", "C"):
        path = args.pools_dir / f"pool_{name}.json"
        samples = json.loads(path.read_text(encoding="utf-8"))
        if args.limit is not None:
            samples = samples[: args.limit]
        print(f"[pool {name}] loaded {len(samples)} samples from {path.name}")
        pool_feats[name] = process_pool(
            name, samples, model, tokenizer, sae, args.layer, device, args.out_dir, args.max_length,
        )

    # rank features by mean differences
    mean_A = pool_feats["A"].mean(0)
    mean_B = pool_feats["B"].mean(0)
    mean_C = pool_feats["C"].mean(0)

    diff_AB = mean_A - mean_B
    diff_AC = mean_A - mean_C
    # combined score: mean of normalised AB and AC, both must be positive
    def zscore(v):
        s = v.std() + 1e-9
        return (v - v.mean()) / s
    combined = 0.5 * (zscore(diff_AB) + zscore(diff_AC))

    top_idx = np.argsort(-combined)[: args.top_k]

    candidates = []
    for rank, idx in enumerate(top_idx):
        candidates.append({
            "rank": rank,
            "feature_idx": int(idx),
            "mean_A": float(mean_A[idx]),
            "mean_B": float(mean_B[idx]),
            "mean_C": float(mean_C[idx]),
            "diff_AB": float(diff_AB[idx]),
            "diff_AC": float(diff_AC[idx]),
            "combined_z": float(combined[idx]),
        })

    summary = {
        "model": args.model,
        "release": args.release,
        "layer": args.layer,
        "pool_sizes": {k: int(v.shape[0]) for k, v in pool_feats.items()},
        "top_features": candidates,
    }
    out_json = args.out_dir / f"{out_suffix}_top_features.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {out_json}")
    print("\n[top 20 features by combined A-B/A-C z-score]")
    print(f"  {'rank':>4} {'feat':>6} {'meanA':>8} {'meanB':>8} {'meanC':>8} {'A-B':>8} {'A-C':>8} {'z':>6}")
    for c in candidates[:20]:
        print(
            f"  {c['rank']:>4} {c['feature_idx']:>6} "
            f"{c['mean_A']:>8.3f} {c['mean_B']:>8.3f} {c['mean_C']:>8.3f} "
            f"{c['diff_AB']:>8.3f} {c['diff_AC']:>8.3f} {c['combined_z']:>6.2f}"
        )


if __name__ == "__main__":
    main()
