"""Probe residual-stream norm under steering, to distinguish register-collapse
from off-manifold push.

For a (model, layer, feature_or_features) tuple and a sweep of coefs, forward
the prompt through the model with the steering hook installed, capture the
layer-L hidden state at every token position, and compute:
  - mean norm(h_steered)
  - mean norm(h_steered - h_baseline)  (perturbation magnitude)
  - mean cosine(h_steered, h_baseline) (manifold drift)

Usage:
    uv run python src/norm_probe.py \
        --model Qwen/Qwen3-1.7B \
        --release qwen-scope-3-1.7b-base-w32k-l50 \
        --layer 20 \
        --features 29108 26221 4405 \
        --coefficients -1500 -1000 -500 0 500 1000 \
        --prompt "What's a question that fascinates you?" \
        --out figures/joint_norm_probe.png \
        --json data/interventions/joint_norm_probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--release", default="qwen-scope-3-1.7b-base-w32k-l50")
    ap.add_argument("--sae-id", default=None)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--feature", type=int, default=None)
    ap.add_argument("--features", type=int, nargs="+", default=None)
    ap.add_argument("--coefficients", type=float, nargs="+", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()

    device = pick_device()
    sae_id = args.sae_id if args.sae_id else f"layer{args.layer}"
    sae = SAE.from_pretrained(release=args.release, sae_id=sae_id, device=device)

    if args.features:
        feature_ids = list(args.features)
    elif args.feature is not None:
        feature_ids = [args.feature]
    else:
        raise SystemExit("must pass --feature or --features")

    dirs = []
    for f in feature_ids:
        d = sae.W_dec[f].detach().clone()
        d = d / d.norm()
        dirs.append(d)
    decoder_dir = torch.stack(dirs, dim=0).sum(dim=0).to(device).to(torch.float16)
    print(f"[init] joint dir norm = {decoder_dir.norm().item():.3f} ({len(dirs)} features)")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, trust_remote_code=True
    ).to(device)
    model.eval()

    try:
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    if args.prefix:
        formatted = formatted + args.prefix
    inputs = tok(formatted, return_tensors="pt").to(device)

    steer = {"c": 0.0}
    captured = {}

    def hook(m, args_, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["pre"] = h.detach().clone()
        if steer["c"] != 0.0:
            h = h + steer["c"] * decoder_dir
        captured["post"] = h.detach().clone()
        return (h,) + output[1:] if isinstance(output, tuple) else h

    handle = model.model.layers[args.layer].register_forward_hook(hook)

    # baseline pass
    steer["c"] = 0.0
    with torch.no_grad():
        _ = model(**inputs)
    h_base = captured["post"].squeeze(0).to(torch.float32)
    base_norms = h_base.norm(dim=-1)
    print(f"[baseline] mean residual norm: {base_norms.mean().item():.2f}, "
          f"max: {base_norms.max().item():.2f}")

    results = []
    for c in args.coefficients:
        steer["c"] = float(c)
        with torch.no_grad():
            _ = model(**inputs)
        h_post = captured["post"].squeeze(0).to(torch.float32)
        h_pre = captured["pre"].squeeze(0).to(torch.float32)
        post_norms = h_post.norm(dim=-1)
        delta = (h_post - h_base).norm(dim=-1)
        cos = torch.nn.functional.cosine_similarity(h_post, h_base, dim=-1)
        results.append({
            "coef": c,
            "mean_norm_post": float(post_norms.mean().item()),
            "max_norm_post":  float(post_norms.max().item()),
            "mean_norm_ratio": float((post_norms / base_norms).mean().item()),
            "mean_perturb_norm": float(delta.mean().item()),
            "mean_cosine": float(cos.mean().item()),
        })
        print(f"  coef={c:+8.0f}  mean‖h‖={post_norms.mean().item():>7.1f}  "
              f"ratio={results[-1]['mean_norm_ratio']:.3f}  "
              f"‖Δh‖={delta.mean().item():>6.1f}  cos={cos.mean().item():.3f}")
    handle.remove()

    args.json.write_text(json.dumps({
        "model": args.model,
        "release": args.release,
        "feature_ids": feature_ids,
        "layer": args.layer,
        "prompt": args.prompt,
        "prefix": args.prefix,
        "baseline_mean_norm": float(base_norms.mean().item()),
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"[save] {args.json}")

    # plot
    coefs = [r["coef"] for r in results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160)

    ax1.plot(coefs, [r["mean_norm_ratio"] for r in results],
             "o-", color="#9b1d20", lw=2, label="‖h_steered‖ / ‖h_baseline‖")
    ax1.axhline(1.0, color="#666", lw=0.6, ls="--", label="baseline")
    ax1.set_xlabel("steering coefficient")
    ax1.set_ylabel("mean residual-stream norm ratio")
    ax1.set_title("Residual-stream norm under steering\n"
                  "Ratio close to 1 ⇒ register-collapse (norm preserved). "
                  "Ratio diverging ⇒ off-manifold push.",
                  loc="left", fontsize=10)
    ax1.legend(loc="best", frameon=False)
    ax1.grid(True, color="#eee", linewidth=0.7, zorder=0)
    ax1.set_axisbelow(True)
    ax1.axvline(0, color="#444", lw=0.5)

    ax2.plot(coefs, [r["mean_cosine"] for r in results],
             "o-", color="#1f6f8b", lw=2)
    ax2.axhline(1.0, color="#666", lw=0.6, ls="--")
    ax2.set_xlabel("steering coefficient")
    ax2.set_ylabel("cos(h_steered, h_baseline)")
    ax2.set_title("Cosine to baseline residual\n"
                  "Cos ≈ 1 ⇒ small angular drift. Drop ⇒ direction shift.",
                  loc="left", fontsize=10)
    ax2.grid(True, color="#eee", linewidth=0.7, zorder=0)
    ax2.set_axisbelow(True)
    ax2.axvline(0, color="#444", lw=0.5)

    fig.suptitle(
        f"Norm probe: features {feature_ids}, prompt {args.prompt!r}",
        x=0.02, ha="left", fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"[plot] {args.out}")


if __name__ == "__main__":
    main()
