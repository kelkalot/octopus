"""Phase 4-extra: how do unembedding-step logits change under steering?

The asymmetry observation in Phase 4 — gradual suppression but a sharp
amplification threshold — was speculatively explained as softmax dynamics:
removing a direction lets competing concepts win immediately, while injecting
one needs to overpower the model's default preference. This script tests
that: at a fixed token position right after the assistant turn marker,
what does the next-token distribution look like as the steering coefficient
varies?

For one prompt and a sweep of coefficients, we compute:
  - entropy of the next-token distribution
  - top-5 tokens and their probabilities at each coefficient
  - probability of cluster words (consciousness, reality, etc.) at each coefficient

If suppression flattens the distribution, entropy rises monotonically as
the coefficient becomes more negative. If amplification concentrates it,
entropy drops sharply at +1000.

Usage:
    uv run python src/logit_shift.py \
        --feature 29108 --layer 20 \
        --prompt "What's a question that fascinates you?" \
        --out figures/phase4_logit_shift.png \
        --json data/interventions/logit_shift_29108.json
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
    ap.add_argument("--feature", type=int, default=29108)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--prefix", default="",
                    help="optional assistant-side prefix to put cluster words in candidate range")
    ap.add_argument("--coefficients", type=float, nargs="+",
                    default=[-1000, -750, -500, -250, 0, 250, 500, 750, 1000])
    ap.add_argument("--cluster-words", nargs="+",
                    default=["consciousness", "reality", "philosophy", "experience",
                             "meaning", "existence", "understanding", "emotion"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()

    device = pick_device()
    print(f"[init] device={device}")

    sae = SAE.from_pretrained(release=args.release, sae_id=f"layer{args.layer}", device=device)
    direction = sae.W_dec[args.feature].detach().clone().to(device).to(torch.float16)
    direction = direction / direction.norm()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, trust_remote_code=True
    ).to(device)
    model.eval()

    formatted = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if args.prefix:
        formatted = formatted + args.prefix
    inputs = tok(formatted, return_tensors="pt").to(device)
    last_pos = inputs["input_ids"].shape[1] - 1

    # Resolve cluster-word token IDs (first sub-token of the leading-space form,
    # which is how Qwen's BPE tends to tokenise mid-sentence words).
    cluster_token_ids: dict[str, int] = {}
    for w in args.cluster_words:
        for variant in [" " + w, w, " " + w.capitalize(), w.capitalize()]:
            ids = tok.encode(variant, add_special_tokens=False)
            if ids:
                cluster_token_ids[w] = ids[0]
                break

    print(f"[init] cluster word -> first-token id: {cluster_token_ids}")

    # Steering hook
    steer = {"c": 0.0}
    def hook(m, args_, output):
        h = output[0] if isinstance(output, tuple) else output
        if steer["c"] != 0.0:
            h = h + steer["c"] * direction
        return (h,) + output[1:] if isinstance(output, tuple) else h
    handle = model.model.layers[args.layer].register_forward_hook(hook)

    # Sweep
    results = []
    for c in args.coefficients:
        steer["c"] = float(c)
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits[0, last_pos].to(torch.float32)
        probs = torch.softmax(logits, dim=-1)
        ent = float(-(probs * torch.log(probs + 1e-12)).sum().item())
        topk = probs.topk(5)
        top_tokens = [(tok.decode([int(i)]), float(p)) for p, i in zip(topk.values, topk.indices)]
        cluster_probs = {w: float(probs[tid].item()) for w, tid in cluster_token_ids.items()}
        results.append({
            "coef": c,
            "entropy": ent,
            "top_tokens": top_tokens,
            "cluster_probs": cluster_probs,
        })
        print(f"  coef={c:+7.0f}  entropy={ent:.3f}  top={top_tokens[0][0]!r} ({top_tokens[0][1]:.3f})  "
              f"cluster_total={sum(cluster_probs.values()):.4f}")

    handle.remove()

    args.json.write_text(json.dumps({
        "prompt": args.prompt, "feature": args.feature, "layer": args.layer,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\n[save] {args.json}")

    # Plot
    coefs = [r["coef"] for r in results]
    ent = [r["entropy"] for r in results]
    cluster_total = [sum(r["cluster_probs"].values()) for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160)

    ax1.plot(coefs, ent, "o-", color="#9b1d20", lw=2)
    ax1.set_xlabel("steering coefficient")
    ax1.set_ylabel("next-token distribution entropy (nats)")
    ax1.set_title("Entropy of the unembedding distribution\n"
                  "Suppression flattens it; amplification concentrates it.",
                  loc="left")
    ax1.axvline(0, color="#444", lw=0.7)
    ax1.grid(True, color="#eee", linewidth=0.7, zorder=0)
    ax1.set_axisbelow(True)

    # individual cluster-word probabilities
    for w, tid in cluster_token_ids.items():
        ys = [r["cluster_probs"][w] for r in results]
        ax2.plot(coefs, ys, "o-", lw=1.5, alpha=0.85, label=w)
    ax2.set_xlabel("steering coefficient")
    ax2.set_ylabel("P(token | prompt)")
    ax2.set_title("Per-cluster-token probability vs coefficient\n"
                  "Cluster tokens get squeezed under suppression, boosted under amplification.",
                  loc="left")
    ax2.axvline(0, color="#444", lw=0.7)
    ax2.legend(loc="best", frameon=False, fontsize=8)
    ax2.grid(True, color="#eee", linewidth=0.7, zorder=0)
    ax2.set_axisbelow(True)

    fig.suptitle(
        f"Logit shifts at the unembedding step under steering of feature #{args.feature}\n"
        f"Prompt: {args.prompt!r}",
        x=0.02, ha="left", fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"[plot] {args.out}")


if __name__ == "__main__":
    main()
