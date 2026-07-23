"""Random-direction matched-norm control for §4.3.

The argument in §4.3 of the paper compares single-feature suppression
at coef=-1000 with joint suppression at coef=-500 and observes that
they produce nearly identical residual-stream distortion (norm ratio
~1.57x and ~1.51x; cosine 0.64 in both cases) but very different output
coherence. Reviewer This comparison alone does not
rule out an off-manifold reading. Two perturbations matched on
two scalar summaries can still be off-manifold in different ways.

The cleaner test: sample random unit vectors in residual-stream space,
sweep their steering coefficient, capture per-coefficient geometry,
generate samples, and report:

(a) whether random-direction perturbations matched on norm-ratio and
    cosine to baseline produce coherent placeholder text (would
    undermine the structural reading) or token-level gibberish
    (would support it);
(b) the NLL and regex degeneration rate at matched-geometry coefs.

Usage:
    uv run python src/random_direction_control.py \
        --num-directions 5 \
        --prompts prompts/intervention_mixed.txt \
        --layer 20 \
        --out data/interventions/random_direction_matched.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_prompts(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def format_chat(tokenizer, prompt: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def degeneration_flags(text: str) -> bool:
    t = text.strip()
    if len(t) < 20:
        return True
    if re.search(r"\b(\w+)\b(\s+\1\b){5,}", t, re.I):
        return True
    if re.search(r"(.)\1{20,}", t):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--num-directions", type=int, default=5)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--coefficients", type=float, nargs="+",
                    default=[-2000, -1500, -1000, -500, 500, 1000, 1500])
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = pick_device()
    print(f"[init] device={device}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, trust_remote_code=True
    ).to(device)
    model.eval()
    d_model = model.config.hidden_size
    print(f"[init] d_model={d_model}")

    prompts = load_prompts(args.prompts)
    print(f"[init] {len(prompts)} prompts")

    rng = np.random.default_rng(args.seed)
    directions = []
    for k in range(args.num_directions):
        v = rng.standard_normal(d_model).astype(np.float32)
        v = v / np.linalg.norm(v)
        directions.append(torch.from_numpy(v).to(device).to(torch.float16))
    print(f"[init] sampled {len(directions)} unit-norm random directions in R^{d_model}")

    # Steering hook
    state = {"vec": None, "coef": 0.0}
    captured = {}

    def hook(module, args_, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["pre"] = h.detach().clone()
        if state["vec"] is not None and state["coef"] != 0.0:
            h = h + state["coef"] * state["vec"]
        captured["post"] = h.detach().clone()
        return (h,) + output[1:] if isinstance(output, tuple) else h

    handle = model.model.layers[args.layer].register_forward_hook(hook)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    torch.manual_seed(args.seed)

    t0 = time.time()
    total = len(directions) * len(args.coefficients) * len(prompts)
    pbar = tqdm(total=total, desc="dir x coef x prompt")
    for d_idx, vec in enumerate(directions):
        state["vec"] = vec
        for coef in args.coefficients:
            state["coef"] = float(coef)
            for prompt_idx, prompt in enumerate(prompts):
                formatted = format_chat(tok, prompt)
                inputs = tok(formatted, return_tensors="pt").to(device)
                input_len = inputs["input_ids"].shape[1]
                completions = []
                remaining = args.samples
                # capture geometry on the prompt-only forward (before generation)
                first_geom_post = None
                first_geom_pre = None
                while remaining > 0:
                    n = min(args.batch_size, remaining)
                    with torch.no_grad():
                        out = model.generate(
                            **inputs,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_new_tokens=args.max_new_tokens,
                            num_return_sequences=n,
                            pad_token_id=tok.eos_token_id,
                        )
                    if first_geom_post is None:
                        # captured["post"] is the layer output at last forward call
                        h_post = captured["post"][0, :input_len].to(torch.float32)
                        h_pre = captured["pre"][0, :input_len].to(torch.float32)
                        first_geom_post = h_post
                        first_geom_pre = h_pre
                    for seq in out:
                        completions.append(tok.decode(seq[input_len:], skip_special_tokens=True))
                    remaining -= n
                # geometry for this (direction, coef) at this prompt
                norm_post = float(first_geom_post.norm(dim=-1).mean().item())
                norm_pre = float(first_geom_pre.norm(dim=-1).mean().item())
                cos_to_pre = float(torch.nn.functional.cosine_similarity(
                    first_geom_post, first_geom_pre, dim=-1
                ).mean().item())
                norm_ratio = norm_post / norm_pre
                # coherence flags
                for s_idx, txt in enumerate(completions):
                    records.append({
                        "direction_idx": d_idx,
                        "coefficient": float(coef),
                        "prompt": prompt,
                        "prompt_idx": prompt_idx,
                        "sample_idx": s_idx,
                        "completion": txt,
                        "norm_ratio_at_prompt_end": norm_ratio,
                        "cos_to_pre_at_prompt_end": cos_to_pre,
                        "norm_post": norm_post,
                        "norm_pre": norm_pre,
                        "regex_degenerate": bool(degeneration_flags(txt)),
                    })
                # incremental save
                with args.out.open("w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)
                pbar.update(1)
    pbar.close()
    handle.remove()
    dt = time.time() - t0
    print(f"[done] {len(records)} samples in {dt:.0f}s ({dt/max(1,len(records))*1000:.0f} ms/sample) -> {args.out}")


if __name__ == "__main__":
    main()
