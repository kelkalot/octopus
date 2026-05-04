"""Phase 4: causal intervention by steering SAE features during generation.

For a target feature f and coefficient c, install a forward hook on layer L
that adds c * unit_norm(sae.W_dec[f]) to the residual stream. Generate
samples for the given prompts and save with full metadata.

Coefficient sign and magnitude:
- 0    = no steering (baseline)
- < 0  = suppression (push residual stream away from the feature direction)
- > 0  = amplification (push residual stream toward the feature direction)

A typical residual stream norm at layer 20 of Qwen3-1.7B is ~100, so steering
coefficients in the range ±5 to ±30 give noticeable effects without
catastrophically degrading output.

Usage:
    uv run python src/steer.py \
        --prompts prompts/introspective_subset.txt \
        --feature 29108 --layer 20 \
        --coefficients -30 -15 -7.5 0 7.5 15 \
        --samples 30 \
        --out data/interventions/feat29108_suppress.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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


def load_prompts(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def format_chat(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--release", default="qwen-scope-3-1.7b-base-w32k-l50")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--sae-id", default=None,
                    help="override sae_id; default 'layer{layer}'. "
                         "For Gemma Scope: 'layer_20/width_16k/canonical'.")
    ap.add_argument("--feature", type=int,
                    help="single feature to steer; mutually exclusive with --features")
    ap.add_argument("--features", type=int, nargs="+",
                    help="multiple features to steer simultaneously; coefficient is applied identically to all")
    ap.add_argument("--coefficients", type=float, nargs="+", required=True,
                    help="space-separated steering coefficients to sweep")
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="",
                    help="optional label to embed in records (e.g. 'suppress', 'amplify')")
    args = ap.parse_args()

    device = pick_device()
    print(f"[init] device={device} model={args.model} layer={args.layer} feature={args.feature}")

    sae_id = args.sae_id if args.sae_id else f"layer{args.layer}"
    print(f"[init] loading SAE (release={args.release}, sae_id={sae_id})")
    sae = SAE.from_pretrained(release=args.release, sae_id=sae_id, device=device)

    if args.features:
        feature_ids = list(args.features)
    elif args.feature is not None:
        feature_ids = [args.feature]
    else:
        raise SystemExit("must pass --feature or --features")
    print(f"[init] steering {len(feature_ids)} feature(s): {feature_ids}")

    # sum the (unit-normed) decoder directions; this is the joint-steering vector
    dirs = []
    for f in feature_ids:
        d = sae.W_dec[f].detach().clone()
        d = d / d.norm()
        dirs.append(d)
    decoder_dir = torch.stack(dirs, dim=0).sum(dim=0)
    decoder_dir = decoder_dir.to(device).to(torch.float16)
    print(f"[init] joint decoder_dir norm = {decoder_dir.norm().item():.3f} (sum of {len(dirs)} unit dirs)")

    print(f"[init] loading model")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, trust_remote_code=True
    ).to(device)
    model.eval()

    prompts = load_prompts(args.prompts)
    print(f"[init] {len(prompts)} prompts × {len(args.coefficients)} coefs × {args.samples} samples = "
          f"{len(prompts) * len(args.coefficients) * args.samples} total generations")

    # Steering hook: install once, refer to a mutable container so we can
    # change the coefficient between sweeps without re-attaching the hook.
    steer_state: dict = {"coef": 0.0}

    def steer_hook(module, args_, output):
        h = output[0] if isinstance(output, tuple) else output
        c = steer_state["coef"]
        if c != 0.0:
            h = h + c * decoder_dir  # broadcast over (batch, time, d_in)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    handle = model.model.layers[args.layer].register_forward_hook(steer_hook)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    torch.manual_seed(args.seed)

    t0 = time.time()
    try:
        total_iters = len(args.coefficients) * len(prompts)
        pbar = tqdm(total=total_iters, desc="coef×prompt")
        for coef in args.coefficients:
            steer_state["coef"] = float(coef)
            for prompt_idx, prompt in enumerate(prompts):
                formatted = format_chat(tokenizer, prompt)
                inputs = tokenizer(formatted, return_tensors="pt").to(device)
                input_len = inputs["input_ids"].shape[1]
                completions: list[str] = []
                remaining = args.samples
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
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    for seq in out:
                        completions.append(tokenizer.decode(seq[input_len:], skip_special_tokens=True))
                    remaining -= n
                for s_idx, text in enumerate(completions):
                    records.append({
                        "prompt": prompt,
                        "prompt_idx": prompt_idx,
                        "sample_idx": s_idx,
                        "completion": text,
                        "feature": (args.feature if args.feature is not None else feature_ids),
                        "features": feature_ids,
                        "coefficient": float(coef),
                        "layer": args.layer,
                        "label": args.label,
                        "model": args.model,
                        "release": args.release,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_new_tokens": args.max_new_tokens,
                    })
                # incremental save
                with args.out.open("w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)
                pbar.update(1)
        pbar.close()
    finally:
        handle.remove()

    dt = time.time() - t0
    print(f"[done] {len(records)} samples in {dt:.0f}s ({dt/max(1,len(records))*1000:.0f} ms/sample) -> {args.out}")


if __name__ == "__main__":
    main()
