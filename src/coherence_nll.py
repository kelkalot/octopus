"""Per-token NLL coherence proxy.

For a steering JSON (one of data/interventions/*.json), score each
completion's per-token NLL under the *unsteered* baseline model. Mean
NLL on the steered text gives a continuous coherence score that catches
placeholder-token outputs (low predictability under baseline) without
needing structural pattern matching like loop-detection.

This addresses the under-counted "4 % degenerate" rate in the joint-
suppression table, where outputs are pseudo-formal text without semantic
content (e.g. "BASIC TOMOATO SOUP RECIOPLEY... Vc. 100+") that the regex
detector misses.

Usage:
    uv run python src/coherence_nll.py \
        --in data/interventions/joint_suppression.json \
        --model Qwen/Qwen3-1.7B \
        --out data/interventions/joint_suppression_nll.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=200,
                    help="cap NLL evaluation at this many completion tokens")
    args = ap.parse_args()

    records = json.loads(args.inp.read_text(encoding="utf-8"))
    print(f"[init] {len(records)} records from {args.inp}")

    device = pick_device()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, trust_remote_code=True
    ).to(device)
    model.eval()

    annotated: list[dict] = []
    for i, r in enumerate(records):
        try:
            formatted_prompt = tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            formatted_prompt = tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
        full = formatted_prompt + r["completion"]
        ids = tok(full, return_tensors="pt").input_ids.to(device)
        prompt_ids = tok(formatted_prompt, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]
        comp_len = min(args.max_tokens, ids.shape[1] - prompt_len)
        if comp_len <= 0:
            annotated.append({**r, "nll": float("nan"), "comp_len": 0})
            continue
        # We need logits at positions [prompt_len-1 : prompt_len-1+comp_len]
        # to predict tokens at [prompt_len : prompt_len+comp_len].
        with torch.no_grad():
            out = model(ids)
        logits = out.logits[0, prompt_len - 1 : prompt_len - 1 + comp_len].to(torch.float32)
        targets = ids[0, prompt_len : prompt_len + comp_len]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        nll = -log_probs[torch.arange(comp_len), targets].mean().item()
        annotated.append({**r, "nll": float(nll), "comp_len": int(comp_len)})
        if (i + 1) % 50 == 0:
            print(f"[nll] {i+1}/{len(records)}")

    args.out.write_text(json.dumps(annotated, indent=2), encoding="utf-8")
    print(f"[save] {args.out}")

    # summary by coefficient
    by_coef: dict[float, list[float]] = defaultdict(list)
    for r in annotated:
        if r["nll"] == r["nll"]:  # not nan
            by_coef[r["coefficient"]].append(r["nll"])
    print(f"\n[per-coefficient mean NLL]")
    print(f"  {'coef':>10} {'n':>5} {'mean_NLL':>10} {'median_NLL':>12} {'p90_NLL':>10}")
    for c in sorted(by_coef):
        v = torch.tensor(by_coef[c])
        print(f"  {c:>10.1f} {len(v):>5} "
              f"{v.mean().item():>10.3f} "
              f"{v.median().item():>12.3f} "
              f"{torch.quantile(v, 0.90).item():>10.3f}")


if __name__ == "__main__":
    main()
