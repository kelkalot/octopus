"""Random-direction matched-geometry control for the steering grid.

The grid's third probe compares single-feature suppression at c=-1000 with
joint suppression at c=-500: nearly identical residual-stream distortion
(norm ratio ~1.57x vs ~1.51x; cosine 0.64 in both cases) but very different
output coherence. That comparison alone does not rule out an off-manifold
reading: two perturbations matched on two scalar summaries can still be
off-manifold in different ways.

The cleaner test run here: sample random unit vectors in residual-stream
space, sweep their steering coefficient, record per-condition geometry with
the unified probe (src/geometry.py), generate samples, and report:

(a) whether random-direction perturbations matched on norm-ratio and
    cosine to baseline produce coherent placeholder text (would
    undermine the structural reading) or diverse-content substitutions
    (would support it);
(b) the regex degeneration rate at matched-geometry coefficients.

Geometry keys written per record (see src/geometry.py for definitions):
    norm_ratio_last_prompt_token, norm_ratio_completion_mean/sd,
    cos_last_prompt_token, cos_completion_mean/sd
Dumps produced before the unified probe carry the legacy key
``norm_ratio_at_prompt_end`` (actually the final decode step of the first
sequence in the last batch); analysis code reads both.

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
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

try:
    from src.detectors import is_degenerate
    from src.geometry import GeometryRecorder, make_steering_hook
except ImportError:  # invoked as `python src/random_direction_control.py`
    from detectors import is_degenerate
    from geometry import GeometryRecorder, make_steering_hook


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

    state = {"vec": None, "coef": 0.0}
    recorder = GeometryRecorder()
    handle = model.model.layers[args.layer].register_forward_hook(
        make_steering_hook(state, recorder)
    )

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
                recorder.reset()
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
                            pad_token_id=tok.eos_token_id,
                        )
                    for seq in out:
                        completions.append(tok.decode(seq[input_len:], skip_special_tokens=True))
                    remaining -= n
                geom = recorder.summary()
                for s_idx, txt in enumerate(completions):
                    records.append({
                        "direction_idx": d_idx,
                        "coefficient": float(coef),
                        "prompt": prompt,
                        "prompt_idx": prompt_idx,
                        "sample_idx": s_idx,
                        "completion": txt,
                        "norm_ratio_last_prompt_token": geom["norm_ratio_last_prompt_token"],
                        "norm_ratio_completion_mean": geom["norm_ratio_completion_mean"],
                        "norm_ratio_completion_sd": geom["norm_ratio_completion_sd"],
                        "cos_last_prompt_token": geom["cos_last_prompt_token"],
                        "cos_completion_mean": geom["cos_completion_mean"],
                        "cos_completion_sd": geom["cos_completion_sd"],
                        "regex_degenerate": bool(is_degenerate(txt)),
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
