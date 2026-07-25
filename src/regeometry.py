"""Recompute steering geometry for existing random-direction dumps.

The released random-direction dumps predate the unified probe (src/geometry.py)
and carry the legacy key `norm_ratio_at_prompt_end`, which was in fact read
from the hook state after generate() returned -- a late decode step, not the
prompt forward. This script replays each (direction, coefficient, prompt) cell
with the unified probe and writes the correct keys back.

It does NOT regenerate completions: the sampled text, and therefore every
behavioural count the paper reports (placeholder rates, degeneration,
diversity), is left untouched. Only geometry fields are added.

Directions are reconstructed from the recorded seed and direction index, using
the same RNG procedure as random_direction_control.py, so the replayed
geometry corresponds to the directions actually used.

Usage:
    uv run python src/regeometry.py \
        --in data/interventions/random_direction_K50_at_c-1000.json \
        --model Qwen/Qwen3-1.7B --layer 20 --num-directions 50 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from src.geometry import GeometryRecorder, make_steering_hook
except ImportError:  # invoked as `python src/regeometry.py`
    from geometry import GeometryRecorder, make_steering_hook


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_chat(tok, prompt: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: overwrite the input file")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--num-directions", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = json.loads(args.inp.read_text())
    cells = sorted({(r["direction_idx"], r["coefficient"], r["prompt"])
                    for r in records})
    print(f"[init] {len(records)} records, {len(cells)} (direction, coef, prompt) cells")

    device = pick_device()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, trust_remote_code=True
    ).to(device)
    model.eval()
    d_model = model.config.hidden_size

    # Same construction as random_direction_control.py.
    rng = np.random.default_rng(args.seed)
    directions = []
    for _ in range(args.num_directions):
        v = rng.standard_normal(d_model).astype(np.float32)
        v = v / np.linalg.norm(v)
        directions.append(torch.from_numpy(v).to(device).to(torch.float16))
    print(f"[init] reconstructed {len(directions)} unit directions in R^{d_model}")

    state = {"vec": None, "coef": 0.0}
    recorder = GeometryRecorder()
    handle = model.model.layers[args.layer].register_forward_hook(
        make_steering_hook(state, recorder)
    )

    geom: dict[tuple, dict] = {}
    for i, (d_idx, coef, prompt) in enumerate(cells):
        state["vec"] = directions[int(d_idx)]
        state["coef"] = float(coef)
        recorder.reset()
        inputs = tok(format_chat(tok, prompt), return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
        geom[(d_idx, coef, prompt)] = recorder.summary()
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(cells)}")
    handle.remove()

    for r in records:
        g = geom[(r["direction_idx"], r["coefficient"], r["prompt"])]
        r["norm_ratio_prompt_mean"] = g["norm_ratio_prompt_mean"]
        r["norm_ratio_last_prompt_token"] = g["norm_ratio_last_prompt_token"]
        r["cos_prompt_mean"] = g["cos_prompt_mean"]
        r["cos_last_prompt_token"] = g["cos_last_prompt_token"]
        r["geometry_estimator"] = "unified probe, prompt-forward positions"

    out = args.out or args.inp
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"[save] {out}")

    # Report the shift against the legacy key.
    import statistics as st
    seen, legacy, new = set(), [], []
    for r in records:
        key = (r["direction_idx"], r["coefficient"], r["prompt"])
        if key in seen:
            continue
        seen.add(key)
        if r.get("norm_ratio_at_prompt_end") is not None:
            legacy.append(r["norm_ratio_at_prompt_end"])
            new.append(r["norm_ratio_prompt_mean"])
    if legacy:
        print(f"[compare] legacy mean {st.mean(legacy):.4f}  "
              f"unified mean {st.mean(new):.4f}  "
              f"max |delta| {max(abs(a-b) for a, b in zip(legacy, new)):.4f}")


if __name__ == "__main__":
    main()
