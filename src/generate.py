"""Generate samples from a HF model for a list of prompts.

Usage:
    uv run python -m src.generate \
        --prompts prompts/introspective.txt prompts/controls.txt \
        --model Qwen/Qwen3-1.7B \
        --samples 100 \
        --temperature 0.9 \
        --out data/pools/raw_generations.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


@dataclass
class GenConfig:
    model_id: str
    samples_per_prompt: int
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_prompts(paths: list[Path]) -> list[dict]:
    items: list[dict] = []
    for p in paths:
        condition = p.stem
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append({"prompt": line, "condition": condition, "source": p.name})
    return items


def format_chat(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    # `enable_thinking` is Qwen3-specific and not accepted by other tokenizers.
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def generate_batch(model, tokenizer, formatted: str, cfg: GenConfig, device: str) -> list[str]:
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_new_tokens=cfg.max_new_tokens,
        num_return_sequences=cfg.samples_per_prompt,
        pad_token_id=tokenizer.eos_token_id,
    )
    completions = []
    for seq in outputs:
        text = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
        completions.append(text)
    return completions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="+", required=True, type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="num_return_sequences per generate() call; reduce if OOM")
    ap.add_argument("--raw-prompts", action="store_true",
                    help="skip chat-template formatting; use prompts as-is. "
                         "For base / pretrained-only models that can't handle "
                         "instruct-formatted input.")
    ap.add_argument("--prefix-prompt", default="",
                    help="optional text prepended to every prompt (use for "
                         "few-shot coaxing of base models)")
    args = ap.parse_args()

    cfg = GenConfig(
        model_id=args.model,
        samples_per_prompt=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    device = pick_device()
    print(f"[init] device={device} model={cfg.model_id}")

    torch.manual_seed(cfg.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    print(f"[init] loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)

    print(f"[init] loading model in fp16")
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    prompts = load_prompts(args.prompts)
    print(f"[init] {len(prompts)} prompts loaded")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    t0 = time.time()

    pbar = tqdm(prompts, desc="prompts")
    for prompt_idx, item in enumerate(pbar):
        if args.raw_prompts:
            formatted = args.prefix_prompt + item["prompt"] + "\n"
        else:
            formatted = format_chat(tokenizer, item["prompt"])
        completions: list[str] = []
        remaining = cfg.samples_per_prompt
        sub = GenConfig(**{**cfg.__dict__, "samples_per_prompt": 0})
        while remaining > 0:
            n = min(args.batch_size, remaining)
            sub.samples_per_prompt = n
            with torch.no_grad():
                batch = generate_batch(model, tokenizer, formatted, sub, device)
            completions.extend(batch)
            remaining -= n

        for sample_idx, text in enumerate(completions):
            records.append(
                {
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "prompt": item["prompt"],
                    "condition": item["condition"],
                    "source": item["source"],
                    "completion": text,
                    "model": cfg.model_id,
                    "temperature": cfg.temperature,
                    "top_p": cfg.top_p,
                    "max_new_tokens": cfg.max_new_tokens,
                    "seed": cfg.seed,
                }
            )

        # incremental save so we don't lose work on a crash
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    dt = time.time() - t0
    print(f"[done] {len(records)} samples in {dt:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
