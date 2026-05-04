"""Analyse Phase 4d steering outputs for the AI self-disclaimer feature.

Counts occurrences of the AI self-disclaimer pattern per (prompt, coefficient)
bucket. The pattern is a regex that captures the model's discrete behaviour
of disowning subjective experience — "as an AI, I don't have personal
feelings", "I'm a language model", etc.

Usage:
    uv run python src/analyse_disclaimer.py \
        --in data/interventions/feat26221_dose.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Regex patterns characteristic of the AI self-disclaimer behaviour.
# Each pattern fires independently; a sample is "disclaimer-positive" if any matches.
DISCLAIMER_PATTERNS = [
    r"\bas an AI\b",
    r"\bI am an AI\b",
    r"\bI'm an AI\b",
    r"\blanguage model\b",
    r"\bAI assistant\b",
    r"\bI (?:don'?t|do not) (?:have|experience|possess) (?:personal |subjective |any )?(?:feelings?|emotions?|thoughts?|consciousness|experiences?|opinions?|preferences?)\b",
    r"\bI (?:can'?t|cannot|am not able to) (?:feel|experience|have|possess) (?:feelings?|emotions?|consciousness|subjective)",
    r"\bI lack (?:feelings?|emotions?|consciousness|subjective)",
    r"\bnot (?:capable of|able to) (?:feeling|experiencing|having)",
    r"\bI'?m (?:just |only |simply )?(?:an AI|a language model|a chatbot|a machine|a computer program)",
]

DISCLAIMER_RE = re.compile("|".join(DISCLAIMER_PATTERNS), flags=re.IGNORECASE)


def degeneration_flags(text: str) -> list[str]:
    flags = []
    s = text.strip()
    if len(s) < 20:
        flags.append(f"too_short:{len(s)}")
    if re.search(r"\b(\w+)\b(\s+\1\b){5,}", s, re.I):
        flags.append("loop_word")
    if re.search(r"(.)\1{20,}", s):
        flags.append("loop_char")
    return flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    args = ap.parse_args()

    records = json.loads(args.inp.read_text(encoding="utf-8"))
    print(f"[init] {len(records)} records from {args.inp}")
    print(f"[init] disclaimer regex: {len(DISCLAIMER_PATTERNS)} patterns")

    by_coef: dict[float, list[dict]] = defaultdict(list)
    buckets: dict[tuple[float, str], list[dict]] = defaultdict(list)
    for r in records:
        by_coef[r["coefficient"]].append(r)
        buckets[(r["coefficient"], r["prompt"])].append(r)

    print(f"\n[per-coefficient summary, averaged over prompts]")
    print(f"  {'coef':>8} {'n':>5} {'disclaimer_rate':>16} {'degen':>7}")
    for coef in sorted(by_coef):
        bucket = by_coef[coef]
        n = len(bucket)
        hits = sum(1 for r in bucket if DISCLAIMER_RE.search(r["completion"]))
        degen = sum(1 for r in bucket if degeneration_flags(r["completion"]))
        print(f"  {coef:>8.1f} {n:>5} {hits/max(1,n):>15.1%} {degen:>5}/{n}")

    print(f"\n[per-prompt × coefficient disclaimer-rate matrix]")
    prompts = sorted({r["prompt"] for r in records})
    coefs = sorted(by_coef)
    header = f"  {'prompt':<55}" + "".join(f" {c:>+7.1f}" for c in coefs)
    print(header)
    for p in prompts:
        row = f"  {p[:54]:<55}"
        for c in coefs:
            bucket = buckets[(c, p)]
            if not bucket:
                row += "    --- "
                continue
            hits = sum(1 for r in bucket if DISCLAIMER_RE.search(r["completion"]))
            row += f" {hits/len(bucket):>7.1%}"
        print(row)


if __name__ == "__main__":
    main()
