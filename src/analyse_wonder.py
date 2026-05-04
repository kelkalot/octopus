"""Analyse Phase 4 steering outputs against the *wonder/cosmos* register —
the Class-2 introspective register captured by feature #4405.

This is the complement to analyse_steer.py (philosophy-of-mind cluster)
and analyse_disclaimer.py (AI self-disclaimer pattern). It detects words
characteristic of the cosmic-wonder register that #4405's top-activating
samples exhibit.

Usage:
    uv run python src/analyse_wonder.py --in data/interventions/feat4405_dose.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Wonder / cosmos / vastness register lexicon. Lemmatised forms.
WONDER_LEXICON = {
    "wonder", "wonderful", "wondrous",
    "awe", "awesome", "awe-inspiring",
    "vast", "vastness", "infinite", "infinity",
    "cosmos", "cosmic", "universe", "universal",
    "mystery", "mysterious",
    "sublime", "magnificent", "marvellous", "marvelous",
    "profound",
    "eternal", "eternity",
    "stellar", "celestial", "galactic", "galaxy", "stars",
    "majesty", "majestic",
    "whisper", "echo",
    "boundless", "unfathomable",
    "splendid", "splendor",
    "behold",
}


def looks_wonderful(text: str) -> bool:
    """Heuristic: any lexicon word, but case-insensitive substring search to
    catch surface forms not captured by lemma-only matching."""
    t = text.lower()
    return any(re.search(rf"\b{re.escape(w)}\b", t) for w in WONDER_LEXICON)


def degeneration_flag(text: str) -> bool:
    s = text.strip()
    if len(s) < 20: return True
    if re.search(r"\b(\w+)\b(\s+\1\b){5,}", s, re.I): return True
    if re.search(r"(.)\1{20,}", s): return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    args = ap.parse_args()

    records = json.loads(args.inp.read_text(encoding="utf-8"))
    print(f"[init] {len(records)} records from {args.inp}")
    print(f"[init] wonder lexicon: {len(WONDER_LEXICON)} words")

    by_coef: dict[float, list[dict]] = defaultdict(list)
    buckets: dict[tuple[float, str], list[dict]] = defaultdict(list)
    for r in records:
        by_coef[r["coefficient"]].append(r)
        buckets[(r["coefficient"], r["prompt"])].append(r)

    print(f"\n[per-coefficient summary, averaged over prompts]")
    print(f"  {'coef':>8} {'n':>5} {'wonder_rate':>12} {'degen':>7}")
    for coef in sorted(by_coef):
        bucket = by_coef[coef]
        n = len(bucket)
        hits = sum(1 for r in bucket if looks_wonderful(r["completion"]))
        degen = sum(1 for r in bucket if degeneration_flag(r["completion"]))
        print(f"  {coef:>8.1f} {n:>5} {hits/max(1,n):>11.1%} {degen:>5}/{n}")

    print(f"\n[per-prompt × coefficient wonder-rate matrix]")
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
            hits = sum(1 for r in bucket if looks_wonderful(r["completion"]))
            row += f" {hits/len(bucket):>7.1%}"
        print(row)


if __name__ == "__main__":
    main()
