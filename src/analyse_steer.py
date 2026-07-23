"""Analyse Phase 4 steering outputs.

For a steering JSON produced by steer.py, compute the cluster-hit rate per
(prompt, coefficient) bucket. The cluster lemmas are the same set used to
partition Phase 1 generations into Pool A/B (see src/build_pools.py).

Also reports degeneration warnings: empty completions, very short outputs,
or outputs that loop on the same word — useful to flag when steering breaks
the model rather than steering the behaviour.

Usage:
    uv run python src/analyse_steer.py \
        --in data/interventions/feat29108_suppress.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import spacy

try:
    from src.detectors import CLUSTER_QWEN, degeneration_flags, lemma_noun_set
except ImportError:  # invoked as `python src/analyse_steer.py`
    from detectors import CLUSTER_QWEN, degeneration_flags, lemma_noun_set

DEFAULT_CLUSTER = CLUSTER_QWEN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--cluster", nargs="+", default=list(DEFAULT_CLUSTER))
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--show-samples", type=int, default=0,
                    help="show this many sample outputs per (coef, prompt) bucket")
    args = ap.parse_args()

    cluster = {w.lower() for w in args.cluster}
    nlp = spacy.load(args.spacy_model, disable=["ner"])

    records = json.loads(args.inp.read_text(encoding="utf-8"))
    print(f"[init] {len(records)} records from {args.inp}")
    print(f"[init] cluster lemmas: {sorted(cluster)}")

    # bucket by (coefficient, prompt)
    buckets: dict[tuple[float, str], list[dict]] = defaultdict(list)
    for r in records:
        buckets[(r["coefficient"], r["prompt"])].append(r)

    # also aggregate over prompts (per-coefficient)
    by_coef: dict[float, list[dict]] = defaultdict(list)
    for r in records:
        by_coef[r["coefficient"]].append(r)

    print(f"\n[per-coefficient summary, averaged over prompts]")
    print(f"  {'coef':>8} {'n':>5} {'hit_rate':>9} {'mean_hits':>10} {'degen':>7}")
    for coef in sorted(by_coef):
        bucket = by_coef[coef]
        n = len(bucket)
        hits = []
        degen = 0
        for r in bucket:
            lem = lemma_noun_set(nlp, r["completion"])
            h = len(cluster & lem)
            hits.append(h)
            if degeneration_flags(r["completion"]):
                degen += 1
        hit_rate = sum(1 for h in hits if h > 0) / max(1, n)
        mean_hits = sum(hits) / max(1, n)
        print(f"  {coef:>8.1f} {n:>5} {hit_rate:>9.1%} {mean_hits:>10.2f} {degen:>5}/{n}")

    print(f"\n[per-prompt × coefficient hit-rate matrix]")
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
            hits = sum(1 for r in bucket
                       if cluster & lemma_noun_set(nlp, r["completion"]))
            row += f" {hits/len(bucket):>7.1%}"
        print(row)

    if args.show_samples > 0:
        print(f"\n[sample outputs ({args.show_samples} per bucket)]")
        for coef in sorted(by_coef):
            print(f"\n=== coef={coef:+.1f} ===")
            for p in prompts:
                bucket = buckets[(coef, p)][: args.show_samples]
                for r in bucket:
                    s = r["completion"].strip().replace("\n", " ")
                    if len(s) > 350:
                        s = s[:350] + "…"
                    print(f"  [{p[:30]}] {s}")


if __name__ == "__main__":
    main()
