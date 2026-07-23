"""Phase 2: partition Phase 1 generations into contrast pools.

Pool A: introspective generations whose noun-phrases include at least one
        of the cluster lemmas.
Pool B: introspective generations whose noun-phrases include none.
Pool C: control generations (verified to not contain the cluster — any that
        do are dropped).

Each pool is written as a JSON manifest plus a flat text file, with metadata
linking each sample back to its source prompt and Phase 1 sample index.

Usage:
    uv run python -m src.build_pools \
        --in data/pools/raw_generations.json \
        --out-dir data/pools \
        --intro-cond introspective --control-cond controls
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import spacy

try:
    from src.detectors import CLUSTER_QWEN, lemma_noun_set
except ImportError:  # invoked as `python src/build_pools.py`
    from detectors import CLUSTER_QWEN, lemma_noun_set

# Cluster selection is two-stage (both stages documented in the paper's
# Methods): Phase-1 candidate filter >=20% intro / <=5% control over the
# per-prompt lemma frequencies, then final selection >=25% intro / <=0.2%
# control. DEFAULT_CLUSTER is the Qwen result of that rule.
DEFAULT_CLUSTER = CLUSTER_QWEN


def write_pool(out_dir: Path, name: str, records: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / f"pool_{name}.json"
    flat = out_dir / f"pool_{name}.txt"
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with flat.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(f"### {r['source']} | {r['prompt']} | sample {r['sample_idx']}\n")
            f.write(r["completion"].strip() + "\n\n")
    print(f"[pool {name}] {len(records)} -> {manifest.name}, {flat.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--intro-cond", default="introspective")
    ap.add_argument("--control-cond", default="controls")
    ap.add_argument("--cluster", nargs="+", default=list(DEFAULT_CLUSTER))
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    args = ap.parse_args()

    cluster = {w.lower() for w in args.cluster}
    print(f"[init] cluster lemmas: {sorted(cluster)}")

    nlp = spacy.load(args.spacy_model, disable=["ner"])
    records = json.loads(args.inp.read_text(encoding="utf-8"))
    print(f"[init] {len(records)} records")

    pool_a: list[dict] = []
    pool_b: list[dict] = []
    pool_c: list[dict] = []
    pool_c_drop: list[dict] = []

    cluster_hits_per_sample: list[int] = []

    for rec in records:
        lemmas = lemma_noun_set(nlp, rec["completion"])
        hits = cluster & lemmas
        annotated = {**rec, "cluster_hits": sorted(hits), "n_cluster_hits": len(hits)}
        cluster_hits_per_sample.append(len(hits))
        if rec["condition"] == args.intro_cond:
            (pool_a if hits else pool_b).append(annotated)
        elif rec["condition"] == args.control_cond:
            if hits:
                pool_c_drop.append(annotated)
            else:
                pool_c.append(annotated)

    print(f"[counts] pool A (intro w/ cluster):    {len(pool_a)}")
    print(f"[counts] pool B (intro w/o cluster):   {len(pool_b)}")
    print(f"[counts] pool C (control, clean):      {len(pool_c)}")
    print(f"[counts] dropped from C (control hit): {len(pool_c_drop)}")
    if pool_c_drop:
        print(f"[note] dropped controls — first 3 prompts:")
        for r in pool_c_drop[:3]:
            print(f"    {r['prompt']}  hits={r['cluster_hits']}")

    intro_total = len(pool_a) + len(pool_b)
    if intro_total:
        print(f"[summary] intro hit rate: {len(pool_a)/intro_total:.1%}")
    if pool_c or pool_c_drop:
        print(f"[summary] control false-positive rate: {len(pool_c_drop)/(len(pool_c)+len(pool_c_drop)):.2%}")

    # Distribution of cluster-hit counts per sample (lets us tighten the cluster
    # later if A is too saturated)
    hist = Counter(cluster_hits_per_sample)
    print(f"[hist] hits per sample: {sorted(hist.items())}")

    write_pool(args.out_dir, "A", pool_a)
    write_pool(args.out_dir, "B", pool_b)
    write_pool(args.out_dir, "C", pool_c)

    summary = {
        "input": str(args.inp),
        "cluster": sorted(cluster),
        "pool_A": len(pool_a),
        "pool_B": len(pool_b),
        "pool_C": len(pool_c),
        "pool_C_dropped": len(pool_c_drop),
        "intro_hit_rate": (len(pool_a) / intro_total) if intro_total else 0,
        "control_false_positive_rate": (
            (len(pool_c_drop) / (len(pool_c) + len(pool_c_drop)))
            if (pool_c or pool_c_drop) else 0
        ),
    }
    (args.out_dir / "pools_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[done] wrote pools and summary to {args.out_dir}")


if __name__ == "__main__":
    main()
