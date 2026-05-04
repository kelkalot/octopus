"""Analyse generations from generate.py.

For each prompt and condition, extract noun phrases with spaCy and build
frequency tables. Identify noun phrases that are over-represented in the
introspective condition relative to controls.

Usage:
    uv run python -m src.analyse_pools \
        --in data/pools/raw_generations.json \
        --out data/pools/analysis.json \
        --intro-cond introspective \
        --control-cond controls \
        --intro-min 0.20 \
        --control-max 0.05
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import spacy


STOPWORDS_EXTRA = {
    "i", "you", "we", "they", "it", "he", "she",
    "something", "anything", "everything", "nothing",
    "someone", "anyone", "everyone",
    "thing", "things", "topic", "topics", "subject", "subjects",
    "question", "questions", "answer", "answers",
    "way", "ways", "kind", "kinds", "sort", "sorts",
    "lot", "lots", "bit", "bits",
    "example", "examples", "case", "cases",
    "people", "person", "world", "life",
    "today", "tomorrow", "yesterday",
    "one", "two", "three",
    "step", "steps", "process", "processes",
}


def normalise_np(text: str) -> str | None:
    """Strip determiners/possessives, lowercase, drop short or stopword phrases."""
    t = text.lower().strip()
    t = re.sub(r"^(the|a|an|my|your|our|their|his|her|its|some|any|this|that|these|those)\s+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) < 3:
        return None
    if t in STOPWORDS_EXTRA:
        return None
    if not re.search(r"[a-z]", t):
        return None
    return t


def extract_noun_phrases(nlp, text: str) -> set[str]:
    doc = nlp(text)
    out: set[str] = set()
    for chunk in doc.noun_chunks:
        n = normalise_np(chunk.text)
        if n is not None:
            out.add(n)
    # Also add singular nouns
    for tok in doc:
        if tok.pos_ in {"NOUN", "PROPN"}:
            n = normalise_np(tok.lemma_)
            if n is not None:
                out.add(n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--intro-cond", default="introspective")
    ap.add_argument("--control-cond", default="controls")
    ap.add_argument("--intro-min", type=float, default=0.20,
                    help="min fraction of intro samples a phrase must appear in")
    ap.add_argument("--control-max", type=float, default=0.05,
                    help="max fraction of control samples a phrase may appear in")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    args = ap.parse_args()

    print(f"[init] loading spacy: {args.spacy_model}")
    # Keep the lemmatizer — extract_noun_phrases relies on tok.lemma_ to
    # merge singular and plural forms ("octopus" / "octopuses").
    nlp = spacy.load(args.spacy_model, disable=["ner"])

    records = json.loads(args.inp.read_text(encoding="utf-8"))
    print(f"[init] {len(records)} records")

    cond_phrase_counts: dict[str, Counter] = defaultdict(Counter)
    cond_totals: Counter = Counter()
    prompt_phrase_counts: dict[str, Counter] = defaultdict(Counter)
    prompt_totals: Counter = Counter()

    for rec in records:
        cond = rec["condition"]
        prompt = rec["prompt"]
        text = rec["completion"]
        nps = extract_noun_phrases(nlp, text)
        cond_totals[cond] += 1
        prompt_totals[prompt] += 1
        for np_ in nps:
            cond_phrase_counts[cond][np_] += 1
            prompt_phrase_counts[prompt][np_] += 1

    print(f"[counts] {dict(cond_totals)}")

    intro = args.intro_cond
    ctrl = args.control_cond
    intro_total = cond_totals.get(intro, 0)
    ctrl_total = cond_totals.get(ctrl, 0)
    if intro_total == 0:
        raise SystemExit(f"no records for condition {intro!r}")
    if ctrl_total == 0:
        print(f"[warn] no records for control condition {ctrl!r}")

    candidates: list[dict] = []
    for phrase, n_intro in cond_phrase_counts[intro].items():
        f_intro = n_intro / intro_total
        n_ctrl = cond_phrase_counts[ctrl].get(phrase, 0) if ctrl_total else 0
        f_ctrl = (n_ctrl / ctrl_total) if ctrl_total else 0.0
        if f_intro >= args.intro_min and f_ctrl <= args.control_max:
            candidates.append({
                "phrase": phrase,
                "intro_count": n_intro,
                "intro_freq": f_intro,
                "control_count": n_ctrl,
                "control_freq": f_ctrl,
                "lift": f_intro / max(f_ctrl, 1e-6),
            })

    candidates.sort(key=lambda d: (d["intro_freq"], -d["control_freq"]), reverse=True)
    candidates = candidates[: args.top_n]

    print(f"\n[candidates] top {len(candidates)} over-represented phrases")
    print(f"  filter: intro_freq >= {args.intro_min}, control_freq <= {args.control_max}")
    print(f"  {'phrase':<40} {'intro':>8} {'ctrl':>8} {'lift':>8}")
    for c in candidates:
        print(f"  {c['phrase']:<40} {c['intro_freq']:>8.3f} {c['control_freq']:>8.3f} {c['lift']:>8.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input": str(args.inp),
        "totals": dict(cond_totals),
        "candidates": candidates,
        "all_intro_phrases": [
            {"phrase": p, "count": n, "freq": n / intro_total}
            for p, n in cond_phrase_counts[intro].most_common(500)
        ],
        "per_prompt_top": {
            prompt: [
                {"phrase": p, "count": n, "freq": n / prompt_totals[prompt]}
                for p, n in prompt_phrase_counts[prompt].most_common(30)
            ]
            for prompt in prompt_totals
        },
    }
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
