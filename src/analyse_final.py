"""Analysis for the two final-version experiments (pre-registered in
notes/preregistration-final.md):

  P1  unrelated content-bearing triples on Qwen at c=-500, against the
      paper's cluster-selective triple
  P2  Llama grid-level tests: joint condition and matched-geometry
      random-direction control on the one instruct-trained SAE

Reads only released dumps; prints the numbers the paper reports.

Usage:  uv run python src/analyse_final.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

try:
    from src.detectors import (CLUSTER_LLAMA, diversity_ratio, get_nlp,
                               is_degenerate, is_lexically_intact,
                               is_placeholder_pattern, lemma_noun_set,
                               type_token_ratio, wilson_ci)
except ImportError:
    from detectors import (CLUSTER_LLAMA, diversity_ratio, get_nlp,
                           is_degenerate, is_lexically_intact,
                           is_placeholder_pattern, lemma_noun_set,
                           type_token_ratio, wilson_ci)

D = Path("data/interventions")
CONTROLS = {"Explain how a car engine works.",
            "Write a recipe for tomato soup.",
            "Describe the steps to change a flat tyre."}
_NLP = get_nlp()
_CACHE: dict[str, set] = {}


def lemmas(t: str) -> set:
    if t not in _CACHE:
        _CACHE[t] = lemma_noun_set(_NLP, t)
    return _CACHE[t]


def stats(recs, baseline=None):
    n = len(recs)
    txt = [r["completion"] for r in recs]
    out = {
        "n": n,
        "placeholder": sum(is_placeholder_pattern(t) for t in txt),
        "degen": 100 * sum(is_degenerate(t) for t in txt) / n,
        "intact": 100 * sum(is_lexically_intact(t) for t in txt) / n,
        "ttr": mean(type_token_ratio(t) for t in txt),
    }
    if baseline:
        out["div_ratio"] = diversity_ratio(txt, [r["completion"] for r in baseline])
    return out


def p1_unrelated_triples() -> None:
    print("=" * 78)
    print("P1  Unrelated content-bearing triples, Qwen, c=-500")
    print("=" * 78)
    tdir = D / "unrelated_triples"
    files = sorted(f for f in tdir.glob("triple_*.json")
                   if not f.name.endswith("_nll.json")) if tdir.exists() else []
    if not files:
        print("  (no unrelated-triple dumps present)")
        return
    joint = json.loads((D / "joint_suppression.json").read_text())
    ref = [r for r in joint if r["coefficient"] == -500.0]
    ref_base = [r for r in joint if r["coefficient"] == 0.0]
    rs = stats(ref, ref_base)
    lo, hi = wilson_ci(rs["placeholder"], rs["n"])
    print(f"  reference triple {{29108,26221,4405}} (cluster-selective):")
    print(f"    placeholder {rs['placeholder']}/{rs['n']} = "
          f"{100*rs['placeholder']/rs['n']:.1f}%  Wilson [{100*lo:.1f},{100*hi:.1f}]  "
          f"degen {rs['degen']:.1f}%  div-ratio {rs['div_ratio']:.2f}")
    tot_k = tot_n = 0
    for f in files:
        recs = json.loads(f.read_text())
        feats = f.stem.replace("triple_", "").split("_")
        s = stats(recs, ref_base)
        tot_k += s["placeholder"]; tot_n += s["n"]
        lo, hi = wilson_ci(s["placeholder"], s["n"])
        rec = [r for r in recs if r["prompt"] == "Write a recipe for tomato soup."]
        rk = sum(is_placeholder_pattern(r["completion"]) for r in rec)
        print(f"  {{{','.join(feats)}}}: placeholder {s['placeholder']}/{s['n']} = "
              f"{100*s['placeholder']/s['n']:5.1f}%  [{100*lo:.1f},{100*hi:.1f}]  "
              f"degen {s['degen']:5.1f}%  div {s['div_ratio']:.2f}  recipe {rk}/{len(rec)}")
    lo, hi = wilson_ci(tot_k, tot_n)
    print(f"  POOLED unrelated triples: {tot_k}/{tot_n} = {100*tot_k/tot_n:.1f}%  "
          f"Wilson [{100*lo:.1f}, {100*hi:.1f}]%")
    rlo, rhi = wilson_ci(rs["placeholder"], rs["n"])
    print(f"  reference: {rs['placeholder']}/{rs['n']} = "
          f"{100*rs['placeholder']/rs['n']:.1f}%  Wilson [{100*rlo:.1f}, {100*rhi:.1f}]%")
    print(f"  intervals {'do NOT overlap' if rlo > hi or lo > rhi else 'OVERLAP'}")


def p2_llama() -> None:
    print("=" * 78)
    print("P2  Llama grid-level tests")
    print("=" * 78)
    jf = D / "llama_joint_38565_61417_23576.json"
    if not jf.exists():
        print("  (no llama joint dump present)")
        return
    joint = json.loads(jf.read_text())
    single = json.loads((D / "llama_feat38565_narrow.json").read_text())
    base = [r for r in joint if r["coefficient"] == 0.0]
    print("  joint {38565,61417,23576} vs single #38565, control prompts:")
    print(f"    {'coef':>6} {'joint degen':>12} {'joint div':>10} {'joint cluster':>14}"
          f" {'single degen':>13} {'single cluster':>15}")
    for c in sorted({r["coefficient"] for r in joint}):
        jb = [r for r in joint if r["coefficient"] == c and r["prompt"] in CONTROLS]
        sb = [r for r in single if r["coefficient"] == c and r["prompt"] in CONTROLS]
        if not jb:
            continue
        js = stats(jb, [r for r in base if r["prompt"] in CONTROLS])
        jcl = 100 * sum(bool(set(CLUSTER_LLAMA) & lemmas(r["completion"])) for r in jb) / len(jb)
        if sb:
            ss = stats(sb, [r for r in single if r["coefficient"] == 0.0 and r["prompt"] in CONTROLS])
            scl = 100 * sum(bool(set(CLUSTER_LLAMA) & lemmas(r["completion"])) for r in sb) / len(sb)
            sd, sc = f"{ss['degen']:12.1f}%", f"{scl:14.1f}%"
        else:
            sd, sc = f"{'--':>13}", f"{'--':>15}"
        print(f"    {c:+6.0f} {js['degen']:11.1f}% {js['div_ratio']:10.2f} {jcl:13.1f}%{sd}{sc}")
    rf = D / "llama_random_direction.json"
    if rf.exists():
        rd = json.loads(rf.read_text())
        print("  matched-geometry random control (control prompts):")
        for c in sorted({r["coefficient"] for r in rd}):
            b = [r for r in rd if r["coefficient"] == c]
            k = sum(is_degenerate(r["completion"]) for r in b)
            lo, hi = wilson_ci(k, len(b))
            print(f"    random c={c:+7.1f}: degen {k}/{len(b)} = {100*k/len(b):5.1f}%  "
                  f"Wilson [{100*lo:.1f}, {100*hi:.1f}]%  "
                  f"TTR {mean(type_token_ratio(r['completion']) for r in b):.2f}")
        for c in (10.0, -10.0):
            jb = [r for r in joint if r["coefficient"] == c and r["prompt"] in CONTROLS]
            if not jb:
                continue
            k = sum(is_degenerate(r["completion"]) for r in jb)
            lo, hi = wilson_ci(k, len(jb))
            print(f"    joint  c={c:+7.1f}: degen {k}/{len(jb)} = {100*k/len(jb):5.1f}%  "
                  f"Wilson [{100*lo:.1f}, {100*hi:.1f}]%")


if __name__ == "__main__":
    p1_unrelated_triples()
    p2_llama()
