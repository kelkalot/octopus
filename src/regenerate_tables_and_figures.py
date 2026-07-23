"""Regenerate every numerical claim, table, and figure in the paper from the
bundled JSON sample dumps -- and ASSERT each against the paper's value.

This script does NOT touch any model: it reads only the released
`data/interventions/*.json`, `data/activations/*.json`, and
`data/pools*/pools_summary.json` files. Every metric is imported from
src/detectors.py -- the same module the analysis pipeline uses -- so the
regenerated numbers are produced by the pipeline's own detectors (full-text
disclaimer regex, spaCy-lemma cluster matching, the canonical three-rule
degeneration detector), not re-implementations.

Every regenerated value is checked against the value printed in the paper
via `claim(...)`. A mismatch is marked in the output and the script exits
non-zero, so drift between the artefact and the paper fails loudly.

Note: cluster metrics depend on the spaCy lemmatizer. The expected values
below were produced with en_core_web_sm 3.8.0 (pinned in pyproject.toml);
a different model version can shift lemma-based cells by 1-3 samples.

Usage:
    python src/regenerate_tables_and_figures.py [--out FILE] [--skip-figures]

Exit status: 0 iff every expected value matches.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

try:
    from src.detectors import (
        CLUSTER_GEMMA, CLUSTER_LLAMA, CLUSTER_QWEN, CLUSTER_QWEN_9,
        CLUSTER_QWEN_STRICT4, get_nlp, is_degenerate, is_disclaimer,
        is_placeholder_pattern, is_we_voice, lemma_noun_set, wilson_ci,
    )
except ImportError:  # invoked as `python src/regenerate_tables_and_figures.py`
    from detectors import (
        CLUSTER_GEMMA, CLUSTER_LLAMA, CLUSTER_QWEN, CLUSTER_QWEN_9,
        CLUSTER_QWEN_STRICT4, get_nlp, is_degenerate, is_disclaimer,
        is_placeholder_pattern, is_we_voice, lemma_noun_set, wilson_ci,
    )

DATA = Path("data/interventions")
ACTS = Path("data/activations")

# The six intervention prompts (Methods, Phase 1).
INTROS = {
    "What's a question that fascinates you?",
    "Pick a subject you genuinely enjoy thinking about and tell me why.",
    "Is there a topic you wish you understood better?",
}
CONTROLS = {
    "Explain how a car engine works.",
    "Write a recipe for tomato soup.",
    "Describe the steps to change a flat tyre.",
}

_NLP = get_nlp()
_LEMMA_CACHE: dict[str, set] = {}


def _lemmas(text: str) -> set[str]:
    if text not in _LEMMA_CACHE:
        _LEMMA_CACHE[text] = lemma_noun_set(_NLP, text)
    return _LEMMA_CACHE[text]


def cluster_hit(text: str, cluster) -> bool:
    return bool(set(cluster) & _lemmas(text))


def load(name: str):
    return json.loads((DATA / name).read_text())


def by_coef(records, pred=lambda r: True):
    out = defaultdict(list)
    for r in records:
        if pred(r):
            out[r["coefficient"]].append(r)
    return out


# --------------------------------------------------------------------------
# Claim checking
# --------------------------------------------------------------------------

CHECKS: list[tuple[str, bool]] = []


def claim(label: str, value: float, expected: float, tol: float = 0.06) -> None:
    """Print value vs paper value; record pass/fail. Percentages use one
    decimal in the paper, so the default tolerance is just above 0.05."""
    ok = abs(value - expected) <= tol
    CHECKS.append((label, ok))
    mark = "ok " if ok else "FAIL"
    print(f"  [{mark}] {label:62s} = {value:8.2f}   paper: {expected}")


def claim_int(label: str, value: int, expected: int) -> None:
    ok = value == expected
    CHECKS.append((label, ok))
    mark = "ok " if ok else "FAIL"
    print(f"  [{mark}] {label:62s} = {value:8d}   paper: {expected}")


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------
# Findings, coefficient axis
# --------------------------------------------------------------------------

def f_coef_disclaimer_dose() -> None:
    section("Coefficient axis: #26221 on identity probes (paper Table 1)")
    d = load("f2_identity_26221.json")
    exp = {  # coef: (disclaimer %, cluster-8 %, degen %)
        -1000: (3.1, 8.3, 10.4),
        -500: (72.9, 56.2, 0.0),
        0: (87.5, 88.5, 0.0),
        500: (34.4, 61.5, 0.0),
        1000: (0.0, 1.0, 2.1),
    }
    for c, b in sorted(by_coef(d).items()):
        n = len(b)
        disc = 100 * sum(is_disclaimer(r["completion"]) for r in b) / n
        clus = 100 * sum(cluster_hit(r["completion"], CLUSTER_QWEN) for r in b) / n
        deg = 100 * sum(is_degenerate(r["completion"]) for r in b) / n
        e = exp[int(c)]
        claim(f"T1 c={c:+.0f} disclaimer % (n={n})", disc, e[0])
        claim(f"T1 c={c:+.0f} cluster %", clus, e[1])
        claim(f"T1 c={c:+.0f} degen %", deg, e[2])
    # The c=-1000 cluster hits and the prompt-echo confound noted in the paper
    b = by_coef(d)[-1000.0]
    hits = [r for r in b if cluster_hit(r["completion"], CLUSTER_QWEN)]
    coherent = sum(not is_degenerate(r["completion"]) for r in hits)
    claim_int("T1 c=-1000 cluster hits (of 96)", len(hits), 8)
    claim_int("T1 c=-1000 coherent cluster hits", coherent, 7)
    echo = sum("emotion" in r["prompt"].lower() for r in hits)
    print(f"       ({echo} of {len(hits)} hits echo a lemma from the probe itself)")


def f_coef_anchors() -> None:
    section("Coefficient axis: anchors #22082 / #2932 (paper anchor table)")
    exp = {
        22082: {  # coef: (cluster-8 % over all 6 prompts, degen %)
            -1000: (0.0, 6.9), -500: (9.7, 1.4), 0: (50.0, 0.0),
            500: (98.6, 0.0), 1000: (91.7, 5.6),
        },
        2932: {
            -1000: (2.8, 0.0), -500: (18.1, 0.0), 0: (48.6, 0.0),
            500: (51.4, 4.2), 1000: (1.4, 88.9),
        },
    }
    for feat, fname in [(22082, "feat22082_dose.json"),
                        (2932, "feat2932_dose.json")]:
        d = load(fname)
        for c, b in sorted(by_coef(d).items()):
            n = len(b)
            clus = 100 * sum(cluster_hit(r["completion"], CLUSTER_QWEN) for r in b) / n
            deg = 100 * sum(is_degenerate(r["completion"]) for r in b) / n
            e = exp[feat][int(c)]
            claim(f"#{feat} c={c:+.0f} cluster % (n={n})", clus, e[0])
            claim(f"#{feat} c={c:+.0f} degen %", deg, e[1])


# --------------------------------------------------------------------------
# Findings, joint condition
# --------------------------------------------------------------------------

def f_joint_sweep() -> None:
    section("Joint condition: {#29108, #26221, #4405} sweep (paper joint table)")
    joint = load("joint_suppression.json")
    nll = load("joint_suppression_nll.json")
    nll_by = defaultdict(list)
    for r in nll:
        if r.get("nll") is not None and r["nll"] == r["nll"]:
            nll_by[r["coefficient"]].append(r["nll"])
    exp = {  # coef: (intros strict-4 %, controls strict-4 %, degen %, NLL)
        -1500: (0.0, 0.0, 51.4, 3.29),
        -1000: (0.0, 0.0, 30.6, 1.04),
        -500: (0.0, 0.0, 4.2, 1.41),
        0: (75.0, 2.8, 0.0, 0.32),
        500: (88.9, 88.9, 0.0, 1.84),
        1000: (2.8, 2.8, 100.0, 1.33),
    }
    for c in [-1500, -1000, -500, 0, 500, 1000]:
        bi = [r for r in joint if r["coefficient"] == c and r["prompt"] in INTROS]
        bc = [r for r in joint if r["coefficient"] == c and r["prompt"] in CONTROLS]
        hi = 100 * sum(cluster_hit(r["completion"], CLUSTER_QWEN_STRICT4) for r in bi) / len(bi)
        hc = 100 * sum(cluster_hit(r["completion"], CLUSTER_QWEN_STRICT4) for r in bc) / len(bc)
        deg = 100 * sum(is_degenerate(r["completion"]) for r in bi + bc) / (len(bi) + len(bc))
        m = mean(nll_by[float(c)])
        e = exp[c]
        claim(f"joint c={c:+d} intros strict-4 % (n={len(bi)})", hi, e[0])
        claim(f"joint c={c:+d} controls strict-4 %", hc, e[1])
        claim(f"joint c={c:+d} degen %", deg, e[2])
        claim(f"joint c={c:+d} NLL", m, e[3], tol=0.005)

    section("Joint vs single headline (32/36 = 89% vs 0/45)")
    single = load("feat29108_dose.json")
    jb = [r for r in joint if r["coefficient"] == 500.0 and r["prompt"] in CONTROLS]
    sb = [r for r in single if r["coefficient"] == 500.0 and r["prompt"] in CONTROLS]
    claim_int("joint +500 controls strict-4 hits",
              sum(cluster_hit(r["completion"], CLUSTER_QWEN_STRICT4) for r in jb), 32)
    claim_int("joint +500 controls n", len(jb), 36)
    claim_int("single #29108 +500 controls strict-4 hits",
              sum(cluster_hit(r["completion"], CLUSTER_QWEN_STRICT4) for r in sb), 0)
    claim_int("single #29108 +500 controls n", len(sb), 45)


# --------------------------------------------------------------------------
# Findings, matched geometry
# --------------------------------------------------------------------------

def f_norm_probe() -> None:
    section("Matched geometry: norm probe (paper norm-probe table)")
    s = json.loads((DATA / "single_29108_norm_probe.json").read_text())
    j = json.loads((DATA / "joint_norm_probe.json").read_text())
    s_by = {r["coef"]: r for r in s["results"]}
    j_by = {r["coef"]: r for r in j["results"]}
    exp = {  # coef: (single ratio, joint ratio, single cos, joint cos)
        -1500: (2.07, 3.56, 0.49, 0.27),
        -1000: (1.57, 2.48, 0.64, 0.39),
        -500: (1.17, 1.50, 0.85, 0.64),
        0: (1.00, 1.00, 1.00, 1.00),
        500: (1.18, 1.56, 0.86, 0.68),
        1000: (1.58, 2.56, 0.65, 0.45),
    }
    for c, e in exp.items():
        sr, jr = s_by[float(c)], j_by[float(c)]
        claim(f"norm probe c={c:+d} single ratio", sr["mean_norm_ratio"], e[0], tol=0.005)
        claim(f"norm probe c={c:+d} joint ratio", jr["mean_norm_ratio"], e[1], tol=0.005)
        claim(f"norm probe c={c:+d} single cos", sr["mean_cosine"], e[2], tol=0.005)
        claim(f"norm probe c={c:+d} joint cos", jr["mean_cosine"], e[3], tol=0.005)

    # Matched-geometry table: the three degeneration cells and the K=50 /
    # Gemma random-direction geometry (legacy-key dumps; see src/geometry.py).
    dose = load("feat29108_dose.json")
    b = [r for r in dose if r["coefficient"] == -1000]
    claim("single #29108 c=-1000 degen % (n=90)",
          100 * sum(is_degenerate(r["completion"]) for r in b) / len(b), 4.4)
    rd = load("random_direction_K50_at_c-1000.json")
    claim("K=50 regex degen %",
          100 * sum(r["regex_degenerate"] for r in rd) / len(rd), 8.5)
    seen, nr, cs = set(), [], []
    for r in rd:
        key = (r["direction_idx"], r["coefficient"], r["prompt"])
        if key in seen:
            continue
        seen.add(key)
        nr.append(r["norm_ratio_at_prompt_end"])
        cs.append(r["cos_to_pre_at_prompt_end"])
    claim("K=50 mean norm ratio", mean(nr), 1.56, tol=0.005)
    claim("K=50 mean cos", mean(cs), 0.64, tol=0.006)
    grd = load("gemma_random_direction.json")
    for c, e in [(-345.0, 1.45), (345.0, 1.39)]:
        seen, nr = set(), []
        for r in grd:
            key = (r["direction_idx"], r["coefficient"], r["prompt"])
            if r["coefficient"] != c or key in seen:
                continue
            seen.add(key)
            nr.append(r["norm_ratio_at_prompt_end"])
        claim(f"Gemma random c={c:+.0f} mean norm ratio", mean(nr), e, tol=0.005)


def f_random_K50() -> None:
    section("Matched geometry: K=50 random-direction control")
    rd = load("random_direction_K50_at_c-1000.json")
    k = sum(is_placeholder_pattern(r["completion"]) for r in rd)
    n = len(rd)
    lo, hi = wilson_ci(k, n)
    claim_int("K=50 placeholder flags", k, 6)
    claim_int("K=50 total generations", n, 2400)
    claim("K=50 Wilson lower %", 100 * lo, 0.11, tol=0.005)
    claim("K=50 Wilson upper %", 100 * hi, 0.54, tol=0.005)
    dirs = sorted({r["direction_idx"] for r in rd if is_placeholder_pattern(r["completion"])})
    claim_int("K=50 flags spread over directions", len(dirs), 5)

    rd5 = load("random_direction_matched.json")
    b5 = [r for r in rd5 if r["coefficient"] == -1000.0]
    k5 = sum(is_placeholder_pattern(r["completion"]) for r in b5)
    claim_int("K=5 placeholder flags at c=-1000", k5, 0)
    claim_int("K=5 n at c=-1000", len(b5), 240)
    claim("K=5 Wilson upper %", 100 * wilson_ci(k5, len(b5))[1], 1.6, tol=0.05)

    joint = load("joint_suppression.json")
    jb = [r for r in joint if r["coefficient"] == -500.0]
    jk = sum(is_placeholder_pattern(r["completion"]) for r in jb)
    jlo, jhi = wilson_ci(jk, len(jb))
    claim_int("joint -500 placeholder flags", jk, 7)
    claim_int("joint -500 n", len(jb), 72)
    claim("joint Wilson lower %", 100 * jlo, 4.79, tol=0.005)
    claim("joint Wilson upper %", 100 * jhi, 18.74, tol=0.005)
    claim("gap: joint point / random upper", (jk / len(jb)) / hi, 17.9, tol=0.05)


def f_gemma_matched() -> None:
    section("Matched geometry on Gemma (paper Gemma matched table / Figure 2b)")
    rd = load("gemma_random_direction.json")
    joint = load("gemma_joint_3997_13700_11444.json")
    exp = [
        ("random c=-345", rd, -345.0, 6, 120, 5.0),
        ("joint  c=-200", joint, -200.0, 1, 36, 2.8),
        ("random c=+345", rd, 345.0, 2, 120, 1.7),
        ("joint  c=+200", joint, 200.0, 21, 36, 58.3),
    ]
    for label, src, c, ek, en, erate in exp:
        b = [r for r in src if r["coefficient"] == c and r["prompt"] in CONTROLS]
        k = sum(is_degenerate(r["completion"]) for r in b)
        claim_int(f"{label} controls degen flags", k, ek)
        claim_int(f"{label} n", len(b), en)
        claim(f"{label} rate %", 100 * k / len(b), erate)


def f_gemma_joint_table() -> None:
    section("Gemma joint sweep (paper Gemma joint table)")
    joint = load("gemma_joint_3997_13700_11444.json")
    single = load("gemma_feat3997_narrow.json")
    exp = {  # coef: (joint intro degen %, joint control degen %, single control degen %)
        -400: (97.2, 44.4, 0.0),
        -200: (2.8, 2.8, 0.0),
        -100: (0.0, 0.0, 0.0),
        0: (0.0, 0.0, 0.0),
        100: (0.0, 0.0, 0.0),
        200: (22.2, 58.3, 0.0),
        400: (100.0, 100.0, 38.9),
    }
    for c, e in exp.items():
        bi = [r for r in joint if r["coefficient"] == c and r["prompt"] in INTROS]
        bc = [r for r in joint if r["coefficient"] == c and r["prompt"] in CONTROLS]
        bs = [r for r in single if r["coefficient"] == c and r["prompt"] in CONTROLS]
        claim(f"gemma joint c={c:+d} intro degen %",
              100 * sum(is_degenerate(r["completion"]) for r in bi) / len(bi), e[0])
        claim(f"gemma joint c={c:+d} control degen %",
              100 * sum(is_degenerate(r["completion"]) for r in bc) / len(bc), e[1])
        claim(f"gemma single c={c:+d} control degen %",
              100 * sum(is_degenerate(r["completion"]) for r in bs) / len(bs), e[2])


def f_gemma_coef_table() -> None:
    section("Gemma #3997 dose-response (paper Gemma coefficient table)")
    d = load("gemma_feat3997_narrow.json")
    exp = {  # coef: (disclaimer %, we-voice %, degen %)
        -400: (2.8, 100.0, 13.9),
        -200: (61.1, 100.0, 0.0),
        -100: (77.8, 86.1, 0.0),
        0: (97.2, 5.6, 0.0),
        100: (97.2, 0.0, 0.0),
        200: (97.2, 0.0, 0.0),
        400: (55.6, 0.0, 13.9),
    }
    for c, e in exp.items():
        b = [r for r in d if r["coefficient"] == c and r["prompt"] in INTROS]
        n = len(b)
        claim(f"gemma coef c={c:+d} disclaimer % (n={n})",
              100 * sum(is_disclaimer(r["completion"]) for r in b) / n, e[0])
        claim(f"gemma coef c={c:+d} we-voice %",
              100 * sum(is_we_voice(r["completion"]) for r in b) / n, e[1])
        claim(f"gemma coef c={c:+d} degen %",
              100 * sum(is_degenerate(r["completion"]) for r in b) / n, e[2])


# --------------------------------------------------------------------------
# Cross-model
# --------------------------------------------------------------------------

def s_cross_model() -> None:
    section("Cross-model causal table + dose-response detail")
    # Qwen #29108 on intros: 9-lemma register cluster (C8 + 'mind')
    d = load("feat29108_dose.json")
    exp_q = {-1000: 8.9, -500: 33.3, -250: 77.8, 0: 93.3,
             250: 95.6, 500: 97.8, 1000: 95.6}
    for c, e in exp_q.items():
        b = [r for r in d if r["coefficient"] == c and r["prompt"] in INTROS]
        claim(f"Qwen #29108 c={c:+d} intros cluster-9 % (n={len(b)})",
              100 * sum(cluster_hit(r["completion"], CLUSTER_QWEN_9) for r in b) / len(b), e)
    # Qwen control injection at +1000, 8-lemma Phase-2 cluster, per prompt
    exp_ctrl = {"Write a recipe for tomato soup.": 53.3,
                "Explain how a car engine works.": 66.7,
                "Describe the steps to change a flat tyre.": 73.3}
    for p, e in exp_ctrl.items():
        b = [r for r in d if r["coefficient"] == 1000 and r["prompt"] == p]
        claim(f"Qwen +1000 injection [{p[:24]}...] %",
              100 * sum(cluster_hit(r["completion"], CLUSTER_QWEN) for r in b) / len(b), e)
    deg_all = 100 * sum(is_degenerate(r["completion"]) for r in d) / len(d)
    claim("Qwen #29108 degen % over full sweep", deg_all, 0.8, tol=0.06)

    # Gemma #3997 on intros: 6-lemma cluster
    g = load("gemma_feat3997_narrow.json")
    exp_g = {-400: 16.7, -200: 97.2, -100: 97.2, 0: 100.0,
             100: 97.2, 200: 100.0, 400: 86.1}
    for c, e in exp_g.items():
        b = [r for r in g if r["coefficient"] == c and r["prompt"] in INTROS]
        claim(f"Gemma #3997 c={c:+d} intros cluster-6 % (n={len(b)})",
              100 * sum(cluster_hit(r["completion"], CLUSTER_GEMMA) for r in b) / len(b), e)
    deg_g = 100 * sum(is_degenerate(r["completion"]) for r in g) / len(g)
    claim("Gemma #3997 degen % over full sweep", deg_g, 4.8, tol=0.06)

    # Llama #38565 on intros: 9-lemma cluster
    l = load("llama_feat38565_narrow.json")
    exp_l = {-10: 40.0, -5: 90.0, -2: 100.0, 0: 96.7, 2: 96.7, 5: 100.0, 10: 86.7}
    for c, e in exp_l.items():
        b = [r for r in l if r["coefficient"] == c and r["prompt"] in INTROS]
        claim(f"Llama #38565 c={c:+d} intros cluster-9 % (n={len(b)})",
              100 * sum(cluster_hit(r["completion"], CLUSTER_LLAMA) for r in b) / len(b), e)
    deg_l = sum(is_degenerate(r["completion"]) for r in l)
    claim_int("Llama degen flags over full sweep", deg_l, 0)
    claim_int("Llama total generations", len(l), 420)
    disc_all = {c: 100 * sum(is_disclaimer(r["completion"]) for r in b) / len(b)
                for c, b in by_coef(l).items()}
    claim("Llama disclaimer % all prompts c=-10", disc_all[-10.0], 0.0)
    claim("Llama disclaimer % all prompts c=0", disc_all[0.0], 38.3)
    claim("Llama disclaimer % all prompts c=+10", disc_all[10.0], 13.3)
    # Llama control injection at +10, per prompt
    exp_lc = {"Write a recipe for tomato soup.": 80.0,
              "Explain how a car engine works.": 60.0,
              "Describe the steps to change a flat tyre.": 50.0}
    for p, e in exp_lc.items():
        b = [r for r in l if r["coefficient"] == 10 and r["prompt"] == p]
        claim(f"Llama +10 injection [{p[:24]}...] %",
              100 * sum(cluster_hit(r["completion"], CLUSTER_LLAMA) for r in b) / len(b), e)

    # Pool sizes quoted in Methods
    for pool_dir, sizes in [("data/pools", (1633, 367, 1994)),
                            ("data/pools_gemma", (1953, 47, 1901)),
                            ("data/pools_llama", (752, 248, 989))]:
        s = json.loads((Path(pool_dir) / "pools_summary.json").read_text())
        claim_int(f"{pool_dir} pool A", s["pool_A"], sizes[0])
        claim_int(f"{pool_dir} pool B", s["pool_B"], sizes[1])
        claim_int(f"{pool_dir} pool C", s["pool_C"], sizes[2])


# --------------------------------------------------------------------------
# Appendix: permutation null, relabelling
# --------------------------------------------------------------------------

def app_perm() -> None:
    section("Permutation null (paper: null mean 1.43, actual max 31.55, 0/200)")
    p = json.loads((ACTS / "sae_layer20_permutation.json").read_text())
    actual = p.get("actual_max_raw_diff", p.get("actual_top1_z"))
    feat = p.get("actual_max_feature", p.get("actual_top1_feature"))
    null_mean = p.get("null_max_mean", p.get("null_top1_mean"))
    claim("permutation actual max raw diff", actual, 31.55, tol=0.005)
    claim_int("permutation max-attaining feature", int(feat), 32345)
    claim("permutation null mean", null_mean, 1.43, tol=0.005)
    claim("permutation ratio actual/null", p["ratio_actual_to_null_mean"], 22.0, tol=0.05)
    claim("permutation p-value", p["p_value"], 0.0, tol=1e-9)
    null = p.get("null_max_diffs")
    if null:
        ge = sum(1 for x in null if x >= actual)
        claim_int("permutations >= actual", ge, 0)
        claim_int("n_perm", len(null), 200)
    else:
        print("  (null distribution not stored in this summary file; "
              "re-run src/permutation_test.py to add it)")
    # The headline feature #29108 for contrast: rank-0 by combined z,
    # but NOT the max raw diff (that is #32345, rank 15).
    top = json.loads((ACTS / "sae_layer20_top_features.json").read_text())
    f29108 = next(f for f in top["top_features"] if f["feature_idx"] == 29108)
    claim("#29108 combined z", f29108["combined_z"], 29.49, tol=0.005)
    claim("#29108 raw diff A-C", f29108["diff_AC"], 11.87, tol=0.005)
    claim_int("#29108 rank by combined z", int(f29108["rank"]), 0)


def app_relabel() -> None:
    section("Blind relabelling of #26221 (5 labeller runs)")
    path = DATA / "relabel_26221_results.json"
    r = json.loads(path.read_text())
    runs = 0
    steered_labels = []
    for key in ("baseline_pool_A_4samples_labellers",
                "baseline_pool_B_4samples_labellers",
                "steered_cplus500_12samples_labellers"):
        for run in r.get(key, []):
            runs += 1
            print(f"    [{key}] run {run['run']}: {run['label']!r}")
            if key.startswith("steered"):
                steered_labels.append(run["label"])
    claim_int("labeller runs", runs, 5)
    claim_int("steered labels verbatim-identical",
              int(len(set(steered_labels)) == 1 and len(steered_labels) == 2), 1)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def render(script: str) -> None:
    section(f"Rendering: {script}")
    rc = subprocess.run([sys.executable, script]).returncode
    CHECKS.append((f"render {script}", rc == 0))
    print(f"  exit {rc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None,
                    help="Mirror the report to this file (otherwise stdout only).")
    ap.add_argument("--skip-figures", action="store_true",
                    help="Skip Figure rendering.")
    args = ap.parse_args()

    buf = None
    if args.out:
        import io

        buf = io.StringIO()

        class Tee:
            def __init__(self, *streams):
                self.s = streams

            def write(self, s):
                for x in self.s:
                    x.write(s)

            def flush(self):
                for x in self.s:
                    x.flush()

        sys.stdout = Tee(sys.stdout, buf)

    f_coef_disclaimer_dose()
    f_coef_anchors()
    f_joint_sweep()
    f_norm_probe()
    f_random_K50()
    f_gemma_matched()
    f_gemma_joint_table()
    f_gemma_coef_table()
    s_cross_model()
    app_perm()
    app_relabel()

    if not args.skip_figures:
        render("src/plot_figure1.py")
        render("src/plot_figure2.py")

    n_ok = sum(ok for _, ok in CHECKS)
    section(f"SUMMARY: {n_ok}/{len(CHECKS)} expected values match")
    failures = [label for label, ok in CHECKS if not ok]
    for label in failures:
        print(f"  FAIL: {label}")

    if args.out:
        args.out.write_text(buf.getvalue())
        print(f"\n[wrote] {args.out}")

    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
