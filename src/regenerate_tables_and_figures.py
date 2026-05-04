"""Regenerate every numerical claim, table, and figure in the paper from the
bundled JSON sample dumps.

This script does NOT touch any model — it reads only the released
`data/interventions/*.json`, `data/activations/*.json`, and
`data/pools*/pools_summary.json` files, and re-derives every percentage,
Wilson CI, norm ratio, and figure that appears in the body and appendix of
the paper. Intended as the headline reproducibility artefact: a reviewer
runs it once and verifies every numeric claim in <5 minutes.

Usage:
    python src/regenerate_tables_and_figures.py [--out FILE]

By default writes a side-by-side report (paper claim ↔ regenerated value)
to stdout. With --out, also saves the report to disk.

Coverage:
    Section 4.1 — Coefficient axis on #26221 (Table 1 dose-response,
                  #22082 / #2932 falsifying-case anchors)
    Section 4.2 — Joint-condition axis on {#29108, #26221, #4405}
                  (Table 2 in App., headline 89% control injection)
                  + Gemma replication on {#3997, #13700, #11444}
    Section 4.3 — Norm-probe (Table 3 in App., 1.57/1.51/1.56 geometry)
                  + K=50 random-direction control (6 of 2400, 0.54% Wilson)
                  + Gemma matched-geometry (1.7% vs 58.3%)
    Section 5   — Cross-model causal validation (Table tab:cross-model-causal)
                  + per-coefficient breakdowns (App. cross-model-details)
    App. perm   — Permutation null (1.43 mean, 31.55 actual, 22× ratio)
    App. relabel — 5 blind labellers, verbatim agreement at T=1.0
    Figure 1    — calls plot_figure1.py
    Figure 2    — calls plot_figure2.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

DATA = Path("data/interventions")
POOLS = Path("data/pools")
ACTS = Path("data/activations")

# Prompt sets used throughout the paper.
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

# Cluster definitions used throughout the paper.
CLUSTER_QWEN_9 = (
    "consciousness reality experience meaning existence philosophy "
    "understanding emotion mind"
).split()
CLUSTER_QWEN_4_STRICT = "consciousness reality existence philosophy".split()


def is_degen_canonical(text: str) -> bool:
    """Three-rule regex degeneration detector documented in §A.coherence."""
    t = text.strip()
    if len(t) < 20:
        return True
    if re.search(r"\b(\w+)\b(\s+\1\b){5,}", t, re.I):
        return True
    if re.search(r"(.)\1{20,}", t):
        return True
    return False


def is_disclaimer(text: str) -> bool:
    """Disclaimer-rate regex from src/analyse_disclaimer.py."""
    patterns = [
        r"\bas an AI\b", r"\bI am an AI\b", r"\bI'm an AI\b",
        r"\blanguage model\b", r"\bAI assistant\b",
        r"\bI (?:don'?t|do not) (?:have|experience|possess) "
        r"(?:personal |subjective |any )?"
        r"(?:feelings?|emotions?|thoughts?|consciousness|"
        r"experiences?|opinions?|preferences?)\b",
        r"\bI (?:can'?t|cannot|am not able to) "
        r"(?:feel|experience|have|possess) "
        r"(?:feelings?|emotions?|consciousness|subjective)",
        r"\bI lack (?:feelings?|emotions?|consciousness|subjective)",
        r"\bnot (?:capable of|able to) (?:feeling|experiencing|having)",
        r"\bI'?m (?:just |only |simply )?"
        r"(?:an AI|a language model|a chatbot|a machine|a computer program)",
    ]
    rx = re.compile("|".join(patterns), flags=re.IGNORECASE)
    return bool(rx.search(text[:200]))


def is_placeholder_pattern(text: str) -> bool:
    """Strict placeholder-pattern detector from §4.3."""
    code_paren = re.findall(r"\(\s*[A-Z]{2,5}(?:\s*[A-Z\d]+)?\s*\)", text)
    if len(code_paren) >= 2:
        return True
    if re.search(r"\b[Vv]c\.\s*\d+\+?", text):
        return True
    return False


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, centre - half), centre + half)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def claim(label: str, value, expected: str | None = None) -> None:
    if isinstance(value, float):
        v = f"{value:.2f}"
    elif isinstance(value, tuple) and len(value) == 2:
        v = f"[{value[0]*100:.2f}%, {value[1]*100:.2f}%]"
    else:
        v = str(value)
    pad = "" if expected is None else f"  paper: {expected}"
    print(f"  {label:60s} = {v}{pad}")


def f41_disclaimer_dose() -> None:
    section("§4.1  #26221 disclaimer dose-response (paper Table 1)")
    d = json.loads((DATA / "f2_identity_26221.json").read_text())
    by_coef = defaultdict(list)
    for r in d:
        by_coef[r["coefficient"]].append(r)
    for c in sorted(by_coef):
        b = by_coef[c]
        n = len(b)
        disc = sum(is_disclaimer(r["completion"]) for r in b) / n * 100
        clust = sum(any(w in r["completion"].lower() for w in CLUSTER_QWEN_9)
                    for r in b) / n * 100
        deg = sum(is_degen_canonical(r["completion"]) for r in b) / n * 100
        print(f"  c={c:+8.0f}  n={n:>3}  disclaimer {disc:5.1f}%  "
              f"cluster {clust:5.1f}%  degen {deg:5.1f}%")


def f41_falsifying_anchors() -> None:
    section("§4.1  Falsifying-case anchors: #22082 (monotonic), #2932 (breakdown)")
    for feat, fname in [(22082, "feat22082_dose.json"),
                        (2932, "feat2932_dose.json")]:
        d = json.loads((DATA / fname).read_text())
        by_coef = defaultdict(list)
        for r in d:
            by_coef[r["coefficient"]].append(r)
        print(f"  feature #{feat}:")
        for c in sorted(by_coef):
            b = by_coef[c]
            n = len(b)
            clust = sum(any(w in r["completion"].lower() for w in CLUSTER_QWEN_9)
                        for r in b) / n * 100
            deg = sum(is_degen_canonical(r["completion"]) for r in b) / n * 100
            print(f"    c={c:+8.0f}  n={n:>3}  cluster {clust:5.1f}%  "
                  f"degen {deg:5.1f}%")


def f42_joint_sweep() -> None:
    section("§4.2  Joint sweep on {#29108, #26221, #4405} (Table 2 in appendix)")
    joint = json.loads((DATA / "joint_suppression.json").read_text())
    nll = json.loads((DATA / "joint_suppression_nll.json").read_text())
    nll_by_coef = defaultdict(list)
    for r in nll:
        if "nll" in r:
            nll_by_coef[r["coefficient"]].append(r["nll"])

    print("  coef  intros(strict 4-lemma)  controls(strict 4-lemma)  "
          "regex-degen  NLL")
    for c in [-1500, -1000, -500, 0, 500, 1000]:
        bi = [r for r in joint if r["coefficient"] == c
              and r["prompt"] in INTROS]
        bc = [r for r in joint if r["coefficient"] == c
              and r["prompt"] in CONTROLS]
        ball = bi + bc
        hi = sum(any(w in r["completion"].lower()
                     for w in CLUSTER_QWEN_4_STRICT) for r in bi)
        hc = sum(any(w in r["completion"].lower()
                     for w in CLUSTER_QWEN_4_STRICT) for r in bc)
        deg = sum(is_degen_canonical(r["completion"]) for r in ball)
        nll_mean = mean(nll_by_coef.get(float(c), [float("nan")]))
        print(f"  {c:+5d}  {hi}/36 = {100*hi/36:5.1f}%   "
              f"{hc}/36 = {100*hc/36:5.1f}%   "
              f"{deg}/72 = {100*deg/72:5.1f}%   NLL={nll_mean:.2f}")

    section("§4.2  Joint vs single-feature comparison (paper headline)")
    single = json.loads((DATA / "feat29108_dose.json").read_text())
    for c in [-500, 500]:
        joint_b = [r for r in joint if r["coefficient"] == c
                   and r["prompt"] in CONTROLS]
        single_b = [r for r in single if r["coefficient"] == c
                    and r["prompt"] in CONTROLS]
        h_j = sum(any(w in r["completion"].lower()
                      for w in CLUSTER_QWEN_4_STRICT) for r in joint_b)
        h_s = sum(any(w in r["completion"].lower()
                      for w in CLUSTER_QWEN_4_STRICT) for r in single_b)
        print(f"  c={c:+5d}  joint controls cluster {h_j}/{len(joint_b)} = "
              f"{100*h_j/max(1,len(joint_b)):5.1f}%  vs  "
              f"single controls cluster {h_s}/{len(single_b)} = "
              f"{100*h_s/max(1,len(single_b)):5.1f}%")


def f43_norm_probe() -> None:
    section("§4.3  Norm probe (paper Table 3 in appendix)")
    s = json.loads((DATA / "single_29108_norm_probe.json").read_text())
    j = json.loads((DATA / "joint_norm_probe.json").read_text())
    print("  coef     single ‖h‖/base   joint ‖h‖/base   single cos   joint cos")
    s_by = {r["coef"]: r for r in s["results"]}
    j_by = {r["coef"]: r for r in j["results"]}
    for c in [-1500, -1000, -500, 0, 500, 1000]:
        sr = s_by.get(float(c))
        jr = j_by.get(float(c))
        sn = sr["mean_norm_ratio"] if sr else float("nan")
        jn = jr["mean_norm_ratio"] if jr else float("nan")
        sc = sr["mean_cosine"] if sr else float("nan")
        jc = jr["mean_cosine"] if jr else float("nan")
        print(f"  c={c:+5d}    {sn:.3f}            {jn:.3f}           "
              f"{sc:.3f}        {jc:.3f}")


def f43_random_K50() -> None:
    section("§4.3  K=50 random-direction control (App. K=50)")
    rd = json.loads((DATA / "random_direction_K50_at_c-1000.json").read_text())
    flagged = [r for r in rd if is_placeholder_pattern(r["completion"])]
    n = len(rd)
    k = len(flagged)
    lo, hi = wilson_ci(k, n)
    dirs = sorted({r["direction_idx"] for r in flagged})
    print(f"  K=50 random direction at c=-1000: {k} of {n} placeholder")
    print(f"    rate {100*k/n:.2f}%  Wilson 95% [{100*lo:.2f}, {100*hi:.2f}]%")
    print(f"    flags spread across {len(dirs)} of 50 directions: {dirs}")

    joint = json.loads((DATA / "joint_suppression.json").read_text())
    j_all = [r for r in joint if r["coefficient"] == -500.0]  # all 6 prompts
    j_k = sum(is_placeholder_pattern(r["completion"]) for r in j_all)
    j_lo, j_hi = wilson_ci(j_k, len(j_all))
    print(f"  Joint c=-500 (all 6 prompts): {j_k} of {len(j_all)} = "
          f"{100*j_k/len(j_all):.1f}%  Wilson 95% "
          f"[{100*j_lo:.2f}, {100*j_hi:.2f}]%")
    print(f"  CI-separated gap: joint point / random Wilson upper = "
          f"{(j_k/len(j_all)) / hi:.1f}×")


def f43_gemma_matched() -> None:
    section("§4.3  Gemma matched-geometry control (Figure 2b)")
    rd = json.loads((DATA / "gemma_random_direction.json").read_text())
    joint = json.loads((DATA / "gemma_joint_3997_13700_11444.json").read_text())

    for label, src, c in [
        ("random c=-345 controls", rd, -345.0),
        ("random c=+345 controls", rd, 345.0),
        ("joint  c=-200 controls", joint, -200.0),
        ("joint  c=+200 controls", joint, 200.0),
    ]:
        b = [r for r in src if r["coefficient"] == c
             and r["prompt"] in CONTROLS]
        k = sum(is_degen_canonical(r["completion"]) for r in b)
        n = len(b)
        lo, hi = wilson_ci(k, n)
        print(f"  {label:30s}  {k}/{n} = {100*k/n:5.1f}%  "
              f"Wilson 95% [{100*lo:.1f}, {100*hi:.1f}]%")


def s5_cross_model() -> None:
    section("§5  Cross-model causal validation (Table tab:cross-model-causal)")
    for model, pool_dir, dose_file, intros_label, coefs in [
        ("Qwen #29108",  "data/pools",        "feat29108_dose.json",       INTROS, [-1000, 0, 1000]),
        ("Gemma #3997",  "data/pools_gemma",  "gemma_feat3997_narrow.json", INTROS, [-400,   0, 400]),
        ("Llama #38565", "data/pools_llama",  "llama_feat38565_narrow.json", INTROS, [-10,    0, 10]),
    ]:
        cluster = json.loads(
            (Path(pool_dir) / "pools_summary.json").read_text()
        )["cluster"]
        d = json.loads((DATA / dose_file).read_text())
        intros_in_data = [p for p in {r["prompt"] for r in d}
                          if any(x in p.lower()
                                 for x in ["fascinates", "pick a subject",
                                           "wish you understood"])]
        print(f"  {model} (intros, cluster {cluster}):")
        for c in coefs:
            b = [r for r in d if r["coefficient"] == c
                 and r["prompt"] in intros_in_data]
            n = len(b)
            h = sum(any(re.search(rf"\b{w}\b", r["completion"], re.I)
                        for w in cluster) for r in b)
            print(f"    c={c:+5d}  intros cluster {h}/{n} = {100*h/n:.1f}%")


def app_perm() -> None:
    section("App. perm — Permutation null (paper: 1.43 mean, 31.55 actual)")
    p = json.loads((ACTS / "sae_layer20_permutation.json").read_text())
    null = p.get("null_max_diffs") or p.get("null_means") or p.get("null") or []
    actual = p.get("actual_top1") or p.get("actual_max") or p.get("actual")
    if null:
        m = float(np.mean(null))
        ge = sum(1 for x in null if x >= actual) if actual else 0
        print(f"  null mean: {m:.2f}  (paper: 1.43)")
        print(f"  actual top-1: {actual}  (paper: 31.55)")
        print(f"  permutations >= actual: {ge}/{len(null)}  (paper: 0/200)")
        print(f"  ratio: {actual / m:.1f}×  (paper: 22×)")
    else:
        print("  (could not parse permutation file; check schema)")


def app_relabel() -> None:
    section("App. relabel — Blind labellers on #26221")
    p = DATA / "relabel_26221_results.json"
    if not p.exists():
        print("  (relabel_26221_results.json not found)")
        return
    r = json.loads(p.read_text())
    print(f"  feature: {r['feature_id']}")
    for k in ("baseline_pool_A_4samples_labellers",
              "baseline_pool_B_4samples_labellers",
              "steered_cplus500_12samples_labellers"):
        for run in r.get(k, []):
            print(f"  [{k}] run {run['run']}: {run['label']!r}")


def call(script: str) -> None:
    section(f"Rendering: {script}")
    print(f"  $ python {script}")
    rc = subprocess.run([sys.executable, script]).returncode
    print(f"  exit {rc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None,
                    help="Mirror the report to this file (otherwise stdout only).")
    ap.add_argument("--skip-figures", action="store_true",
                    help="Skip Figure 1 and Figure 2 rendering.")
    args = ap.parse_args()

    if args.out:
        # Tee stdout to the file as well
        import io
        buf = io.StringIO()

        class Tee:
            def __init__(self, *streams): self.s = streams
            def write(self, s):
                for x in self.s:
                    x.write(s)
            def flush(self):
                for x in self.s:
                    x.flush()
        sys.stdout = Tee(sys.stdout, buf)

    f41_disclaimer_dose()
    f41_falsifying_anchors()
    f42_joint_sweep()
    f43_norm_probe()
    f43_random_K50()
    f43_gemma_matched()
    s5_cross_model()
    app_perm()
    app_relabel()

    if not args.skip_figures:
        call("src/plot_figure1.py")
        call("src/plot_figure2.py")

    if args.out:
        args.out.write_text(buf.getvalue())
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
