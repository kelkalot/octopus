"""E1 prevalence sweep: coefficient axis over ALL top-50 Class-1 features.

Converts the paper's existence proof (1 coherent mode-switch in an N=3
Class-1 sample) into a prevalence estimate: "X of 50 top-ranked features
(Wilson 95% CI [a, b]) exhibit coherent mode switches under the coefficient
axis that their top-context labels do not name."

Two subcommands:

  run     Generate the sweep. For each feature in the layer's top-50
          ranking: c in {-1000, -500, 0, +500, +1000} x 6 intervention
          prompts x 8 samples = 12,000 generations total on Qwen
          (~10 h on an M4 Pro). Incremental save + resume, so the run
          can be interrupted freely.

  screen  Classify each swept feature from the dump with the
          pre-registered decision rule (no model needed). A feature is a
          candidate coherent MODE-SWITCH at coefficient c* iff:
            (a) its baseline-regime marker rate drops >= 30 points from
                its per-sweep peak at c*;
            (b) canonical degeneration at c* is < 10%;
            (c) mean NLL under the unsteered model at c* is < 2x the
                feature's own baseline (c=0) mean NLL (requires the
                optional NLL pass, --nll-file; criterion marked
                "pending" otherwise).
          The baseline-regime marker is the feature's own top-context
          register: the per-feature modal Pool-A lemmas extracted from
          the Phase-3 interpretation dump, falling back to the model's
          Pool-A cluster.
          Monotonic features fail (a); breakdown features fail (b);
          the two anchor classes fall out of the same rule that admits
          the positive case.

  Screen output feeds the confirmation stage (blind relabelling per
  App. relabel) on every flagged feature plus an equal number of
  randomly chosen unflagged features.

Usage:
    # generation sweep (resumable)
    uv run python src/sweep_class1.py run \
        --top-features data/activations/sae_layer20_top_features.json \
        --prompts prompts/intervention_mixed.txt \
        --out data/interventions/class1_sweep

    # optional coherence pass (adds the NLL criterion; a few additional hours)
    for f in data/interventions/class1_sweep/feat*.json; do
        uv run python src/coherence_nll.py --in "$f" --out "${f%.json}_nll.json"
    done

    # screen + report (no GPU)
    uv run python src/sweep_class1.py screen \
        --sweep-dir data/interventions/class1_sweep \
        --pools-dir data/pools \
        --report data/interventions/class1_sweep/screen_report.json
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
        CLUSTER_QWEN, get_nlp, is_degenerate, lemma_noun_set, wilson_ci,
    )
except ImportError:  # invoked as `python src/sweep_class1.py`
    from detectors import (
        CLUSTER_QWEN, get_nlp, is_degenerate, lemma_noun_set, wilson_ci,
    )

COEFFICIENTS = [-1000, -500, 0, 500, 1000]
SAMPLES = 8


def cmd_run(args: argparse.Namespace) -> None:
    top = json.loads(args.top_features.read_text())["top_features"][: args.n_features]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[plan] {len(top)} features x {len(COEFFICIENTS)} coefs x 6 prompts "
          f"x {SAMPLES} samples")
    for i, cand in enumerate(top):
        feat = cand["feature_idx"]
        out = args.out / f"feat{feat}.json"
        if out.exists():
            try:
                n = len(json.loads(out.read_text()))
            except json.JSONDecodeError:
                n = -1
            expected = len(COEFFICIENTS) * 6 * SAMPLES
            if n >= expected:
                print(f"[skip] #{feat} complete ({n} records)")
                continue
            print(f"[redo] #{feat} incomplete ({n} records)")
        print(f"[run ] {i+1}/{len(top)} feature #{feat}")
        cmd = [
            sys.executable, "src/steer.py",
            "--prompts", str(args.prompts),
            "--feature", str(feat),
            "--layer", str(args.layer),
            "--model", args.model,
            "--release", args.release,
            "--coefficients", *[str(c) for c in COEFFICIENTS],
            "--samples", str(SAMPLES),
            "--out", str(out),
        ]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"[fail] steer.py exited {rc} on #{feat}; continuing")


def _feature_marker_lemmas(interp_md: Path, feature: int) -> set[str] | None:
    """Extract the feature's own register lemmas from its top-activating
    Pool-A samples in the Phase-3 interpretation dump: noun lemmas that
    appear in >= 3 of its listed top-5 Pool-A samples."""
    if not interp_md.exists():
        return None
    text = interp_md.read_text(encoding="utf-8")
    marker = f"## rank "
    blocks = text.split(marker)
    block = next((b for b in blocks if f"feature #{feature}" in b.split("\n", 1)[0]),
                 None)
    if block is None:
        return None
    pool_a = block.split("**Top")[1] if "**Top" in block else ""
    samples = [s.split("\n\n")[-1] for s in pool_a.split("_act=")[1:6]]
    if not samples:
        return None
    nlp = get_nlp()
    counts: dict[str, int] = defaultdict(int)
    for s in samples:
        for lem in lemma_noun_set(nlp, s):
            counts[lem] += 1
    markers = {l for l, k in counts.items() if k >= 3 and len(l) > 2}
    return markers or None


def screen_sweep(sweep_dir: Path, pools_dir: Path,
                 drop_threshold: float = 0.30,
                 degen_threshold: float = 0.10) -> dict:
    """Apply the pre-registered mode-switch rule to every swept feature and
    return the summary dict. Used by cmd_screen and by the regeneration
    script, so the paper's prevalence number is re-derived from the dump by
    the same code path."""
    nlp = get_nlp()
    cluster_default = set(CLUSTER_QWEN)
    interp_md = pools_dir.parent / "activations" / "sae_layer20_interpretations.md"

    results = []
    files = sorted(sweep_dir.glob("feat*.json"))
    files = [f for f in files if not f.name.endswith("_nll.json")]
    print(f"[screen] {len(files)} sweep files")
    for path in files:
        feat = int(path.stem.replace("feat", ""))
        records = json.loads(path.read_text())
        markers = _feature_marker_lemmas(interp_md, feat) or cluster_default
        nll_path = path.with_name(path.stem + "_nll.json")
        nll_by = defaultdict(list)
        if nll_path.exists():
            for r in json.loads(nll_path.read_text()):
                if r.get("nll") is not None and r["nll"] == r["nll"]:
                    nll_by[r["coefficient"]].append(r["nll"])

        per_coef = {}
        for c, b in sorted(
            ((c, [r for r in records if r["coefficient"] == c])
             for c in {r["coefficient"] for r in records})
        ):
            marker_rate = mean(
                bool(markers & lemma_noun_set(nlp, r["completion"])) for r in b
            )
            degen_rate = mean(is_degenerate(r["completion"]) for r in b)
            per_coef[c] = {
                "marker_rate": marker_rate,
                "degen_rate": degen_rate,
                "mean_nll": mean(nll_by[c]) if nll_by.get(c) else None,
                "n": len(b),
            }

        peak = max(v["marker_rate"] for v in per_coef.values())
        nll_base = per_coef.get(0.0, {}).get("mean_nll")
        flags = []
        for c, v in per_coef.items():
            if c == 0.0:
                continue
            drop = (peak - v["marker_rate"]) >= drop_threshold
            coherent = v["degen_rate"] < degen_threshold
            if v["mean_nll"] is not None and nll_base:
                nll_ok = v["mean_nll"] < 2 * nll_base
            else:
                nll_ok = None  # pending the NLL pass
            if drop and coherent and (nll_ok is not False):
                flags.append({"coef": c, "marker_rate": v["marker_rate"],
                              "degen_rate": v["degen_rate"],
                              "nll_criterion": nll_ok})
        classification = (
            "mode_switch_candidate" if flags
            else "breakdown" if max(v["degen_rate"] for v in per_coef.values()) >= 0.5
            else "monotonic_or_flat"
        )
        results.append({
            "feature": feat,
            "markers": sorted(markers),
            "per_coef": {str(c): v for c, v in per_coef.items()},
            "flagged_coefs": flags,
            "classification": classification,
        })
        print(f"  #{feat}: {classification}"
              + (f" at {[f['coef'] for f in flags]}" if flags else ""))

    k = sum(r["classification"] == "mode_switch_candidate" for r in results)
    n = len(results)
    lo, hi = wilson_ci(k, n)
    return {
        "n_features": n,
        "mode_switch_candidates": k,
        "rate": k / n if n else 0,
        "wilson_95": [lo, hi],
        "decision_rule": {
            "marker_drop_points": drop_threshold,
            "degen_below": degen_threshold,
            "nll_below_x_baseline": 2.0,
        },
        "results": results,
    }


def cmd_screen(args: argparse.Namespace) -> None:
    summary = screen_sweep(args.sweep_dir, args.pools_dir,
                           args.drop_threshold, args.degen_threshold)
    k, n = summary["mode_switch_candidates"], summary["n_features"]
    lo, hi = summary["wilson_95"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2))
    print(f"\n[headline] {k} of {n} Class-1 features are coherent mode-switch "
          f"candidates (Wilson 95% [{100*lo:.1f}, {100*hi:.1f}]%)")
    print(f"[note] candidates require the blind-relabelling confirmation stage "
          f"before the label-scoped claim is made")
    print(f"[wrote] {args.report}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="generate the sweep (GPU, ~10 h)")
    r.add_argument("--top-features", type=Path,
                   default=Path("data/activations/sae_layer20_top_features.json"))
    r.add_argument("--n-features", type=int, default=50)
    r.add_argument("--prompts", type=Path,
                   default=Path("prompts/intervention_mixed.txt"))
    r.add_argument("--model", default="Qwen/Qwen3-1.7B")
    r.add_argument("--release", default="qwen-scope-3-1.7b-base-w32k-l50")
    r.add_argument("--layer", type=int, default=20)
    r.add_argument("--out", type=Path,
                   default=Path("data/interventions/class1_sweep"))
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("screen", help="classify swept features (no GPU)")
    s.add_argument("--sweep-dir", type=Path,
                   default=Path("data/interventions/class1_sweep"))
    s.add_argument("--pools-dir", type=Path, default=Path("data/pools"))
    s.add_argument("--drop-threshold", type=float, default=0.30,
                   help="required marker-rate drop from peak (fraction)")
    s.add_argument("--degen-threshold", type=float, default=0.10,
                   help="max canonical degeneration rate at the flagged coef")
    s.add_argument("--report", type=Path,
                   default=Path("data/interventions/class1_sweep/screen_report.json"))
    s.set_defaults(func=cmd_screen)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
