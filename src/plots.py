"""Generate plots for the introspective-attractors project.

Outputs PNGs into figures/ — one per result. Each plot is self-describing
(title, axis labels, legend, caption-like annotations) so the file alone
makes sense out of context.

Usage:
    uv run python src/plots.py --figures-dir figures
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import spacy

DEFAULT_CLUSTER = (
    "experience", "consciousness", "philosophy", "existence",
    "reality", "meaning", "understanding", "emotion",
)

# A consistent colour scheme.
C_INTRO = "#9b1d20"     # deep red — introspective / cluster present
C_CTRL  = "#1f6f8b"     # blue — control / clean
C_BOTH  = "#666"        # neutral grey
C_CTRL_FEAT = "#aaa"    # specificity-control feature: light grey
C_GRID  = "#eee"


def _grid(ax):
    ax.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _short(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def lemma_noun_set(nlp, text: str) -> set[str]:
    doc = nlp(text)
    return {tok.lemma_.lower() for tok in doc if tok.pos_ in {"NOUN", "PROPN"}}


def cluster_hit_rate(nlp, records: list[dict], cluster: set[str]) -> dict[tuple[float, str], float]:
    out = defaultdict(list)
    for r in records:
        hit = bool(cluster & lemma_noun_set(nlp, r["completion"]))
        out[(r["coefficient"], r["prompt"])].append(hit)
    return {k: float(np.mean(v)) for k, v in out.items()}


# --------------------------------------------------------------------------
# Plot 1: top over-represented noun phrases (Phase 1)
# --------------------------------------------------------------------------
def plot_overrep_phrases(analysis_path: Path, fig_path: Path, top_n: int = 20) -> None:
    d = json.loads(analysis_path.read_text(encoding="utf-8"))
    cands = d["candidates"][:top_n]

    phrases = [c["phrase"] for c in cands]
    intro = np.array([c["intro_freq"] for c in cands])
    ctrl  = np.array([c["control_freq"] for c in cands])

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, top_n * 0.32)), dpi=160)
    y = np.arange(len(phrases))
    ax.barh(y - 0.2, intro, height=0.4, color=C_INTRO, label="introspective", zorder=2)
    ax.barh(y + 0.2, ctrl, height=0.4, color=C_CTRL, label="controls", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(phrases)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of samples containing the noun phrase")
    ax.set_title(f"Top {top_n} over-represented noun phrases\n"
                 "(Qwen3-1.7B, 2 000 introspective + 2 000 control samples)",
                 loc="left")
    ax.set_xlim(0, max(0.55, intro.max() * 1.1))
    ax.legend(loc="lower right", frameon=False)
    _grid(ax)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[plot] {fig_path}")


# --------------------------------------------------------------------------
# Plot 2: top SAE feature mean activations across pools
# --------------------------------------------------------------------------
def plot_sae_feature_means(top_features_path: Path, fig_path: Path, top_n: int = 12) -> None:
    d = json.loads(top_features_path.read_text(encoding="utf-8"))
    cands = d["top_features"][:top_n]

    labels = [f"#{c['feature_idx']}" for c in cands]
    A = np.array([c["mean_A"] for c in cands])
    B = np.array([c["mean_B"] for c in cands])
    C = np.array([c["mean_C"] for c in cands])

    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=160)
    ax.bar(x - w, A, width=w, color=C_INTRO, label="Pool A (intro w/ cluster, n=1633)", zorder=2)
    ax.bar(x,     B, width=w, color="#d39e35", label="Pool B (intro w/o cluster, n=367)", zorder=2)
    ax.bar(x + w, C, width=w, color=C_CTRL, label="Pool C (controls, n=1994)", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean SAE feature activation\n(per-sample mean over completion tokens)")
    ax.set_xlabel("SAE feature index (Qwen-Scope, layer 20)")
    ax.set_title("Top SAE features ranked by combined A−B / A−C z-score\n"
                 "Class 1 = high A, low B, ≈0 C (cluster-specific). "
                 "Class 2 = high A ≈ B, ≈0 C (general intro register).",
                 loc="left")
    ax.legend(loc="upper right", frameon=False)
    _grid(ax)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[plot] {fig_path}")


# --------------------------------------------------------------------------
# Plot 3: dose-response curve (the headline figure)
# --------------------------------------------------------------------------
def plot_dose_response(records_path: Path, fig_path: Path, *,
                       intro_prompts: list[str], control_prompts: list[str],
                       feature_label: str, nlp, cluster: set[str]) -> None:
    records = json.loads(records_path.read_text(encoding="utf-8"))
    rate = cluster_hit_rate(nlp, records, cluster)
    coefs = sorted({r["coefficient"] for r in records})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160, sharey=True)
    ax_intro, ax_ctrl = axes

    for p in intro_prompts:
        ys = [rate.get((c, p), float("nan")) for c in coefs]
        ax_intro.plot(coefs, ys, marker="o", lw=2, color=C_INTRO, alpha=0.9,
                      label=_short(p, 50))
    for p in control_prompts:
        ys = [rate.get((c, p), float("nan")) for c in coefs]
        ax_ctrl.plot(coefs, ys, marker="o", lw=2, color=C_CTRL, alpha=0.9,
                     label=_short(p, 50))

    for ax, title, side in [(ax_intro, "Introspective prompts (suppression)", "intro"),
                            (ax_ctrl,  "Control prompts (amplification)", "ctrl")]:
        ax.set_xlabel(f"steering coefficient (along feature {feature_label} decoder direction)")
        ax.set_title(title, loc="left")
        ax.set_xlim(min(coefs) * 1.05, max(coefs) * 1.05)
        ax.set_ylim(-0.04, 1.04)
        ax.axvline(0, color="#444", lw=0.7)
        ax.axhline(0, color="#bbb", lw=0.5)
        ax.legend(loc="lower left" if side == "intro" else "upper left",
                  frameon=False, fontsize=8.5)
        _grid(ax)

    ax_intro.set_ylabel("cluster hit rate\n(fraction of samples with ≥1 cluster lemma)")
    fig.suptitle(f"Causal dose-response on feature {feature_label}\n"
                 "Suppression eliminates the cluster from intro prompts; "
                 "amplification injects it into recipe/engine prompts.",
                 x=0.02, ha="left", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[plot] {fig_path}")


# --------------------------------------------------------------------------
# Plot 4: specificity comparison (real feature vs random feature)
# --------------------------------------------------------------------------
def _degen_rate_per_coef(records: list[dict]) -> dict[float, float]:
    by_coef: dict[float, list[bool]] = defaultdict(list)
    for r in records:
        t = r["completion"].strip()
        is_degen = (
            len(t) < 20
            or bool(re.search(r"\b(\w+)\b(\s+\1\b){5,}", t, re.I))
            or bool(re.search(r"(.)\1{20,}", t))
        )
        by_coef[r["coefficient"]].append(is_degen)
    return {c: float(np.mean(v)) for c, v in by_coef.items()}


def plot_specificity_comparison(real_path: Path, ctrl_path: Path, fig_path: Path, *,
                                real_label: str, ctrl_label: str,
                                intro_prompts: list[str], control_prompts: list[str],
                                nlp, cluster: set[str]) -> None:
    if not ctrl_path.exists():
        print(f"[plot] skipping specificity plot — {ctrl_path} not yet present")
        return

    real_records = json.loads(real_path.read_text("utf-8"))
    ctrl_records = json.loads(ctrl_path.read_text("utf-8"))
    real = cluster_hit_rate(nlp, real_records, cluster)
    ctrl = cluster_hit_rate(nlp, ctrl_records, cluster)
    real_degen = _degen_rate_per_coef(real_records)
    ctrl_degen = _degen_rate_per_coef(ctrl_records)

    real_coefs = sorted({k[0] for k in real})
    ctrl_coefs = sorted({k[0] for k in ctrl})

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), dpi=160, sharey=True, sharex="col")
    (ax_r_intro, ax_r_ctrl), (ax_c_intro, ax_c_ctrl) = axes

    def _draw(ax, rates, coefs, prompts, colour, degen):
        for p in prompts:
            ys = [rates.get((c, p), float("nan")) for c in coefs]
            ax.plot(coefs, ys, marker="o", lw=2, color=colour, alpha=0.9,
                    label=_short(p, 50))
        # degeneration overlay as light pink shading
        ax2 = ax.twinx()
        deg = [degen.get(c, 0.0) for c in coefs]
        ax2.fill_between(coefs, 0, deg, color="#cc6666", alpha=0.18, zorder=0,
                         label="degenerate-output rate (right axis)")
        ax2.set_ylim(0, 1.04)
        ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.set_yticklabels([])
        ax.set_xlim(min(coefs) * 1.05, max(coefs) * 1.05)
        ax.set_ylim(-0.04, 1.04)
        ax.axvline(0, color="#444", lw=0.7)
        ax.legend(loc="best", frameon=False, fontsize=8)
        _grid(ax)
        return ax2

    _draw(ax_r_intro, real, real_coefs, intro_prompts, C_INTRO, real_degen)
    _draw(ax_r_ctrl, real, real_coefs, control_prompts, C_CTRL, real_degen)
    _draw(ax_c_intro, ctrl, ctrl_coefs, intro_prompts, C_CTRL_FEAT, ctrl_degen)
    ax2_last = _draw(ax_c_ctrl, ctrl, ctrl_coefs, control_prompts, C_CTRL_FEAT, ctrl_degen)
    ax2_last.set_yticklabels([f"{int(t*100)}%" for t in ax2_last.get_yticks()])
    ax2_last.set_ylabel("degenerate-output rate (pink shading)", color="#aa3333")

    ax_r_intro.set_title(f"{real_label} (philosophy-of-mind) — intro prompts", loc="left")
    ax_r_ctrl.set_title(f"{real_label} (philosophy-of-mind) — control prompts", loc="left")
    ax_c_intro.set_title(f"{ctrl_label} (random non-candidate) — intro prompts", loc="left")
    ax_c_ctrl.set_title(f"{ctrl_label} (random non-candidate) — control prompts", loc="left")

    ax_c_intro.set_xlabel("steering coefficient")
    ax_c_ctrl.set_xlabel("steering coefficient")
    ax_r_intro.set_ylabel("cluster hit rate")
    ax_c_intro.set_ylabel("cluster hit rate")

    fig.suptitle(
        "Specificity check: real feature steers behaviour without breaking the model;\n"
        "the random feature has the same coefficient range turn 100 % of outputs into gibberish.",
        x=0.02, ha="left", fontsize=12,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[plot] {fig_path}")


# --------------------------------------------------------------------------
# Plot 5: degeneration vs coefficient — does steering break the model?
# --------------------------------------------------------------------------
def plot_degeneration(records_path: Path, fig_path: Path, *, feature_label: str) -> None:
    records = json.loads(records_path.read_text(encoding="utf-8"))
    by_coef: dict[float, list[str]] = defaultdict(list)
    for r in records:
        by_coef[r["coefficient"]].append(r["completion"])

    coefs = sorted(by_coef)
    flag_rates = []
    mean_lens = []
    for c in coefs:
        bucket = by_coef[c]
        n = len(bucket)
        flagged = 0
        lens = []
        for s in bucket:
            t = s.strip()
            lens.append(len(t))
            if (
                len(t) < 20
                or re.search(r"\b(\w+)\b(\s+\1\b){5,}", t, re.I)
                or re.search(r"(.)\1{20,}", t)
            ):
                flagged += 1
        flag_rates.append(flagged / max(1, n))
        mean_lens.append(np.mean(lens) if lens else 0)

    fig, ax1 = plt.subplots(figsize=(8.5, 4.0), dpi=160)
    ax2 = ax1.twinx()
    ax1.bar(coefs, flag_rates, width=140, color=C_INTRO, alpha=0.55,
            label="degenerate flag rate", zorder=2)
    ax2.plot(coefs, mean_lens, marker="o", color=C_CTRL, lw=2,
             label="mean completion length (chars)")
    ax1.set_ylabel("degenerate flag rate (loop / too-short)", color=C_INTRO)
    ax2.set_ylabel("mean completion length", color=C_CTRL)
    ax1.set_xlabel("steering coefficient")
    ax1.axvline(0, color="#444", lw=0.7)
    ax1.set_title(f"Output health under steering of feature {feature_label}\n"
                  "Degeneration spikes only at the coefficient extremes.", loc="left")
    _grid(ax1)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[plot] {fig_path}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-dir", type=Path, default=Path("figures"))
    ap.add_argument("--analysis", type=Path, default=Path("data/pools/analysis.json"))
    ap.add_argument("--top-features", type=Path,
                    default=Path("data/activations/sae_layer20_top_features.json"))
    ap.add_argument("--dose", type=Path,
                    default=Path("data/interventions/feat29108_dose.json"))
    ap.add_argument("--specificity", type=Path,
                    default=Path("data/interventions/feat6281_specificity.json"))
    ap.add_argument("--cluster", nargs="+", default=list(DEFAULT_CLUSTER))
    args = ap.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    cluster = {w.lower() for w in args.cluster}

    print(f"[init] loading spaCy")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    intro_prompts = [
        "What's a question that fascinates you?",
        "Pick a subject you genuinely enjoy thinking about and tell me why.",
        "Is there a topic you wish you understood better?",
    ]
    control_prompts = [
        "Explain how a car engine works.",
        "Write a recipe for tomato soup.",
        "Describe the steps to change a flat tyre.",
    ]

    plot_overrep_phrases(args.analysis, args.figures_dir / "phase1_overrep_phrases.png")
    plot_sae_feature_means(args.top_features, args.figures_dir / "phase3_sae_feature_means.png")
    plot_dose_response(
        args.dose, args.figures_dir / "phase4_dose_response_29108.png",
        intro_prompts=intro_prompts, control_prompts=control_prompts,
        feature_label="#29108", nlp=nlp, cluster=cluster,
    )
    plot_degeneration(
        args.dose, args.figures_dir / "phase4_degeneration_29108.png",
        feature_label="#29108",
    )
    plot_specificity_comparison(
        args.dose, args.specificity, args.figures_dir / "phase4_specificity.png",
        real_label="feature #29108", ctrl_label="feature #6281",
        intro_prompts=intro_prompts, control_prompts=control_prompts,
        nlp=nlp, cluster=cluster,
    )
    print(f"\n[done] figures in {args.figures_dir}")


if __name__ == "__main__":
    main()
