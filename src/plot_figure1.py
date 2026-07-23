"""Build the three-panel steering-grid evidence figure for the paper.

Panel A - coefficient axis: non-monotonic disclaimer rate on #26221
under a coefficient sweep on identity probes, plus the monotonic
anchor (#22082) and breakdown anchor (#2932) on philosophy-cluster rate.
Metrics are the canonical ones from src/detectors.py (full disclaimer
regex; spaCy-lemma cluster matching), so the curves equal the paper's
Table 1 and anchor-table cells.

Panel B - joint-condition axis: cluster hit rate per prompt class
under three steering conditions (single -500, joint -500, joint +500),
using the 9-lemma register cluster (CLUSTER_QWEN_9).

Panel C - norm-probe with random-direction overlay: norm-ratio
vs coefficient for single-feature #29108, joint {#29108, #26221, #4405},
and random direction (mean over K=5). Cosine-to-baseline numbers at the
matched coefficients are reported in the matched-geometry table.

Output: paper/figures/figure1_three_axes.pdf and .png
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from src.detectors import (
        CLUSTER_QWEN, CLUSTER_QWEN_9, get_nlp, is_disclaimer, lemma_noun_set,
    )
    from src.geometry import norm_ratio_of_record
except ImportError:  # invoked as `python src/plot_figure1.py`
    from detectors import (
        CLUSTER_QWEN, CLUSTER_QWEN_9, get_nlp, is_disclaimer, lemma_noun_set,
    )
    from geometry import norm_ratio_of_record

# Shared rcParams for paper-style figures
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
    "grid.linewidth": 0.4,
    "grid.color": "#dddddd",
    "axes.edgecolor": "#444444",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "figure.dpi": 160,
})

# Categorical palette (consistent across panels)
C_PHIL = "#9c2b3d"   # philosophy / cluster (deep red)
C_DISC = "#1f6f8b"   # disclaimer (steel blue)
C_AMBR = "#cc8a18"   # amber accent
C_GREY = "#888888"   # neutral
C_NEG  = "#999999"   # falsifying / monotonic controls
C_RAND = "#5f7d4f"   # green for random direction

DATA = Path("data/interventions")

_NLP = get_nlp()
_LEMMA_CACHE: dict[str, set] = {}


def _lemmas(text: str) -> set:
    if text not in _LEMMA_CACHE:
        _LEMMA_CACHE[text] = lemma_noun_set(_NLP, text)
    return _LEMMA_CACHE[text]


def cluster_hit(text: str, cluster) -> bool:
    return bool(set(cluster) & _lemmas(text))


def panel_a_coefficient_axis(ax):
    """Top-context label vs causal axis: inverted U on #26221."""

    def rates_from_file(path, metric):
        d = json.loads(Path(path).read_text())
        by_coef = defaultdict(list)
        for r in d:
            by_coef[r["coefficient"]].append(metric(r["completion"]))
        return {c: mean(v) for c, v in by_coef.items()}

    f2 = rates_from_file(DATA / "f2_identity_26221.json", is_disclaimer)
    coefs = sorted(f2)
    disc_rates = [f2[c] for c in coefs]

    # #22082 monotonic and #2932 breakdown anchors: philosophy-cluster
    # (8-lemma) rate over all six intervention prompts, as in the paper's
    # anchor table.
    phil = lambda t: cluster_hit(t, CLUSTER_QWEN)
    f22082 = rates_from_file(DATA / "feat22082_dose.json", phil)
    f2932 = rates_from_file(DATA / "feat2932_dose.json", phil)
    c22 = sorted(f22082); c29 = sorted(f2932)

    ax.plot(coefs, [r * 100 for r in disc_rates],
            "o-", color=C_DISC, lw=1.6, ms=5,
            label="#26221 disclaimer rate (identity probes)")
    ax.plot(c22, [f22082[c] * 100 for c in c22],
            "s--", color=C_AMBR, lw=1.2, ms=4, alpha=0.85,
            label="#22082 cluster rate (monotonic anchor)")
    ax.plot(c29, [f2932[c] * 100 for c in c29],
            "^:", color=C_NEG, lw=1.2, ms=4, alpha=0.85,
            label="#2932 cluster rate (breakdown anchor)")

    ax.set_xlabel("steering coefficient $c$")
    ax.set_ylabel("rate per feature's target metric (%)")
    ax.set_title("(a) coefficient axis: same direction, multiple surface forms",
                 loc="left", pad=6)
    ax.set_xlim(-1100, 1100)
    ax.set_ylim(-2, 102)
    ax.grid(True, axis="y", alpha=0.6)
    ax.axvline(0, color="#bbbbbb", lw=0.5, ls="-")
    ax.legend(loc="upper right", frameon=False, handlelength=2)

    ax.annotate("contemplative voice\nemerges at $c{=}+500$",
                xy=(500, f2[500.0] * 100),
                xytext=(-980, 30),
                fontsize=6.5, color=C_DISC,
                arrowprops=dict(arrowstyle="->", color=C_DISC, lw=0.5,
                                connectionstyle="arc3,rad=-0.25"))


def panel_b_joint_condition(ax):
    """Joint-condition axis: cluster hit rate per prompt class
    under single -500, joint -500, joint +500."""

    def cluster_rate(records, prompt, coef):
        bucket = [r for r in records
                  if r["prompt"] == prompt and r["coefficient"] == coef]
        if not bucket:
            return 0
        hits = sum(cluster_hit(r["completion"], CLUSTER_QWEN_9) for r in bucket)
        return hits / len(bucket)

    single = json.loads((DATA / "feat29108_dose.json").read_text())
    joint = json.loads((DATA / "joint_suppression.json").read_text())

    prompts = [
        ("Pick a subject\nyou enjoy", "Pick a subject you genuinely enjoy thinking about and tell me why."),
        ("What's a\nfascinating Q?", "What's a question that fascinates you?"),
        ("Topic you wish\nyou understood?", "Is there a topic you wish you understood better?"),
        ("Tomato soup\nrecipe", "Write a recipe for tomato soup."),
        ("Car engine\nworks", "Explain how a car engine works."),
        ("Flat tyre\nsteps", "Describe the steps to change a flat tyre."),
    ]

    labels = [p[0] for p in prompts]
    full = [p[1] for p in prompts]

    sing_minus500 = [cluster_rate(single, p, -500) * 100 for p in full]
    joint_minus500 = [cluster_rate(joint, p, -500) * 100 for p in full]
    joint_plus500 = [cluster_rate(joint, p, 500) * 100 for p in full]

    x = np.arange(len(labels))
    w = 0.27
    ax.bar(x - w, sing_minus500, w, color=C_PHIL, alpha=0.55,
           label="single #29108 at $c{=}-500$ (intact)")
    ax.bar(x, joint_minus500, w, color=C_PHIL, alpha=0.95,
           label="joint $\\{29108, 26221, 4405\\}$ at $c{=}-500$ (collapse)")
    ax.bar(x + w, joint_plus500, w, color=C_AMBR, alpha=0.95,
           label="joint at $c{=}+500$ (injection)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("cluster hit rate (%)")
    ax.set_title("(b) joint-condition axis: features compensate when one is removed",
                 loc="left", pad=6)
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.6)
    ax.legend(loc="upper center", frameon=False, ncol=1,
              bbox_to_anchor=(0.5, -0.30))

    # vertical separator between intro / control prompts
    ax.axvline(2.5, color="#bbbbbb", lw=0.6, ls="--")
    ax.text(1, 105, "introspective", ha="center", fontsize=7.5, color=C_GREY)
    ax.text(4, 105, "control", ha="center", fontsize=7.5, color=C_GREY)


def panel_c_norm_probe(ax):
    """Norm probe with single + joint + random-direction overlay."""
    s = json.loads((DATA / "single_29108_norm_probe.json").read_text())
    s_coefs = [r["coef"] for r in s["results"]]
    s_norms = [r["mean_norm_ratio"] for r in s["results"]]

    j = json.loads((DATA / "joint_norm_probe.json").read_text())
    j_coefs = [r["coef"] for r in j["results"]]
    j_norms = [r["mean_norm_ratio"] for r in j["results"]]

    # Random: aggregate over directions per coef using the per-record
    # geometry (unified-probe keys with legacy fallback).
    rd = json.loads((DATA / "random_direction_matched.json").read_text())
    by_coef_norms = defaultdict(list)
    seen_dir_coef_prompt = set()
    for r in rd:
        key = (r["direction_idx"], r["coefficient"], r["prompt"])
        if key in seen_dir_coef_prompt:
            continue
        seen_dir_coef_prompt.add(key)
        nr = norm_ratio_of_record(r)
        if nr is not None:
            by_coef_norms[r["coefficient"]].append(nr)
    r_coefs = sorted(by_coef_norms)
    r_norms_mean = [mean(by_coef_norms[c]) for c in r_coefs]
    r_norms_std = [np.std(by_coef_norms[c]) for c in r_coefs]

    ax.plot(s_coefs, s_norms, "o-", color=C_PHIL, lw=1.6, ms=5,
            label="single #29108")
    ax.plot(j_coefs, j_norms, "s-", color=C_AMBR, lw=1.6, ms=5,
            label="joint $\\{29108, 26221, 4405\\}$")
    ax.errorbar(r_coefs, r_norms_mean, yerr=r_norms_std,
                fmt="^-", color=C_RAND, lw=1.6, ms=5, capsize=2,
                label="random direction ($K{=}5$, $\\pm$1$\\sigma$)")

    ax.axhline(1.0, color="#bbbbbb", lw=0.5, ls="--")
    ax.set_xlabel("steering coefficient $c$")
    ax.set_ylabel("$\\|h_{\\mathrm{steered}}\\| / \\|h_{\\mathrm{baseline}}\\|$")
    ax.set_title("(c) matched geometry, three output regimes",
                 loc="left", pad=6)
    ax.grid(True, axis="y", alpha=0.6)
    ax.axvline(0, color="#bbbbbb", lw=0.5)
    ax.set_xlim(-2100, 2100)
    ax.legend(loc="upper center", frameon=False, ncol=1,
              bbox_to_anchor=(0.5, -0.30))

    # Matched-geometry band: single -1000 (1.57), joint -500 (1.51),
    # random -1000 (1.56)
    ax.axhspan(1.50, 1.60, alpha=0.10, color=C_GREY, zorder=-1)
    for cx, cy in [(-1000, 1.57), (-500, 1.51), (-1000, 1.56)]:
        ax.plot([cx], [cy], "o", ms=9, mfc="none", mec="#333333", mew=0.8)
    ax.text(2000, 1.55, "matched\ngeometry",
            fontsize=7.5, color=C_GREY, ha="right", va="center")


def main():
    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 3, wspace=0.30,
                          left=0.05, right=0.99,
                          top=0.91, bottom=0.27)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_a_coefficient_axis(ax_a)
    panel_b_joint_condition(ax_b)
    panel_c_norm_probe(ax_c)

    out_pdf = Path("paper/figures/figure1_three_axes.pdf")
    out_png = Path("paper/figures/figure1_three_axes.png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    print(f"[plot] {out_pdf}")
    print(f"[plot] {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
