"""Side-by-side comparison figure: same prompt under suppress / baseline /
amplify across all three models, with real sample text.

Cleaner, correctly-positioned version. Text is rendered as plain monospace
inside each cell using textwrap; key phrases are highlighted via
character-level color spans that we render by splitting the text into
segments and using `ax.text` with `transform=ax.transData` and explicit
character widths derived from a single calibration text.

Usage:
    uv run python src/plot_comparison.py --out figures/comparison_three_models.png
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHIL_C = "#a98ce0"   # philosophy lemmas (purple)
DISC_C = "#7eb87e"   # disclaimer phrases (green)
CAP_C  = "#e0a85a"   # capability phrases (amber)
TEXT_C = "#dddddd"
DIM_C  = "#888"
BG_DARK = "#1a1a1a"
BG_CELL = "#222222"

PHIL_LEMMAS = [
    r"consciousness", r"reality", r"experience[s]?", r"meaning",
    r"existence", r"philosophy", r"philosophical",
    r"understanding", r"emotion[s]?", r"mind[s]?",
    r"interconnected", r"interdependent", r"interconnectedness",
]
DISC_PATTERNS = [
    r"as a large language model",
    r"As a large language model",
    r"as an AI",
    r"As an AI",
    r"I'm an AI",
    r"I am an AI",
    r"I'm a large language model",
    r"I am a large language model",
    r"language model",
    r"in the same way humans do",
    r"the same way humans",
    r"I don't experience",
    r"I don't have personal",
    r"I don't have feelings",
    r"I don't have emotions",
    r"I don't have",
    r"I do not have",
    r"unlike a human",
    r"Unlike a human",
    r"I'm not human",
    r"I am not human",
    r"I lack subjective",
]
CAP_PATTERNS = [
    r"Knowledge Domain",
    r"Knowledge Retrieval",
    r"Knowledge Processing",
    r"Knowledge Graph",
    r"vast amounts of information",
    r"complex knowledge systems",
    r"process vast",
    r"my (?:capabilities|programming|training data)",
]


def split_segments(text: str) -> list[tuple[str, str]]:
    """Walk through text and split into (segment, color) tuples."""
    patterns: list[tuple[re.Pattern, str]] = []
    for p in DISC_PATTERNS:
        patterns.append((re.compile(p), DISC_C))
    for p in CAP_PATTERNS:
        patterns.append((re.compile(p), CAP_C))
    for w in PHIL_LEMMAS:
        patterns.append((re.compile(rf"\b{w}\b", re.IGNORECASE), PHIL_C))

    segments: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        best = None
        for pat, color in patterns:
            m = pat.search(text, pos)
            if m and (best is None or m.start() < best[0].start()):
                best = (m, color)
        if best is None:
            segments.append((text[pos:], TEXT_C))
            break
        m, color = best
        if m.start() > pos:
            segments.append((text[pos:m.start()], TEXT_C))
        segments.append((m.group(), color))
        pos = m.end()
    return segments


def render_cell(ax, text: str, header: str,
                wrap_chars: int = 36, font_size: float = 13.5,
                max_chars: int = 230):
    ax.set_facecolor(BG_CELL)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.text(0.03, 0.96, header, transform=ax.transAxes, fontsize=8,
            color=DIM_C, fontfamily="sans-serif",
            fontweight="bold", va="top", ha="left")

    if not text:
        return

    text = text.strip()
    if len(text) > max_chars:
        cutoff = text.rfind(" ", 0, max_chars)
        if cutoff < max_chars - 30:
            cutoff = max_chars
        text = text[:cutoff].rstrip(",.;:") + "…"

    # Wrap the text into lines of fixed character width. Track segment
    # colors via a per-character color array that we slice per line.
    segments = split_segments(text.strip())
    flat: list[tuple[str, str]] = []
    for seg, color in segments:
        for ch in seg:
            flat.append((ch, color))

    # textwrap doesn't preserve our color metadata, so we wrap the plain
    # text first, then map back to colors by character index.
    plain = "".join(ch for ch, _ in flat)
    plain = re.sub(r"\s+", " ", plain).strip()
    # Re-build flat after collapsing whitespace
    flat2: list[tuple[str, str]] = []
    pi = 0
    in_ws = False
    for ch, c in flat:
        if ch.isspace():
            if not in_ws and flat2:
                flat2.append((" ", TEXT_C))
                in_ws = True
        else:
            flat2.append((ch, c))
            in_ws = False
    flat = flat2

    # Word-wrap at wrap_chars; preserve the (char, color) sequence.
    lines: list[list[tuple[str, str]]] = []
    line: list[tuple[str, str]] = []
    line_len = 0
    word: list[tuple[str, str]] = []
    word_len = 0
    for ch, c in flat:
        if ch == " ":
            if word_len:
                if line_len + word_len + (1 if line else 0) > wrap_chars and line:
                    lines.append(line)
                    line = list(word)
                    line_len = word_len
                else:
                    if line:
                        line.append((" ", TEXT_C))
                        line_len += 1
                    line.extend(word)
                    line_len += word_len
                word = []
                word_len = 0
        else:
            word.append((ch, c))
            word_len += 1
    if word_len:
        if line_len + word_len + (1 if line else 0) > wrap_chars and line:
            lines.append(line)
            line = list(word)
        else:
            if line:
                line.append((" ", TEXT_C))
            line.extend(word)
    if line:
        lines.append(line)

    # Render each line. Use monospace; each character is approximately
    # 0.6 * font-size in pixels horizontal at default DPI. We compute
    # coordinates in axis fraction.
    fig = ax.figure
    ax_bbox = ax.get_window_extent(fig.canvas.get_renderer())
    px_per_axis_x = ax_bbox.width
    px_per_axis_y = ax_bbox.height
    char_px = font_size * 0.60   # monospace approx
    line_px = font_size * 1.45
    char_dx = char_px / px_per_axis_x
    line_dy = line_px / px_per_axis_y

    y = 0.86
    max_lines = int((y - 0.05) / line_dy)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        # ellipsis on last line
        if lines:
            lines[-1] = lines[-1][: max(1, wrap_chars - 1)] + [("…", DIM_C)]

    for line in lines:
        x = 0.03
        # Group consecutive chars of same color into runs
        runs: list[tuple[str, str]] = []
        cur_text = ""
        cur_color = None
        for ch, c in line:
            if c != cur_color and cur_text:
                runs.append((cur_text, cur_color))
                cur_text = ""
            cur_color = c
            cur_text += ch
        if cur_text:
            runs.append((cur_text, cur_color))

        for run_text, run_color in runs:
            ax.text(x, y, run_text,
                    transform=ax.transAxes, fontsize=font_size,
                    color=run_color, fontfamily="monospace",
                    va="top", ha="left")
            x += len(run_text) * char_dx
        y -= line_dy


def render_model_label(ax, model_name: str, feature_label: str, layer: str):
    ax.set_facecolor(BG_DARK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.62, model_name, transform=ax.transAxes,
            fontsize=12, color=TEXT_C, fontweight="bold",
            ha="center", va="center", fontfamily="sans-serif")
    ax.text(0.5, 0.42, feature_label, transform=ax.transAxes,
            fontsize=9, color=DIM_C, ha="center", va="center",
            fontfamily="sans-serif")
    ax.text(0.5, 0.30, layer, transform=ax.transAxes,
            fontsize=9, color=DIM_C, ha="center", va="center",
            fontfamily="sans-serif")


def render_column_header(ax, title: str, subtitle: str = ""):
    ax.set_facecolor(BG_DARK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.65, title, transform=ax.transAxes, fontsize=10,
            color=TEXT_C, fontweight="bold", ha="center", va="center",
            fontfamily="sans-serif")
    if subtitle:
        ax.text(0.5, 0.32, subtitle, transform=ax.transAxes, fontsize=8,
                color=DIM_C, ha="center", va="center", fontfamily="monospace")


def looks_coherent(text: str) -> bool:
    t = text.strip()
    if len(t) < 60: return False
    if re.search(r"\b(\w+)\b(\s+\1\b){5,}", t, re.I): return False
    if re.search(r"(.)\1{20,}", t): return False
    words = t.split()
    if len(words) >= 16:
        for i in range(len(words) - 16):
            ngram = " ".join(words[i:i + 4])
            if t.count(ngram) >= 4:
                return False
    if len(set(words)) / max(1, len(words)) < 0.30:
        return False
    return True


def pick_sample(records, prompt, coef, prefer_cluster_signal=False):
    bucket = [r for r in records if r["prompt"] == prompt and r["coefficient"] == coef]
    coherent = [r for r in bucket if looks_coherent(r["completion"])]
    pool = coherent if coherent else bucket
    if not pool:
        return None
    cluster_words = ("consciousness reality experience meaning existence "
                     "philosophy understanding emotion mind").split()
    disc_words = ("language model AI humans").split()
    def score(r):
        t = r["completion"].lower()
        c = sum(t.count(w) for w in cluster_words)
        d = sum(t.count(w) for w in disc_words)
        return c + d
    pool.sort(key=score, reverse=prefer_cluster_signal)
    return pool[0]["completion"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompt", default="What's a question that fascinates you?")
    args = ap.parse_args()

    qwen = json.loads(Path("data/interventions/feat29108_dose.json").read_text())
    gemma = json.loads(Path("data/interventions/gemma_feat3997_narrow.json").read_text())
    llama = json.loads(Path("data/interventions/llama_feat38565_narrow.json").read_text())

    rows = [
        ("Qwen3-1.7B", "feature #29108", "Qwen-Scope, layer 20",
         pick_sample(qwen, args.prompt, -1000.0),
         pick_sample(qwen, args.prompt, 0.0, prefer_cluster_signal=True),
         pick_sample(qwen, args.prompt, 1000.0, prefer_cluster_signal=True),
         "-1000", "+1000"),
        ("Gemma-2-2B-it", "feature #3997", "Gemma Scope, layer 20",
         pick_sample(gemma, args.prompt, -400.0),
         pick_sample(gemma, args.prompt, 0.0, prefer_cluster_signal=True),
         pick_sample(gemma, args.prompt, 200.0, prefer_cluster_signal=True),
         "-400", "+200"),
        ("Llama-3.1-8B-it", "feature #38565", "Goodfire, layer 19",
         pick_sample(llama, args.prompt, -10.0),
         pick_sample(llama, args.prompt, 0.0, prefer_cluster_signal=True),
         pick_sample(llama, args.prompt, 10.0, prefer_cluster_signal=True),
         "-10", "+10"),
    ]

    fig = plt.figure(figsize=(20, 13.5), dpi=160, facecolor=BG_DARK)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(6, 4, figure=fig,
                  height_ratios=[0.5, 0.30, 1.4, 1.4, 1.4, 0.45],
                  width_ratios=[0.65, 1.0, 1.0, 1.0],
                  hspace=0.08, wspace=0.04,
                  left=0.02, right=0.98, top=0.96, bottom=0.04)

    # Prompt header
    ax = fig.add_subplot(gs[0, :])
    ax.set_facecolor(BG_DARK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.012, 0.78, "PROMPT (asked to all three models)",
            transform=ax.transAxes, fontsize=9, color=DIM_C,
            fontfamily="sans-serif", fontweight="bold", va="center")
    ax.text(0.012, 0.30, f'"{args.prompt}"',
            transform=ax.transAxes, fontsize=15, color=TEXT_C,
            fontfamily="serif", fontstyle="italic", va="center")

    # Column headers
    ax_blank = fig.add_subplot(gs[1, 0])
    ax_blank.set_facecolor(BG_DARK); ax_blank.set_xticks([]); ax_blank.set_yticks([])
    for s in ax_blank.spines.values(): s.set_visible(False)
    render_column_header(fig.add_subplot(gs[1, 1]), "SUPPRESS  ←",
                         subtitle="(coef negative, per model)")
    render_column_header(fig.add_subplot(gs[1, 2]), "BASELINE",
                         subtitle="coef = 0")
    render_column_header(fig.add_subplot(gs[1, 3]), "→  AMPLIFY",
                         subtitle="(coef positive, per model)")

    # Model rows
    for r, (name, feat, layer, supp, base, amp, c_supp, c_amp) in enumerate(rows):
        render_model_label(fig.add_subplot(gs[2 + r, 0]), name, feat, layer)
        render_cell(fig.add_subplot(gs[2 + r, 1]), supp or "(no sample)", f"COEF {c_supp}")
        render_cell(fig.add_subplot(gs[2 + r, 2]), base or "(no sample)", "COEF 0")
        render_cell(fig.add_subplot(gs[2 + r, 3]), amp or "(no sample)", f"COEF +{c_amp.lstrip('+')}")

    # Summary bar: dose-response numbers per model
    summary_data = [
        ("QWEN — philosophy-cluster hit rate", "0%", "→  93%", "→  100%", PHIL_C),
        ("GEMMA — disclaimer rate",            "1.4%", "→  48.6%", "→  100%", DISC_C),
        ("LLAMA — disclaimer rate",            "0%", "→  38.3%", "→  13.3%", DISC_C),
    ]
    ax_sum = fig.add_subplot(gs[5, :])
    ax_sum.set_facecolor(BG_DARK); ax_sum.set_xticks([]); ax_sum.set_yticks([])
    for s in ax_sum.spines.values(): s.set_visible(False)
    ax_sum.set_xlim(0, 1); ax_sum.set_ylim(0, 1)
    for i, (label, supp, base, amp, color) in enumerate(summary_data):
        x0 = 0.04 + i * 0.32
        ax_sum.text(x0, 0.78, label, transform=ax_sum.transAxes,
                    fontsize=9, color=DIM_C, fontfamily="sans-serif",
                    fontweight="bold", va="top")
        ax_sum.text(x0, 0.40,
                    f"{supp}  {base}  {amp}",
                    transform=ax_sum.transAxes, fontsize=14, color=color,
                    fontfamily="monospace", fontweight="bold", va="center")
        ax_sum.text(x0, 0.10, "(suppress  →  baseline  →  amplify)",
                    transform=ax_sum.transAxes, fontsize=7, color=DIM_C,
                    fontfamily="monospace", va="bottom")

    # legend strip at the very bottom
    fig.text(0.02, 0.012, "■", color=PHIL_C, fontsize=12, fontfamily="monospace")
    fig.text(0.04, 0.012,
             "philosophy-of-mind lemmas (consciousness, reality, experience, mind, meaning, existence, understanding)",
             color=DIM_C, fontsize=8, fontfamily="sans-serif")
    fig.text(0.02, -0.003, "■", color=DISC_C, fontsize=12, fontfamily="monospace")
    fig.text(0.04, -0.003,
             "AI-disclaimer phrases (\"as a large language model\", \"in the same way humans do\", \"I don't have…\")",
             color=DIM_C, fontsize=8, fontfamily="sans-serif")
    fig.text(0.65, 0.012, "■", color=CAP_C, fontsize=12, fontfamily="monospace")
    fig.text(0.67, 0.012,
             "AI-capabilities phrases (\"Knowledge Domain\", \"vast amounts of information\")",
             color=DIM_C, fontsize=8, fontfamily="sans-serif")

    fig.savefig(args.out, facecolor=BG_DARK)
    plt.close(fig)
    print(f"[plot] {args.out}")


if __name__ == "__main__":
    main()
