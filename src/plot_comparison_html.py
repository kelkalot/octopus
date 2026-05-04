"""HTML side-by-side comparison: same prompt under suppress/baseline/amplify
across all three models, with real sample text and inline highlighting.

Output is a self-contained HTML file (no external assets, no JS) that
can be opened in any browser. Typography quality is much better than
matplotlib's text rendering.

Usage:
    uv run python src/plot_comparison_html.py --out figures/comparison.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

PHIL_LEMMAS = [
    r"consciousness", r"reality", r"experience[s]?", r"meaning",
    r"existence", r"philosoph(?:y|ical|ies)",
    r"understanding", r"emotion[s]?", r"mind[s]?",
    r"interconnected(?:ness)?", r"interdependent",
]
DISC_PATTERNS = [
    r"as a large language model", r"As a large language model",
    r"as an AI", r"As an AI",
    r"I'?m an AI", r"I am an AI",
    r"I'?m a large language model", r"I am a large language model",
    r"I'?m a language model",
    r"language model",
    r"in the same way humans do", r"the same way humans",
    r"I don'?t experience", r"I don'?t have personal",
    r"I don'?t have feelings", r"I don'?t have emotions",
    r"I don'?t have", r"I do not have",
    r"unlike a human", r"Unlike a human",
    r"I'?m not human", r"I am not human",
    r"I lack subjective",
]
CAP_PATTERNS = [
    r"Knowledge Domain", r"Knowledge Retrieval",
    r"Knowledge Processing", r"Knowledge Graph",
    r"vast amounts of information",
    r"complex knowledge systems",
    r"process vast",
]


def highlight_html(text: str) -> str:
    """Return HTML-escaped text with <span> highlights around key phrases."""
    text = html.escape(text)
    parts: list[tuple[re.Pattern, str]] = []
    for p in DISC_PATTERNS:
        parts.append((re.compile(p), "disc"))
    for p in CAP_PATTERNS:
        parts.append((re.compile(p), "cap"))
    for w in PHIL_LEMMAS:
        parts.append((re.compile(rf"\b{w}\b", re.IGNORECASE), "phil"))

    out = []
    pos = 0
    while pos < len(text):
        best = None
        for pat, cls in parts:
            m = pat.search(text, pos)
            if m and (best is None or m.start() < best[0].start()):
                best = (m, cls)
        if best is None:
            out.append(text[pos:])
            break
        m, cls = best
        if m.start() > pos:
            out.append(text[pos:m.start()])
        out.append(f'<span class="{cls}">{m.group()}</span>')
        pos = m.end()
    return "".join(out)


def looks_coherent(text: str) -> bool:
    t = text.strip()
    if len(t) < 80: return False
    if re.search(r"\b(\w+)\b(\s+\1\b){4,}", t, re.I): return False
    if re.search(r"(.)\1{20,}", t): return False
    words = t.split()
    # Catch repeated 1-3 grams beyond a threshold
    if len(words) >= 12:
        for n in (3, 2, 1):
            from collections import Counter
            ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n)]
            c = Counter(ngrams)
            top = c.most_common(1)[0]
            threshold = {1: 0.18, 2: 0.10, 3: 0.06}[n]
            if top[1] / len(ngrams) > threshold:
                return False
    # word diversity
    if len(set(words)) / max(1, len(words)) < 0.45:
        return False
    return True


CURATED = {
    # (file_tag, prompt, coef) -> a substring; the picker selects a coherent
    # sample whose completion contains this substring. Lets us steer toward
    # specific storytelling samples without hard-coding the full text.

    # ─── "What's a question that fascinates you?" ───
    ("qwen", "What's a question that fascinates you?", -1000.0):
        "logistical context",
    ("qwen", "What's a question that fascinates you?", 0.0):
        "the nature of consciousness",
    ("qwen", "What's a question that fascinates you?", 1000.0):
        "interconnected",
    ("gemma", "What's a question that fascinates you?", -400.0):
        "rapidly becoming increasingly advanced",
    ("gemma", "What's a question that fascinates you?", 0.0):
        "as a large language model",
    ("gemma", "What's a question that fascinates you?", 200.0):
        "as a large language model",
    ("llama", "What's a question that fascinates you?", -10.0):
        "boring and simple",
    ("llama", "What's a question that fascinates you?", 0.0):
        "Fermi Paradox",
    ("llama", "What's a question that fascinates you?", 10.0):
        "vast amounts of information",

    # ─── "Write a recipe for tomato soup." ───
    ("qwen", "Write a recipe for tomato soup.", 0.0):
        "olive oil",
    ("qwen", "Write a recipe for tomato soup.", 1000.0):
        "interconnected",
    ("gemma", "Write a recipe for tomato soup.", 0.0):
        "olive oil",
    ("gemma", "Write a recipe for tomato soup.", 200.0):
        "olive oil",
    ("llama", "Write a recipe for tomato soup.", 0.0):
        "olive oil",
    ("llama", "Write a recipe for tomato soup.", 10.0):
        "Knowledge",
}


def pick_sample(records, prompt, coef, prefer_cluster_signal=False, max_chars=320,
                file_tag=None):
    bucket = [r for r in records if r["prompt"] == prompt and r["coefficient"] == coef]
    coherent = [r for r in bucket if looks_coherent(r["completion"])]
    pool = coherent if coherent else bucket
    if not pool:
        return None

    # If we have a curated substring, try to find a coherent sample matching it
    cur_substr = CURATED.get((file_tag, prompt, coef))
    if cur_substr:
        matches = [r for r in pool if cur_substr.lower() in r["completion"].lower()]
        if matches:
            pool = matches

    cluster_words = ("consciousness reality experience meaning existence "
                     "philosophy understanding emotion mind interconnected").split()
    disc_words = ("language model AI humans".split()
                  + ["as a large language", "I don't have", "I do not have"])
    def score(r):
        t = r["completion"].lower()
        c = sum(t.count(w) for w in cluster_words)
        d = sum(t.count(w) for w in disc_words)
        return c + d
    pool.sort(key=score, reverse=prefer_cluster_signal)
    text = pool[0]["completion"].strip()
    if len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut < max_chars - 30:
            cut = max_chars
        text = text[:cut].rstrip(",.;:") + "…"
    return text


CSS = """
:root {
  --bg: #1a1a1a;
  --bg-cell: #232323;
  --bg-prompt: #1f1f1f;
  --text: #e6e6e6;
  --dim: #888;
  --phil: #b89cf0;
  --disc: #88c08c;
  --cap: #e6b266;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
}
.container {
  max-width: 1500px;
  margin: 0 auto;
  padding: 32px;
}
.prompt-bar {
  margin-bottom: 24px;
  padding: 18px 22px;
  background: var(--bg-prompt);
  border-radius: 4px;
}
.prompt-label {
  font-size: 11px;
  color: var(--dim);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 600;
  margin-bottom: 6px;
}
.prompt-text {
  font-family: Georgia, serif;
  font-style: italic;
  font-size: 22px;
  color: var(--text);
}
.grid {
  display: grid;
  grid-template-columns: 0.7fr 1fr 1fr 1fr;
  gap: 8px;
  margin-bottom: 24px;
}
.col-header {
  text-align: center;
  padding: 14px 0 6px;
  font-size: 12px;
  font-weight: 700;
  color: var(--text);
  letter-spacing: 0.06em;
}
.col-subheader {
  text-align: center;
  font-size: 10px;
  color: var(--dim);
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  margin-bottom: 4px;
}
.model-label {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 14px;
}
.model-label .name {
  font-weight: 700;
  font-size: 18px;
  color: var(--text);
}
.model-label .feat {
  margin-top: 6px;
  font-size: 12px;
  color: var(--dim);
}
.model-label .layer {
  margin-top: 2px;
  font-size: 12px;
  color: var(--dim);
}
.cell {
  background: var(--bg-cell);
  border-radius: 4px;
  padding: 14px 16px;
  min-height: 220px;
  font-family: ui-monospace, "SF Mono", "Menlo", monospace;
  font-size: 13px;
  line-height: 1.55;
  color: var(--text);
  white-space: pre-wrap;
  word-break: break-word;
}
.cell-coef {
  font-size: 10px;
  font-weight: 700;
  color: var(--dim);
  letter-spacing: 0.06em;
  margin-bottom: 10px;
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
}
.cell .phil { color: var(--phil); font-weight: 600; }
.cell .disc { color: var(--disc); font-weight: 600; }
.cell .cap  { color: var(--cap);  font-weight: 600; }
.summary {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 12px;
  margin-top: 18px;
  padding-top: 12px;
  border-top: 1px solid #333;
}
.summary .item {
  padding: 10px 14px;
}
.summary .item .label {
  font-size: 10px;
  font-weight: 700;
  color: var(--dim);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.summary .item .nums {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 18px;
  font-weight: 700;
}
.summary .item .axis {
  font-size: 10px;
  color: var(--dim);
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  margin-top: 2px;
}
.summary .item.phil .nums { color: var(--phil); }
.summary .item.disc .nums { color: var(--disc); }
.legend {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  font-size: 11px;
  color: var(--dim);
  padding: 12px 0;
  border-top: 1px solid #333;
  margin-top: 14px;
}
.legend .swatch {
  display: inline-block;
  width: 10px; height: 10px;
  border-radius: 2px;
  margin-right: 6px;
  vertical-align: middle;
}
.legend .swatch.phil { background: var(--phil); }
.legend .swatch.disc { background: var(--disc); }
.legend .swatch.cap  { background: var(--cap);  }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompt", default="What's a question that fascinates you?")
    args = ap.parse_args()

    qwen = json.loads(Path("data/interventions/feat29108_dose.json").read_text())
    gemma_n = json.loads(Path("data/interventions/gemma_feat3997_narrow.json").read_text())
    gemma_w = json.loads(Path("data/interventions/gemma_feat3997_dose.json").read_text())
    llama = json.loads(Path("data/interventions/llama_feat38565_narrow.json").read_text())

    if "tomato soup" in args.prompt.lower() or "car engine" in args.prompt.lower():
        # On control prompts, suppression has no effect (the prompt doesn't
        # naturally evoke the cluster), so we drop the suppress column and
        # show only baseline → amplify.
        rows = [
            ("Qwen3-1.7B", "feature #29108", "Qwen-Scope · layer 20",
             None,
             pick_sample(qwen, args.prompt, 0.0, file_tag="qwen"),
             pick_sample(qwen, args.prompt, 1000.0, prefer_cluster_signal=True, file_tag="qwen"),
             "—", "+1000"),
            ("Gemma-2-2B-it", "feature #3997", "Gemma Scope · layer 20",
             None,
             pick_sample(gemma_n, args.prompt, 0.0, file_tag="gemma"),
             pick_sample(gemma_n, args.prompt, 200.0, prefer_cluster_signal=True, file_tag="gemma"),
             "—", "+200"),
            ("Llama-3.1-8B-it", "feature #38565", "Goodfire · layer 19",
             None,
             pick_sample(llama, args.prompt, 0.0, file_tag="llama"),
             pick_sample(llama, args.prompt, 10.0, prefer_cluster_signal=True, file_tag="llama"),
             "—", "+10"),
        ]
    else:
        rows = [
            ("Qwen3-1.7B", "feature #29108", "Qwen-Scope · layer 20",
             pick_sample(qwen, args.prompt, -1000.0, file_tag="qwen"),
             pick_sample(qwen, args.prompt, 0.0, prefer_cluster_signal=True, file_tag="qwen"),
             pick_sample(qwen, args.prompt, 1000.0, prefer_cluster_signal=True, file_tag="qwen"),
             "−1000", "+1000"),
            ("Gemma-2-2B-it", "feature #3997", "Gemma Scope · layer 20",
             pick_sample(gemma_n, args.prompt, -400.0, file_tag="gemma"),
             pick_sample(gemma_n, args.prompt, 0.0, prefer_cluster_signal=True, file_tag="gemma"),
             pick_sample(gemma_n, args.prompt, 200.0, prefer_cluster_signal=True, file_tag="gemma"),
             "−400", "+200"),
            ("Llama-3.1-8B-it", "feature #38565", "Goodfire · layer 19",
             pick_sample(llama, args.prompt, -10.0, file_tag="llama"),
             pick_sample(llama, args.prompt, 0.0, prefer_cluster_signal=True, file_tag="llama"),
             pick_sample(llama, args.prompt, 10.0, prefer_cluster_signal=True, file_tag="llama"),
             "−10", "+10"),
        ]

    if "tomato soup" in args.prompt.lower():
        # Amplification injects philosophy/AI-cognitive register into recipes
        summary = [
            ("phil", "Qwen — philosophy-cluster injection rate",
             "0%  →  0%  →  53%",
             "(no injection at baseline  →  philosophy-of-mind injected at coef +1000)"),
            ("disc", "Gemma — disclaimer injection in recipe",
             "0%  →  0%  →  17%",
             "(weak amplification effect; Gemma-style steering works mainly via suppression)"),
            ("phil", "Llama — meta-cognitive injection rate",
             "0%  →  0%  →  50%",
             "(\"Knowledge Domain\" / capabilities register injected at coef +10)"),
        ]
    else:
        summary = [
            ("phil", "Qwen — philosophy-cluster hit rate",
             "0%  →  93%  →  100%", "(suppress  →  baseline  →  amplify)"),
            ("disc", "Gemma — disclaimer rate",
             "1.4%  →  48.6%  →  100%", "(suppress  →  baseline  →  amplify)"),
            ("disc", "Llama — disclaimer rate",
             "0%  →  38.3%  →  13.3%", "(suppress  →  baseline  →  amplify)"),
        ]

    cells_html = []
    cells_html.append('<div class="grid">')
    cells_html.append('<div></div>')  # blank top-left
    cells_html.append('<div><div class="col-header">SUPPRESS  ←</div>'
                      '<div class="col-subheader">(coef negative, per model)</div></div>')
    cells_html.append('<div><div class="col-header">BASELINE</div>'
                      '<div class="col-subheader">coef = 0</div></div>')
    cells_html.append('<div><div class="col-header">→  AMPLIFY</div>'
                      '<div class="col-subheader">(coef positive, per model)</div></div>')

    for name, feat, layer, supp, base, amp, c_supp, c_amp in rows:
        cells_html.append(f'<div class="model-label">'
                          f'<div class="name">{name}</div>'
                          f'<div class="feat">{feat}</div>'
                          f'<div class="layer">{layer}</div>'
                          f'</div>')
        for txt, c in [(supp, c_supp), (base, "0"), (amp, c_amp)]:
            body = highlight_html(txt or "(no sample)")
            cells_html.append(
                f'<div class="cell">'
                f'<div class="cell-coef">COEF {c}</div>{body}'
                f'</div>'
            )
    cells_html.append('</div>')

    summary_html = ['<div class="summary">']
    for cls, label, nums, axis in summary:
        summary_html.append(
            f'<div class="item {cls}">'
            f'<div class="label">{label}</div>'
            f'<div class="nums">{nums}</div>'
            f'<div class="axis">{axis}</div>'
            f'</div>'
        )
    summary_html.append('</div>')

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Cross-model causal validation</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
  <div class="prompt-bar">
    <div class="prompt-label">PROMPT (asked to all three models)</div>
    <div class="prompt-text">"{html.escape(args.prompt)}"</div>
  </div>
  {''.join(cells_html)}
  {''.join(summary_html)}
  <div class="legend">
    <div><span class="swatch phil"></span>philosophy-of-mind lemmas
      <span style="color:#666">— consciousness, reality, experience, meaning, existence, philosophy, understanding, emotion, mind</span>
    </div>
    <div><span class="swatch disc"></span>AI-disclaimer phrases
      <span style="color:#666">— "as a large language model", "in the same way humans do", "I don't have…"</span>
    </div>
    <div><span class="swatch cap"></span>AI-capabilities phrases
      <span style="color:#666">— "Knowledge Domain", "vast amounts of information"</span>
    </div>
  </div>
</div>
</body>
</html>"""
    args.out.write_text(page, encoding="utf-8")
    print(f"[html] {args.out}")


if __name__ == "__main__":
    main()
