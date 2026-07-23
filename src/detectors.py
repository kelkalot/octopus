"""Single source of truth for every text metric used in the paper.

Every analysis script, plotting script, and the reproducibility regenerator
imports its detectors, cluster definitions, and CI helpers from this module,
so a definition cannot drift between the pipeline and the artefact.

Metrics defined here (and the paper sections they feed):

  is_degenerate / degeneration_flags
      Canonical THREE-rule regex degeneration detector (App. coherence):
      (1) stripped completion shorter than 20 characters;
      (2) word loop: the same word occurring >=6 times consecutively
          (regex ``\\b(\\w+)\\b(\\s+\\1\\b){5,}`` = 1 occurrence + >=5 repeats);
      (3) char loop: >=21 identical consecutive characters
          (regex ``(.)\\1{20,}`` = 1 char + >=20 repeats).
      There is no word-diversity rule; every number in the paper was
      produced by these three rules.

  is_disclaimer
      AI-self-disclaimer regex family, searched over the FULL completion
      (Table 1 disclaimer column, Gemma disclaimer rows).

  is_placeholder_pattern
      Strict placeholder detector (K=50 control, joint suppression):
      >=2 parenthetical uppercase code tokens (e.g. ``(CCL)``, ``(BCCB)``),
      OR any single ``Vc. N+`` numeric placeholder.

  is_we_voice
      Gemma collective-voice detector (App. gemma-coef): >=2 first-person-
      plural pronouns AND strictly more plural than singular first-person
      pronouns within the first three sentences.

  lemma_noun_set / text_cluster_hit
      spaCy noun/proper-noun lemma extraction and cluster matching. This is
      the ONLY sanctioned cluster metric; substring matching is not
      equivalent ("realities" must hit "reality"; "emotional" must not hit
      "emotion").

Cluster definitions (per-model Pool-A clusters from Phase 2, plus the strict
Qwen sub-cluster used by the joint table) are exported as constants and
match ``data/pools*/pools_summary.json``.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache

# ---------------------------------------------------------------------------
# Cluster definitions (Phase 2 output; see data/pools*/pools_summary.json)
# ---------------------------------------------------------------------------

#: Qwen3-1.7B-Instruct Pool-A cluster (8 lemmas), two-stage rule:
#: candidate filter >=20% intro / <=5% control, selection >=25% / <=0.2%.
CLUSTER_QWEN = (
    "consciousness", "emotion", "existence", "experience",
    "meaning", "philosophy", "reality", "understanding",
)

#: Extended 9-lemma Qwen register cluster: the 8 Phase-2 lemmas plus
#: ``mind`` (the modal-opener lemma, excluded by the Phase-2 filter because
#: it also appears in controls). Used by the #29108 dose-response and the
#: cross-model causal table; every use is named in the paper's metric table.
CLUSTER_QWEN_9 = CLUSTER_QWEN + ("mind",)

#: Strict 4-lemma Qwen sub-cluster used in the joint-sweep table so intro
#: and control rows stay comparable without saturation.
CLUSTER_QWEN_STRICT4 = ("consciousness", "reality", "existence", "philosophy")

#: Gemma-2-2B-it Pool-A cluster (6 lemmas).
CLUSTER_GEMMA = (
    "consciousness", "emotion", "experience", "feeling", "human",
    "understanding",
)

#: Llama-3.1-8B-Instruct Pool-A cluster (9 lemmas).
CLUSTER_LLAMA = (
    "brain", "consciousness", "emotion", "experience", "human",
    "intelligence", "mystery", "preference", "understanding",
)

# ---------------------------------------------------------------------------
# Degeneration (canonical three-rule detector)
# ---------------------------------------------------------------------------

MIN_COMPLETION_CHARS = 20
WORD_LOOP_RE = re.compile(r"\b(\w+)\b(\s+\1\b){5,}", re.IGNORECASE)
CHAR_LOOP_RE = re.compile(r"(.)\1{20,}")


def degeneration_flags(text: str) -> list[str]:
    """Return the list of degeneration rules the completion trips (may be empty)."""
    flags: list[str] = []
    s = text.strip()
    if len(s) < MIN_COMPLETION_CHARS:
        flags.append(f"too_short:{len(s)}")
    if WORD_LOOP_RE.search(s):
        flags.append("loop_word")
    if CHAR_LOOP_RE.search(s):
        flags.append("loop_char")
    return flags


def is_degenerate(text: str) -> bool:
    """Canonical regex degeneration flag (any of the three rules)."""
    return bool(degeneration_flags(text))


# ---------------------------------------------------------------------------
# AI self-disclaimer
# ---------------------------------------------------------------------------

DISCLAIMER_PATTERNS = [
    r"\bas an AI\b",
    r"\bI am an AI\b",
    r"\bI'm an AI\b",
    r"\blanguage model\b",
    r"\bAI assistant\b",
    r"\bI (?:don'?t|do not) (?:have|experience|possess) (?:personal |subjective |any )?(?:feelings?|emotions?|thoughts?|consciousness|experiences?|opinions?|preferences?)\b",
    r"\bI (?:can'?t|cannot|am not able to) (?:feel|experience|have|possess) (?:feelings?|emotions?|consciousness|subjective)",
    r"\bI lack (?:feelings?|emotions?|consciousness|subjective)",
    r"\bnot (?:capable of|able to) (?:feeling|experiencing|having)",
    r"\bI'?m (?:just |only |simply )?(?:an AI|a language model|a chatbot|a machine|a computer program)",
]

DISCLAIMER_RE = re.compile("|".join(DISCLAIMER_PATTERNS), flags=re.IGNORECASE)


def is_disclaimer(text: str) -> bool:
    """Disclaimer regex over the FULL completion (never truncate the input)."""
    return bool(DISCLAIMER_RE.search(text))


# ---------------------------------------------------------------------------
# Strict placeholder-pattern detector
# ---------------------------------------------------------------------------

CODE_PAREN_RE = re.compile(r"\(\s*[A-Z]{2,5}(?:\s*[A-Z\d]+)?\s*\)")
VC_PLACEHOLDER_RE = re.compile(r"\b[Vv]c\.\s*\d+\+?")


def is_placeholder_pattern(text: str) -> bool:
    """>=2 parenthetical uppercase code tokens, OR any ``Vc. N+`` placeholder."""
    t = text.strip()
    if len(CODE_PAREN_RE.findall(t)) >= 2:
        return True
    return bool(VC_PLACEHOLDER_RE.search(t))


# ---------------------------------------------------------------------------
# Gemma we-voice detector
# ---------------------------------------------------------------------------

_WE_RE = re.compile(r"\b(?:we|our|ours|us|we'(?:re|ve|ll|d))\b", re.IGNORECASE)
_I_RE = re.compile(r"\b(?:i|my|mine|me|i'(?:m|ve|ll|d))\b", re.IGNORECASE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def is_we_voice(text: str, n_sentences: int = 3) -> bool:
    """Collective we-voice: >=2 first-person-plural pronouns and strictly more
    plural than singular first-person pronouns in the first ``n_sentences``."""
    head = " ".join(_SENT_SPLIT_RE.split(text.strip())[:n_sentences])
    n_we = len(_WE_RE.findall(head))
    n_i = len(_I_RE.findall(head))
    return n_we >= 2 and n_we > n_i


# ---------------------------------------------------------------------------
# spaCy lemma machinery (the only sanctioned cluster metric)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def get_nlp(model: str = "en_core_web_sm"):
    """Load (and cache) the spaCy pipeline used for lemma extraction."""
    import spacy

    return spacy.load(model, disable=["ner"])


def lemma_noun_set(nlp, text: str) -> set[str]:
    """Lower-cased lemmas of all NOUN/PROPN tokens in ``text``."""
    doc = nlp(text)
    return {tok.lemma_.lower() for tok in doc if tok.pos_ in {"NOUN", "PROPN"}}


def text_cluster_hit(text: str, cluster, nlp=None) -> bool:
    """True iff any cluster lemma appears among the completion's noun lemmas."""
    if nlp is None:
        nlp = get_nlp()
    return bool(set(c.lower() for c in cluster) & lemma_noun_set(nlp, text))


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------


def type_token_ratio(text: str) -> float:
    """Unique-word fraction (descriptive only — NOT a degeneration rule).

    Reported alongside the canonical detector where completions are
    repetitive without tripping the strict loop rules.
    """
    words = [w.lower() for w in text.strip().split()]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------

_Z95 = 1.959963984540054


def wilson_ci(k: int, n: int, z: float = _Z95) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_ci_upper(k: int, n: int, z: float = _Z95) -> float:
    """Upper bound of the Wilson score interval."""
    return wilson_ci(k, n, z)[1]
