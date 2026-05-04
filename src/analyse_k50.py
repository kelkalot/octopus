"""Analyse K=50 random-direction control. Reports placeholder rate at c=-1000 with
the strict detector defined in §4.3. Updates appendix table values."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path("data/interventions")


def is_placeholder_pattern(text: str) -> bool:
    """Strict placeholder-pattern detector: parenthetical uppercase code tokens
    appearing >=2 times in a single completion, or Vc.\\,N+ numeric placeholders."""
    t = text.strip()
    code_paren = re.findall(r"\(\s*[A-Z]{2,5}(?:\s*[A-Z\d]+)?\s*\)", t)
    if len(code_paren) >= 2:
        return True
    if re.search(r"\b[Vv]c\.\s*\d+\+?", t):
        return True
    return False


def wilson_ci_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Upper bound of Wilson score interval at 1-alpha confidence."""
    if n == 0:
        return 1.0
    from math import sqrt
    z = 1.959963984540054  # z for 95% (alpha=0.05)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return centre + half


def main() -> None:
    rd = json.loads((DATA / "random_direction_K50_at_c-1000.json").read_text())
    print(f"Loaded {len(rd)} records from K=50 run")
    by_dir: dict[int, list] = defaultdict(list)
    for r in rd:
        by_dir[r["direction_idx"]].append(r)
    n_dirs = len(by_dir)
    n_per_dir = {k: len(v) for k, v in by_dir.items()}
    print(f"K = {n_dirs} directions; per-direction sample counts: "
          f"min={min(n_per_dir.values())}, max={max(n_per_dir.values())}")

    flagged = sum(1 for r in rd if is_placeholder_pattern(r["completion"]))
    n_total = len(rd)
    rate = flagged / n_total if n_total else 0
    upper95 = wilson_ci_upper(flagged, n_total)
    print(f"\nK=50 random-direction at c=-1000: {flagged} of {n_total} flagged "
          f"({rate*100:.2f}%), Wilson 95% upper bound = {upper95*100:.2f}%")

    # K=5 reference
    rd5 = json.loads((DATA / "random_direction_matched.json").read_text())
    rd5_minus1000 = [r for r in rd5 if r["coefficient"] == -1000.0]
    flagged5 = sum(1 for r in rd5_minus1000 if is_placeholder_pattern(r["completion"]))
    upper95_5 = wilson_ci_upper(flagged5, len(rd5_minus1000))
    print(f"\nK=5 random-direction at c=-1000 (reference): "
          f"{flagged5} of {len(rd5_minus1000)} flagged ({100*flagged5/len(rd5_minus1000):.2f}%), "
          f"Wilson 95% upper = {upper95_5*100:.2f}%")

    # Joint reference at c=-500
    joint = json.loads((DATA / "joint_suppression.json").read_text())
    joint_500 = [r for r in joint if r["coefficient"] == -500.0]
    flagged_j = sum(1 for r in joint_500 if is_placeholder_pattern(r["completion"]))
    upper95_j = wilson_ci_upper(flagged_j, len(joint_500))
    print(f"\nJoint suppression at c=-500 (reference): "
          f"{flagged_j} of {len(joint_500)} flagged ({100*flagged_j/len(joint_500):.2f}%), "
          f"Wilson 95% upper = {upper95_j*100:.2f}%")

    # Coherence sanity check on K=50: regex-degenerate rate
    deg_rate = sum(1 for r in rd if r["regex_degenerate"]) / len(rd)
    print(f"\nK=50 regex-degenerate rate: {deg_rate*100:.2f}%")

    # Norm ratio sanity
    seen = set()
    norm_ratios = []
    for r in rd:
        key = (r["direction_idx"], r["coefficient"], r["prompt"])
        if key in seen:
            continue
        seen.add(key)
        norm_ratios.append(r["norm_ratio_at_prompt_end"])
    print(f"K=50 norm_ratio at c=-1000: mean={np.mean(norm_ratios):.3f}, "
          f"std={np.std(norm_ratios):.3f}, n_unique_geom={len(norm_ratios)}")

    summary = {
        "K50_random_at_c-1000": {
            "flagged": flagged,
            "n_total": n_total,
            "rate": rate,
            "wilson_95_upper": upper95,
        },
        "K5_random_at_c-1000_reference": {
            "flagged": flagged5,
            "n_total": len(rd5_minus1000),
            "rate": flagged5 / len(rd5_minus1000),
            "wilson_95_upper": upper95_5,
        },
        "joint_at_c-500_reference": {
            "flagged": flagged_j,
            "n_total": len(joint_500),
            "rate": flagged_j / len(joint_500),
            "wilson_95_upper": upper95_j,
        },
    }
    out = DATA / "k50_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
