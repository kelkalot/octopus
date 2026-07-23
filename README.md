# Steering grids for SAE features — code and data

This repository accompanies the paper *"Steering grids for sparse-autoencoder
features: top-context labels capture activation regimes, not causal axes"*
(arXiv:2605.03160). It contains:

- All Python source for the four-phase pipeline and the steering-grid
  experiments (`src/`).
- All JSON sample dumps required to re-derive every numerical claim, table,
  and figure in the paper, without re-running any model (`data/`).
- LaTeX sources for the paper (`paper/`), including the NeurIPS
  reproducibility checklist (`paper/checklist.tex`).
- All prompt files (`prompts/`).

Reproducibility has two tiers:

1. **Data-only reproduction** (minutes, no GPU). Re-derives every percentage,
   Wilson confidence interval, norm ratio, and figure in the paper from the
   bundled JSONs, and asserts each value against the number printed in the
   paper. The script exits non-zero on any mismatch.
2. **Full from-scratch reproduction** (~12–15 hours on an Apple M4 Pro /
   48 GB / MPS / fp16 laptop). Re-runs Phase 1 generation, Phase 2 cluster
   identification, Phase 3 SAE feature ranking, and the Phase 4 steering-grid
   sweeps.

## Quick start (data-only)

```bash
# 1. Install (pins Python 3.11, spaCy, and the en_core_web_sm 3.8.0 model)
uv sync

# 2. Re-derive and assert every numerical claim; render both data figures.
uv run python src/regenerate_tables_and_figures.py
```

The script prints each regenerated value next to the paper's value with an
`ok`/`FAIL` mark, writes `paper/figures/figure1_three_axes.{pdf,png}` and
`paper/figures/figure2_gemma_replication.{pdf,png}`, and exits 0 only if all
checks pass. For a written log:

```bash
uv run python src/regenerate_tables_and_figures.py --out reproduce_report.txt
```

All text metrics (disclaimer regex, lemma clusters, degeneration detector,
placeholder detector, we-voice, Wilson CIs) are implemented once in
`src/detectors.py` and imported everywhere, including by this script.
Cluster metrics depend on the spaCy lemmatizer, so `en_core_web_sm` is
pinned at 3.8.0 in `pyproject.toml`.

## Full from-scratch reproduction

Requirements: Python 3.11 (managed by `uv`), Apple Silicon with MPS or a
CUDA GPU with a comparable memory budget, and network access on first run to
fetch model weights and SAEs (Qwen3-1.7B / Gemma-2-2B-it /
Llama-3.1-8B-Instruct and their Qwen-Scope / Gemma-Scope / Goodfire SAEs).

```bash
# Phase 1 — generation (~75–100 min for Qwen and Gemma; ~9 h for Llama)
uv run python src/generate.py \
    --prompts prompts/introspective.txt prompts/controls.txt \
    --model Qwen/Qwen3-1.7B --samples 100 \
    --out data/pools/raw_generations.json

# Phase 2 — cluster identification + pool construction (<1 min)
uv run python src/build_pools.py \
    --in data/pools/raw_generations.json --out-dir data/pools
uv run python src/analyse_pools.py

# Phase 3 — SAE feature ranking, bootstrap, permutation null (~15 min/layer)
uv run python src/sae_features.py \
    --pools-dir data/pools --out-dir data/activations \
    --model Qwen/Qwen3-1.7B --release qwen-scope-3-1.7b-base-w32k-l50 --layer 20
uv run python src/bootstrap_ranks.py \
    --activations-dir data/activations --layer 20 \
    --out data/activations/sae_layer20_bootstrap.json \
    --figure figures/phase3_rank_stability.png
uv run python src/permutation_test.py \
    --activations-dir data/activations --layer 20 --n-perm 200 \
    --out data/activations/sae_layer20_permutation.json \
    --figure figures/phase3_permutation_null.png

# Phase 4 — steering-grid sweeps (~30 min per feature)
uv run python src/steer.py --feature 26221 --layer 20 \
    --coefficients -1000 -500 0 500 1000 \
    --prompts prompts/identity_probes.txt --samples 12 \
    --out data/interventions/f2_identity_26221.json
uv run python src/steer.py --features 29108 26221 4405 --layer 20 \
    --coefficients -1500 -1000 -500 0 500 1000 \
    --prompts prompts/intervention_mixed.txt --samples 12 \
    --out data/interventions/joint_suppression.json
uv run python src/random_direction_control.py --num-directions 50 \
    --coefficients -1000 --prompts prompts/intervention_mixed.txt \
    --out data/interventions/random_direction_K50_at_c-1000.json
```

Gemma/Llama runs use the same commands with the model, release, layer, and
coefficient scale swapped (see `ARTEFACTS.md` for the full file-to-claim
mapping and per-model parameters).

The per-pool SAE activation matrices (`sae_layer*_pool_{A,B,C}.npz`,
~340 MB) are not stored in git. `bootstrap_ranks.py` and
`permutation_test.py` need them; regenerate them from the bundled Phase-1
samples with the `sae_features.py` command above (~15 minutes per layer),
or take them from the full release bundle.

## Prevalence sweep (top-50 screening harness)

`src/sweep_class1.py` runs the pre-registered mode-switch screen from the
paper over all 50 Class-1 features: one overnight generation run, then a
CPU-only screening pass.

```bash
uv run python src/sweep_class1.py run     # ~10 h, resumable
uv run python src/sweep_class1.py screen  # classification + Wilson CI report
```

## What's in this repo

```
src/                  # pipeline, steering-grid harness, analysis, plots
  detectors.py        # every text metric, single source of truth
  geometry.py         # unified residual-stream geometry probe
data/
├── pools*/           # Phase 1 raw generations + Phase 2 pools (per model)
├── activations*/     # Phase 3 rankings, bootstrap, permutation summaries
└── interventions/    # Phase 4 sweeps, relabelling, matched-geometry runs
paper/                # LaTeX sources, figures, checklist.tex
prompts/              # all prompt files used in Phases 1 and 4
scripts/              # release-bundle builder
pyproject.toml, uv.lock  # pinned environment (incl. the spaCy model)
```

## License

Code, prompts, data files, and documentation: **CC BY 4.0** (see
[`LICENSE`](LICENSE)). The underlying models (Qwen3-1.7B, Gemma-2-2B-it,
Llama-3.1-8B-Instruct) and SAEs (Qwen-Scope, Gemma-Scope, Goodfire) are not
redistributed; users must accept their respective licenses.

## Citation

If you use this work, please cite:

```bibtex
@misc{riegler2026steeringgrids,
      title={Steering grids for sparse-autoencoder features: top-context
             labels capture activation regimes, not causal axes},
      author={Michael A. Riegler and Birk Sebastian Frostelid Torpmann-Hagen},
      year={2026},
      eprint={2605.03160},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.03160},
}
```

## Contact

Michael A. Riegler — michael@simula.no
Birk Torpmann-Hagen — birk@simula.no
