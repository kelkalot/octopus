# Pairwise matrix protocol — reproducibility package

This repository accompanies the paper *"Pairwise matrix protocol for sparse autoencoder feature inspection"* (NeurIPS 2026 submission). It contains:

- All Python source for the four-phase pipeline (`src/`).
- All JSON data dumps required to re-derive every numerical claim, table, and figure in the paper, without re-running any model (`data/`).
- LaTeX sources for the paper itself (`paper/`).
- The NeurIPS reproducibility checklist (`checklist.tex`).

The reproducibility design has two tiers:

1. **Data-only reproduction** (≈ 5 minutes, no GPU). Uses the bundled JSONs to re-derive every percentage, Wilson confidence interval, norm ratio, and figure that appears in the paper.
2. **Full from-scratch reproduction** (≈ 12–15 hours on an Apple M4 Pro / 48 GB / MPS / fp16 laptop). Re-runs Phase 1 generation, Phase 2 cluster identification, Phase 3 SAE feature ranking, and Phase 4 dose-response and joint-condition sweeps.

## Quick start (data-only)

```bash
# 1. Install
uv sync

# 2. Regenerate every numerical claim and both body figures.
python src/regenerate_tables_and_figures.py
```

The script prints a side-by-side report of paper claims vs regenerated values and writes `paper/figures/figure1_three_axes.{pdf,png}` and `paper/figures/figure2_gemma_replication.{pdf,png}`.

For a written log:

```bash
python src/regenerate_tables_and_figures.py --out reproduce_report.txt
```

## Full from-scratch reproduction

Requirements:

- Python 3.11 (managed by `uv`; pinned in `.python-version`).
- Apple Silicon with MPS (or a CUDA GPU with the same memory budget). The paper's experiments fit on a 48 GB M4 Pro laptop using fp16.
- Network access on first run to fetch model weights and SAEs (Qwen3-1.7B / Gemma-2-2B-it / Llama-3.1-8B-Instruct and their respective Scope / Goodfire SAEs).

Run by phase:

```bash
# Phase 1 — generation (~75–100 minutes Qwen and Gemma; ~9 hours Llama)
python src/generate.py --model qwen
python src/generate.py --model gemma
python src/generate.py --model llama

# Phase 2 — cluster identification + pool construction (<1 minute)
python src/build_pools.py --model qwen
python src/build_pools.py --model gemma
python src/build_pools.py --model llama
python src/analyse_pools.py

# Phase 3 — SAE feature ranking, bootstrap, permutation null (~15–30 minutes)
python src/sae_features.py --model qwen --layer 20
python src/bootstrap_ranks.py
python src/permutation_test.py

# Phase 4 — dose-response and joint-condition sweeps (~30 min per feature)
python src/steer.py --feature 26221 --coefficients -1000 -500 0 500 1000 \
                    --prompts prompts/identity_probes.txt --samples 12 \
                    --out data/interventions/f2_identity_26221.json
python src/steer.py --features 29108 26221 4405 \
                    --coefficients -1500 -1000 -500 0 500 1000 \
                    --prompts prompts/intervention_mixed.txt --samples 12 \
                    --out data/interventions/joint_suppression.json
python src/random_direction_control.py --num-directions 50 --coefficients -1000 \
                    --prompts prompts/intervention_mixed.txt \
                    --out data/interventions/random_direction_K50_at_c-1000.json

# (further commands for Gemma joint+matched-geometry, K=5 sweep,
#  norm probes, NLL coherence — see ARTEFACTS.md)
```

See [`ARTEFACTS.md`](ARTEFACTS.md) for the full mapping of data files to paper claims.

## What's in this repo

```
src/                         # generation, steering, analysis, plot scripts
data/
├── pools/, pools_gemma/, pools_llama/    # Phase 1 raw + Phase 2 pools (per model)
├── activations/, activations_gemma/, activations_llama/  # Phase 3 SAE rankings
└── interventions/                        # Phase 4 sweeps + relabel + matched-geometry
paper/                       # LaTeX sources, figures, REVIEW_FIXES.md
prompts/                     # all prompt files used in Phase 1 and Phase 4
checklist.tex                # NeurIPS reproducibility checklist (filled)
pyproject.toml, uv.lock      # pinned environment
```

## License

Code, prompts, and documentation: **CC BY 4.0** (see [`LICENSE`](LICENSE)).

Data files (generation samples, intervention outputs, SAE feature rankings) are released under the same CC BY 4.0 license. The underlying models (Qwen3-1.7B-Instruct, Gemma-2-2B-it, Llama-3.1-8B-Instruct) and SAEs (Qwen-Scope, Gemma-Scope, Goodfire) are not redistributed; users must accept their respective licenses.

## Citation

If you use this work, please cite:

```bibtex
@misc{riegler2026pairwisematricessparseautoencoders,
      title={Pairwise matrices for sparse autoencoders: single-feature inspection mislabels causal axes}, 
      author={Michael A. Riegler and Birk Sebastian Frostelid Torpmann-Hagen},
      year={2026},
      eprint={2605.03160},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.03160}, 
}
```

## Contact
Michael A. Riegler - michael@simula.no
Birk Torpmann-Hagen - birk@simula.no
