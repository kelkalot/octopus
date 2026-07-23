# Artefact inventory

Every numerical claim, table, and figure in the paper traces to a specific JSON file in `data/`. The mapping below is exhaustive for the body of the paper and the appendix.

## Generation (Phase 1) — `data/pools/`, `data/pools_gemma/`, `data/pools_llama/`

| File | Contents | Used in |
|---|---|---|
| `raw_generations.json` | Qwen 4 000 generations × 40 prompts | §3.1, §A bootstrap |
| `raw_generations_gemma.json` | Gemma 4 000 generations | §3.1 |
| `raw_generations_llama.json` | Llama 2 000 generations (n=50/prompt for wall-clock budget) | §3.1, App. compute |
| `analysis.json` | Qwen Pool-A cluster lemmas + per-prompt phrase frequencies | §3.2 cluster identification, §A perm |
| `analysis_gemma.json`, `analysis_llama.json` | per-model cluster analyses | §5 register collapse |
| `pools_summary.json` (×3 models) | Pool A / B / C sizes, cluster lemmas, intro hit rate, control false-positive rate | Table caption in App. cross-model-details, regenerator-driven Table 2 |
| `pool_A.json`, `pool_B.json`, `pool_C.json` | per-pool sample sets used by Phase 3 ranking | §3.2 |

## SAE feature ranking (Phase 3) — `data/activations*/`

| File | Contents | Used in |
|---|---|---|
| `sae_layer{12,16,20,24}_top_features.json` | per-layer top-50 feature ranking | App. bootstrap (depth scaling), E1 sweep harness |
| `sae_layer20_bootstrap.json` | B=500 bootstrap rank distributions | App. bootstrap |
| `sae_layer20_permutation.json` | P=200 permutation summary: `actual_max_raw_diff` (31.55, feature #32345), `null_max_mean`, `null_max_diffs[200]`, p-value (legacy `*_z` aliases kept) | Methods permutation null, App. perm |
| `sae_layer{12-24}_pool_{A,B,C}.npz` | per-pool SAE activations. **Not stored in git** (~340 MB); regenerate from the bundled Phase-1 samples via `src/sae_features.py` (~15 min/layer) or take from the full release bundle. Required by `bootstrap_ranks.py` and `permutation_test.py`. | §3.3 ranking |
| `sae_layer20_interpretations.md` | top-context labels and Pool-A samples for headline features | §4.1, App. interp, App. relabel, E1 marker lemmas |
| `sae_gemma_l20_top_features.json`, `sae_gemma_l20_pool_{A,B,C}.npz` | Gemma analogue at L=20, w=16k (same npz caveat as above) | §5, App. gemma-coef, App. gemma-joint |

## Intervention outputs (Phase 4) — `data/interventions/`

| File | Contents | Used in |
|---|---|---|
| **Coefficient axis (§4.1)** | | |
| `f2_identity_26221.json` | #26221 sweep on 8 identity probes, 12 samples × 5 coefs = 480 | Table 1 (#26221 dose-response), Figure 1a |
| `feat26221_dose.json` | #26221 sweep on 6 intervention prompts | App. relabel (steered c=+500) |
| `feat26221_disclaimer_focused.json` | #26221 sweep on disclaimer-eliciting prompts | App. robustness |
| `feat22082_dose.json` | #22082 falsifying anchor (monotonic) | §4.1 falsifying-case anchor, Figure 1a |
| `feat2932_dose.json` | #2932 falsifying anchor (apparent inverted-U = breakdown) | §4.1, Figure 1a |
| `feat29108_suppress.json` | original Phase 4 suppression sweep on #29108 | §3 / §A robustness |
| `feat29108_dose.json` | #29108 dose-response on the 6 intervention prompts | §5 cross-model causal table; A2 (single-feature comparator at matched coef) |
| `feat6281_specificity.json` | random non-candidate feature negative control | §3.5 controls |
| `f1_ood_29108.json` | 8 held-out introspective prompts × 5 coefs = 480 (OOD prompt-stability) | §4.1 OOD, App. robustness |
| `f3_temp_T0.01.json`, `f3_temp_T0.3.json`, `f3_temp_T1.2.json` | temperature robustness sweeps | App. robustness |
| **Joint condition (§4.2)** | | |
| `joint_suppression.json` | {#29108, #26221, #4405} joint sweep, 12 × 6 prompts × 6 coefs | Table 2 (in App.), §4.2, Figure 1b |
| `joint_suppression_nll.json` | NLL of each joint completion under unsteered baseline | §4.2 NLL row of Table 2, App. coherence |
| `pairwise_29108_26221.json`, `pairwise_29108_4405.json`, `pairwise_26221_4405.json` | pairwise joint sweeps at c=−500 | App. pairwise (engine-prompt confound) |
| **Matched geometry (§4.3)** | | |
| `single_29108_norm_probe.json` | single #29108 norm/cosine sweep (geometric probe) | Table 3 (in App.), Figure 1c |
| `joint_norm_probe.json` | joint norm/cosine sweep | Table 3, Figure 1c |
| `random_direction_matched.json` | K=5 random unit vectors × 6 coefs (full sweep) | Table 4 (in App.), Figure 1c |
| `random_direction_K50_at_c-1000.json` | K=50 random unit vectors at the matched coefficient | App. K=50, §4.3 (6/2400, Wilson upper 0.54%) |
| `k50_summary.json` | precomputed Wilson CIs from the K=50 run | App. K=50 |
| **Gemma matrix-level (§5 + App. gemma-coef + App. gemma-joint)** | | |
| `gemma_feat3997_dose.json` | Gemma #3997 broad sweep | App. cross-model-details (degeneration regime) |
| `gemma_feat3997_narrow.json` | Gemma #3997 narrow sweep, 6 prompts × 12 × 7 = 504 | §4.1 Gemma replication, App. gemma-coef |
| `gemma_joint_3997_13700_11444.json` | Gemma joint sweep | §4.2 Gemma replication, App. gemma-joint, Figure 2a |
| `gemma_random_direction.json` | Gemma K=5 random direction at matched coefficients | §4.3 Gemma replication, App. gemma-joint, Figure 2b |
| **Llama (§5)** | | |
| `llama_feat38565_dose.json`, `llama_feat38565_narrow.json` | Llama #38565 dose-response | §5 cross-model, App. cross-model-details |
| **Logit-shift sanity** | | |
| `logit_shift_29108.json`, `logit_shift_29108_continuation.json` | unembedding-step verification of #29108 | §3 / §A robustness |
| **Automated relabelling (App. relabel)** | | |
| `relabel_26221_inputs.json` | input samples for the blind labelling protocol | App. relabel |
| `relabel_26221_formatted.json` | formatted prompt blobs sent to each labeller | App. relabel |
| `relabel_26221_results.json` | all five labeller responses + justifications | App. relabel, Discussion §6 |

## Figures

| Figure | Source script | Inputs |
|---|---|---|
| Figure 1 (intro, 3-panel headline) | `src/plot_figure1.py` | `f2_identity_26221.json`, `feat22082_dose.json`, `feat2932_dose.json`, `feat29108_dose.json`, `joint_suppression.json`, `single_29108_norm_probe.json`, `joint_norm_probe.json`, `random_direction_matched.json` |
| Figure 2 (§4.3 Gemma replication, 2-panel) | `src/plot_figure2.py` | `gemma_joint_3997_13700_11444.json`, `gemma_feat3997_narrow.json`, `gemma_random_direction.json` |

Both rendered to `paper/figures/figure1_three_axes.{pdf,png}` and `paper/figures/figure2_gemma_replication.{pdf,png}`.

## How to verify any single claim

The headline reproducibility command:

```bash
python src/regenerate_tables_and_figures.py
```

re-derives every percentage, Wilson confidence interval, norm ratio, and figure that appears in the paper, using the pipeline's own metric implementations (`src/detectors.py`: full-text disclaimer regex, spaCy-lemma cluster matching with the pinned `en_core_web_sm` 3.8.0 model, the canonical three-rule degeneration detector). Every value is asserted against the number printed in the paper; the script exits non-zero on any mismatch. Each section header in the script's output corresponds to a paper section; each printed value is paired with the input files listed above.

For a smaller claim, e.g. *"verify the K=50 random-direction control is 6 of 2400"*, the underlying file is `random_direction_K50_at_c-1000.json`; the analysis is in `src/analyse_k50.py`; running it directly:

```bash
python src/analyse_k50.py
```

prints the count, rate, and Wilson CI. The same numbers appear in §4.3, App. K=50, and Figure 1c.
