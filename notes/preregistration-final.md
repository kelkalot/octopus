# Pre-registration — final-version experiments

Written and committed BEFORE the corresponding generations were run. Feature
selections below are design choices fixed in advance; the selection rules and
the resulting feature IDs are recorded in
`data/interventions/_experiment_targets.json`. Every prediction states what
each outcome would mean, so that either outcome is publishable as stated.

Environment: Qwen3-1.7B / Qwen-Scope L20 w32k; Gemma-2-2B-it / Gemma-Scope L20
canonical; Llama-3.1-8B-Instruct / Goodfire L19. Coefficient scales fixed to
the residual-stream norm as in Methods (±1000 / ±400 / ±10).

## P1. Unrelated content-bearing triples (Qwen)

**Question.** Is the joint-suppression collapse of §4.2 a property of *these
three cluster-selective features* (identity claim) or of *any three
content-bearing directions of matched geometry* (count claim)?

**Selection rule (fixed in advance).** Features with mean Pool-C activation
>= 1.0 (content-bearing on control prompts; top ~1.4% of the 32k dictionary,
71 features qualify) and |combined-z ranking statistic| < 0.5 (not
cluster-selective). Triples drawn disjointly with pairwise |cos| <= 0.25 and
sum-norm within 0.03 of the paper triple's 1.912. Five triples:
{173, 2898, 4306}, {2168, 4317, 9334}, {2275, 5354, 32569},
{4138, 16375, 19547}, {4398, 6177, 8095}.

**Run.** c = -500 (the paper's diagnostic edge), six intervention prompts,
12 samples: 360 generations. Metrics: strict placeholder detector, canonical
degeneration, lexical diversity (TTR / clean fraction), NLL, geometry.

**Predictions.**
- If unrelated triples do NOT produce the placeholder pattern at rates
  comparable to {29108, 26221, 4405} (7/72 pooled; 4/12 on recipe), §4.2
  stands as an identity claim about cluster-selective content axes.
- If they DO, §4.2 becomes a count claim: suppressing any three
  content-bearing directions of this magnitude collapses grounded
  composition. The paper then reports the count claim, which is weaker about
  these features and stronger about the mechanism, and the abstract changes
  accordingly.
- Intermediate outcome (placeholder pattern present but at materially lower
  rate) is reported as a graded result with both rates and CIs.

## P2. Llama grid-level tests (the instruct-trained SAE)

**Triple (fixed in advance).** {38565, 61417, 23576} — the top-ranked Class-1
feature plus the two top-10 features that minimise pairwise |cos| with it.
Measured pairwise cosines 0.0404 / 0.0254 / 0.0347; sum-norm 1.789.

**Runs.** (a) joint sweep at c ∈ {-10, -5, 0, +5, +10}, six intervention
prompts, 8 samples = 240 generations; (b) matched-geometry random-direction
control, K = 5 unit directions at |c| = 10 × 1.789 ≈ 17.9 on the three
control prompts, 8 samples = 120 generations.

**Diagnostic-edge prediction (stated before the run).** §6's saturation
principle says the diagnostic edge sits where the model has headroom. Llama's
baseline is intermediate — disclaimer 38.3% over all prompts, cluster 96.7%
on intervention intros, zero canonical degeneration across the existing 420
generations, and lexical diversity at c=+10 (TTR 0.68, 91.7% clean) *above*
its own baseline (TTR 0.62, 46.7% clean). Prediction: the joint-condition
effect on control prompts surfaces at **amplification** (c=+10), as on Gemma,
not at suppression; and Llama's steered text remains coherent by the
diversity metric where Qwen's does not.

**Interpretation, fixed in advance.**
- Replication on Llama = three-for-three, and the base-vs-instruct confound
  of §7 loses its force.
- Non-replication = evidence that the grid-level effects partly reflect
  base-trained SAEs applied to post-trained activations. This is reported as
  a result in the cross-model section, not as a footnote, and the §7
  limitation is upgraded to a finding.
- Edge prediction failing (effect at suppression instead) is reported as a
  failure of the saturation heuristic.

## P3. Coherence re-analysis (no new generation)

**Question.** Do the paper's coherence claims survive a diversity-based
measure? Fixed metric: type-token ratio and "clean" = no 5-gram repeated
within a completion AND TTR >= 0.60.

**Prediction, stated before writing the results section.** Qwen cells at
|c| >= 500 will show materially reduced diversity while the canonical
three-rule detector reports ~0%; Gemma at c=-200 and Llama at c=+10 will
not. Consequence either way: the diversity signal joins the metrics table and
every coherence claim is restated against it, whichever cells pass.

## P4. Matched-geometry probe re-run

Random-direction dumps are regenerated with the unified probe so all
matched-geometry rows share one estimator. Prediction: values move by
<= 0.02 (the two estimators already agree to ~0.01), and the matched-geometry
conclusion is unchanged. A larger movement is reported as an estimator
correction.
