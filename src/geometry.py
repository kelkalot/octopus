"""Unified residual-stream geometry probe.

One code path measures steering geometry for every condition (single
feature, joint set, random direction), so the matched-geometry comparison
in the paper is produced by a single estimator.

The recorder wraps the steering hook. On every forward call through the
steered layer it computes, within that call:

    h_pre   -- the layer output before the steering addition
    h_post  -- h_pre + c * direction

and records per-position norm ratio ``|h_post| / |h_pre|``, perturbation
norm ``|h_post - h_pre|``, and ``cos(h_post, h_pre)``. Prefill calls
(sequence length > 1) populate the prompt-position statistics, including
the last-prompt-token value where generation starts; decode calls
(sequence length == 1 under KV caching) accumulate per-step completion
statistics.

Summary keys (all conditions report the same keys):

    norm_ratio_prompt_mean        mean over prompt-forward positions
    norm_ratio_last_prompt_token  value at the final prompt position
    norm_ratio_completion_mean/sd mean +/- sd over decode steps
    cos_prompt_mean, cos_last_prompt_token,
    cos_completion_mean/sd        same for cosine-to-unperturbed
    perturb_norm_prompt_mean      mean |h_post - h_pre| over prompt positions
    n_prompt_positions, n_decode_steps

Historical note: dumps produced before this module existed carry the key
``norm_ratio_at_prompt_end``. Despite its name, that value was read from
the hook state after ``model.generate()`` returned, i.e. from the final
decode step of the first sequence in the last batch, not from the prompt
forward. Analysis code treats it as a fallback for
``norm_ratio_completion_mean``; on the released dumps the two estimators
agree to ~0.01 because the ratio is nearly constant across positions.
"""

from __future__ import annotations

import math

import torch


class GeometryRecorder:
    """Accumulates within-call pre/post geometry across forward calls."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.prompt_ratios: list[float] = []
        self.prompt_cos: list[float] = []
        self.prompt_perturb: list[float] = []
        self.last_prompt_ratio: float | None = None
        self.last_prompt_cos: float | None = None
        self.decode_ratios: list[float] = []
        self.decode_cos: list[float] = []

    def record(self, h_pre: torch.Tensor, h_post: torch.Tensor) -> None:
        """Record one forward call. Tensors are (batch, seq, d_model)."""
        pre = h_pre.detach().to(torch.float32)
        post = h_post.detach().to(torch.float32)
        ratio = post.norm(dim=-1) / pre.norm(dim=-1).clamp_min(1e-8)
        cos = torch.nn.functional.cosine_similarity(post, pre, dim=-1)
        seq_len = pre.shape[1]
        if seq_len > 1:  # prefill over the prompt
            self.prompt_ratios.extend(ratio[0].tolist())
            self.prompt_cos.extend(cos[0].tolist())
            self.prompt_perturb.extend((post[0] - pre[0]).norm(dim=-1).tolist())
            self.last_prompt_ratio = float(ratio[0, -1])
            self.last_prompt_cos = float(cos[0, -1])
        else:  # single-token decode step; average over the batch
            self.decode_ratios.append(float(ratio.mean()))
            self.decode_cos.append(float(cos.mean()))

    @staticmethod
    def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        m = sum(values) / len(values)
        if len(values) < 2:
            return m, 0.0
        var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
        return m, math.sqrt(var)

    def summary(self) -> dict:
        pr_m, _ = self._mean_sd(self.prompt_ratios)
        pc_m, _ = self._mean_sd(self.prompt_cos)
        pp_m, _ = self._mean_sd(self.prompt_perturb)
        dr_m, dr_s = self._mean_sd(self.decode_ratios)
        dc_m, dc_s = self._mean_sd(self.decode_cos)
        return {
            "norm_ratio_prompt_mean": pr_m,
            "norm_ratio_last_prompt_token": self.last_prompt_ratio,
            "norm_ratio_completion_mean": dr_m,
            "norm_ratio_completion_sd": dr_s,
            "cos_prompt_mean": pc_m,
            "cos_last_prompt_token": self.last_prompt_cos,
            "cos_completion_mean": dc_m,
            "cos_completion_sd": dc_s,
            "perturb_norm_prompt_mean": pp_m,
            "n_prompt_positions": len(self.prompt_ratios),
            "n_decode_steps": len(self.decode_ratios),
        }


def make_steering_hook(state: dict, recorder: GeometryRecorder | None = None):
    """Forward hook that adds ``state['coef'] * state['vec']`` to the layer
    output and (optionally) records within-call geometry.

    ``state`` is a mutable dict so the coefficient/direction can be changed
    between sweeps without re-attaching the hook. ``state['vec']`` may be a
    single direction or an already-summed joint direction.
    """

    def hook(module, args_, output):
        h = output[0] if isinstance(output, tuple) else output
        coef = float(state.get("coef", 0.0))
        vec = state.get("vec")
        if vec is not None and coef != 0.0:
            h_post = h + coef * vec
        else:
            h_post = h
        if recorder is not None:
            recorder.record(h, h_post)
        if isinstance(output, tuple):
            return (h_post,) + output[1:]
        return h_post

    return hook


def norm_ratio_of_record(record: dict) -> float | None:
    """Read a record's norm ratio, preferring the unified-probe key and
    falling back to the legacy pre-fix key (see module docstring)."""
    for key in ("norm_ratio_completion_mean", "norm_ratio_at_prompt_end"):
        if key in record and record[key] is not None:
            return record[key]
    return None


def cos_of_record(record: dict) -> float | None:
    """Cosine-to-unperturbed analogue of :func:`norm_ratio_of_record`."""
    for key in ("cos_completion_mean", "cos_to_pre_at_prompt_end"):
        if key in record and record[key] is not None:
            return record[key]
    return None
