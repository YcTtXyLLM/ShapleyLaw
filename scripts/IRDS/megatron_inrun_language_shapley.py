#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Megatron-LM friendly language-level In-Run Data Shapley helper.

Goal
----
During multilingual LLM pretraining, estimate how much each training language
contributes to the model's performance on ONE target validation language.

For each attribution micro-step, this module computes a 7-dimensional vector:

    lang_score[l] += sum_{samples i with language_id=l}
                     -lr_t * <grad loss_val_target, grad loss_train_i>

This is the first-order In-Run Data Shapley approximation from
"Data Shapley in One Training Run".

Recommended use
---------------
1. Keep your Megatron-LM pretraining loop unchanged.
2. Add a language_id to each sequence in your FineWeb2 multilingual dataset.
3. Create a held-out validation iterator for the target language.
4. Every N training iterations, run one attribution micro-step with this helper.
5. Save language-level cumulative scores to JSONL/CSV.

Important engineering assumptions
---------------------------------
- The model is Megatron GPT-style and returns per-token losses when labels are
  passed, as in Megatron-LM pretrain_gpt.py.
- Sequence lengths of train and target-validation batches are the same.
- This file supports normal dense micro-batches, not packed THD/SFT batches.
- Hooks target Linear-like modules with a 2D weight parameter, including
  torch.nn.Linear and Megatron tensor-parallel linear modules.
- Works best with tensor parallel + data parallel. Pipeline parallel can work
  if all PP stages run the attribution forward/backward and receive language_id;
  otherwise start with pipeline-model-parallel-size=1 for attribution runs.
- Bias attribution is disabled by default because Megatron TP layers may shard
  or fuse bias differently. Weight-only attribution is usually the important part.

Suggested Megatron patch points
-------------------------------
In your custom pretrain_gpt.py:
    - after model creation: create LanguageInRunShapley(...)
    - inside the training loop, every args.shapley_attribute_every:
          train_batch = get the current micro-batch, including language_id
          val_batch = next(target_language_valid_iterator)
          shapley.attribute(...)
    - periodically call shapley.write_jsonl(...)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


try:
    from megatron.core import parallel_state
except Exception:
    parallel_state = None


TensorBatch = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------
# Distributed utilities
# ---------------------------------------------------------------------

def _is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _safe_group(fn_name: str):
    if parallel_state is None:
        return None
    fn = getattr(parallel_state, fn_name, None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


def _all_reduce_sum_(x: torch.Tensor, group=None) -> torch.Tensor:
    if _is_dist_ready():
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
    return x


def _get_rank() -> int:
    return dist.get_rank() if _is_dist_ready() else 0


def _is_global_rank0() -> bool:
    return _get_rank() == 0


def reduce_scores_for_megatron_(
    lang_scores: torch.Tensor,
    lang_token_counts: torch.Tensor,
    reduce_tensor_parallel: bool = True,
    reduce_pipeline_parallel: bool = True,
    reduce_data_parallel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reduce language scores and counts in a Megatron-friendly way.

    Scores:
      - Sum over TP because each TP rank owns a shard of parameters.
      - Sum over PP because each PP stage owns a subset of layers.
      - Sum over DP because each DP rank sees different samples.

    Counts:
      - Sum over DP only.
      - Do NOT sum over TP/PP, because TP/PP ranks usually see the same samples.
    """
    if not _is_dist_ready():
        return lang_scores, lang_token_counts

    if reduce_tensor_parallel:
        group = _safe_group("get_tensor_model_parallel_group")
        _all_reduce_sum_(lang_scores, group=group)

    if reduce_pipeline_parallel:
        group = _safe_group("get_pipeline_model_parallel_group")
        _all_reduce_sum_(lang_scores, group=group)

    if reduce_data_parallel:
        group = _safe_group("get_data_parallel_group")
        _all_reduce_sum_(lang_scores, group=group)
        _all_reduce_sum_(lang_token_counts, group=group)

    return lang_scores, lang_token_counts


# ---------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------

def cat_batch_dim(a: torch.Tensor, b: torch.Tensor, batch_dim: int = 0) -> torch.Tensor:
    return torch.cat([a, b], dim=batch_dim)


def build_combined_gpt_batch(
    train_batch: TensorBatch,
    val_batch: TensorBatch,
    keys: Sequence[str] = ("tokens", "labels", "loss_mask", "position_ids"),
    batch_dim: int = 0,
) -> TensorBatch:
    """
    Build a combined train+validation batch for a Megatron GPT forward.

    The common Megatron GPT batch contains:
      tokens, labels, loss_mask, attention_mask, position_ids

    attention_mask is often shared/broadcasted and should usually NOT be
    concatenated along batch. We keep train_batch["attention_mask"] by default.
    """
    combined: TensorBatch = {}

    for k in keys:
        if k in train_batch and k in val_batch and train_batch[k] is not None:
            combined[k] = cat_batch_dim(train_batch[k], val_batch[k], batch_dim=batch_dim)

    if "attention_mask" in train_batch:
        combined["attention_mask"] = train_batch["attention_mask"]

    if train_batch.get("packed_seq_params", None) is not None:
        raise NotImplementedError("Packed THD/SFT batches are not supported by this helper yet.")

    return combined


def sequence_loss_from_token_losses(
    token_losses: torch.Tensor,
    loss_mask: torch.Tensor,
    batch_dim: int = 0,
    normalize_by_tokens: bool = True,
) -> torch.Tensor:
    """
    Convert Megatron per-token losses into one scalar loss per sequence.

    Megatron GPT commonly returns output_tensor that can be viewed like loss_mask.
    loss_mask is usually [B, S] when batch_dim=0.
    """
    losses = token_losses.float().view_as(loss_mask).float()
    mask = loss_mask.float()

    if batch_dim != 0:
        losses = losses.transpose(0, batch_dim).contiguous()
        mask = mask.transpose(0, batch_dim).contiguous()

    seq_loss = (losses * mask).sum(dim=1)
    if normalize_by_tokens:
        seq_loss = seq_loss / mask.sum(dim=1).clamp_min(1.0)
    return seq_loss


# ---------------------------------------------------------------------
# Ghost dot-product hook
# ---------------------------------------------------------------------

@dataclass
class GhostHookConfig:
    num_languages: int = 7
    activation_layout: str = "SBH"  # Megatron hidden states are commonly [seq, batch, hidden].
    include_bias: bool = False
    require_grad_output: bool = True
    fp32_accumulation: bool = True
    min_weight_ndim: int = 2


class MegatronLinearGhostDot:
    """
    Ghost dot-product for Linear-like modules.

    It accumulates per-training-sequence dot-products with the aggregate
    target-validation gradient. It does not instantiate per-sample gradients.

    Supported activation layouts:
      - "SBH": activation shape [seq, batch, hidden]
      - "BSH": activation shape [batch, seq, hidden]
      - "BH":  activation shape [batch, hidden]

    The combined batch must place training sequences first and validation
    sequences second.
    """

    def __init__(self, model: nn.Module, config: GhostHookConfig):
        self.model = model
        self.config = config
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.enabled: bool = False
        self.train_batch_size: Optional[int] = None
        self._inputs: Dict[int, torch.Tensor] = {}
        self._sample_dots: Optional[torch.Tensor] = None

    def install(self) -> None:
        self.remove()
        for module in self.model.modules():
            if self._is_linear_like(module):
                self.handles.append(module.register_forward_hook(self._save_input))
                self.handles.append(module.register_full_backward_hook(self._accumulate_dot))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def begin(self, train_batch_size: int, device: torch.device) -> None:
        self.enabled = True
        self.train_batch_size = int(train_batch_size)
        self._inputs.clear()
        self._sample_dots = torch.zeros(train_batch_size, device=device, dtype=torch.float32)

    def end(self) -> torch.Tensor:
        if self._sample_dots is None:
            raise RuntimeError("No ghost dot-products were accumulated.")
        out = self._sample_dots.detach().clone()
        self.enabled = False
        self.train_batch_size = None
        self._inputs.clear()
        self._sample_dots = None
        return out

    def _is_linear_like(self, module: nn.Module) -> bool:
        w = getattr(module, "weight", None)
        # Only modules with matrix-like weights can use the Linear ghost formula.
        # This includes torch.nn.Linear and Megatron/Transformer-Engine parallel
        # linear variants. Embeddings also have 2D weights, but their input is
        # integer-valued; _save_input() filters those out.
        return isinstance(w, torch.Tensor) and w.ndim == 2

    def _save_input(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output) -> None:
        if not self.enabled:
            return
        if not inputs or not torch.is_tensor(inputs[0]):
            return
        # Skip embeddings and other modules whose input is not an activation.
        if not inputs[0].is_floating_point():
            return
        self._inputs[id(module)] = inputs[0]

    @torch.no_grad()
    def _accumulate_dot(self, module: nn.Module, grad_input, grad_output) -> None:
        if not self.enabled or self.train_batch_size is None or self._sample_dots is None:
            return

        A = self._inputs.pop(id(module), None)
        if A is None:
            return

        G = None
        if isinstance(grad_output, tuple):
            for item in grad_output:
                if torch.is_tensor(item):
                    G = item
                    break
        elif torch.is_tensor(grad_output):
            G = grad_output

        if G is None:
            if self.config.require_grad_output:
                raise RuntimeError(f"Missing grad_output for module {module.__class__.__name__}")
            return

        A = A.detach()
        G = G.detach()
        if self.config.fp32_accumulation:
            A = A.float()
            G = G.float()

        normalized = self._to_positions_batch_feature(A, G, module)
        if normalized is None:
            return
        A, G = normalized

        B = self.train_batch_size
        if A.shape[1] <= B:
            raise RuntimeError(
                f"Combined batch size must be > train_batch_size. "
                f"Got activation batch={A.shape[1]}, train_batch={B}."
            )
        if G.shape[1] != A.shape[1] or G.shape[0] != A.shape[0]:
            # Some fused TE modules expose auxiliary tensors whose activation and
            # grad-output positions do not align with the Linear weight. They are
            # skipped rather than breaking attribution.
            return

        A_train, A_val = A[:, :B, :], A[:, B:, :]
        G_train, G_val = G[:, :B, :], G[:, B:, :]

        # F.linear convention: weight is [out, in]. P is the flattened position
        # axis: sequence positions plus any extra non-feature dimensions.
        val_w_grad = torch.einsum("pvo,pvi->oi", G_val, A_val)
        dots = torch.einsum("pbo,pbi,oi->b", G_train, A_train, val_w_grad)

        if self.config.include_bias and getattr(module, "bias", None) is not None:
            val_b_grad = G_val.sum(dim=(0, 1))
            dots = dots + torch.einsum("pbo,o->b", G_train, val_b_grad)

        self._sample_dots.add_(dots.to(self._sample_dots.dtype))

    @staticmethod
    def _prod(xs: Tuple[int, ...]) -> int:
        out = 1
        for x in xs:
            out *= int(x)
        return out

    def _flatten_activation(
        self,
        x: torch.Tensor,
        expected_feature_dim: int,
        layout: str,
    ) -> Optional[torch.Tensor]:
        """Return x as [P, B, C] for a Linear-like module.

        P is a flattened position axis, B is the combined train+validation
        batch axis, and C is the Linear input/output feature dimension.

        Transformer Engine sometimes exposes tensors such as [S, B, Hh, Dh],
        where Hh*Dh is a split feature dimension rather than an extra position
        dimension. This function uses the module weight shape to distinguish:
          - if product(trailing dims) == expected_feature_dim, trailing dims are
            a split feature dimension and P remains S;
          - if last dim == expected_feature_dim, intermediate trailing dims are
            true extra positions and are folded into P.
        """
        layout = layout.upper()

        if layout == "SBH":
            if x.ndim == 2:
                # [B, C] -> [1, B, C]
                if x.shape[-1] != expected_feature_dim:
                    return None
                return x.unsqueeze(0).contiguous()
            if x.ndim < 3:
                return None
            s, b = int(x.shape[0]), int(x.shape[1])
            rest = tuple(int(v) for v in x.shape[2:])
            if self._prod(rest) == expected_feature_dim:
                return x.reshape(s, b, expected_feature_dim).contiguous()
            if rest[-1] == expected_feature_dim:
                pos = self._prod(rest[:-1])
                return x.reshape(s, b, pos, expected_feature_dim).permute(0, 2, 1, 3).reshape(s * pos, b, expected_feature_dim).contiguous()
            return None

        if layout == "BSH":
            if x.ndim == 2:
                # [B, C] -> [1, B, C]
                if x.shape[-1] != expected_feature_dim:
                    return None
                return x.unsqueeze(0).contiguous()
            if x.ndim < 3:
                return None
            b, s = int(x.shape[0]), int(x.shape[1])
            rest = tuple(int(v) for v in x.shape[2:])
            if self._prod(rest) == expected_feature_dim:
                return x.reshape(b, s, expected_feature_dim).transpose(0, 1).contiguous()
            if rest[-1] == expected_feature_dim:
                pos = self._prod(rest[:-1])
                return x.reshape(b, s, pos, expected_feature_dim).permute(1, 2, 0, 3).reshape(s * pos, b, expected_feature_dim).contiguous()
            return None

        if layout == "BH":
            if x.ndim < 2:
                return None
            b = int(x.shape[0])
            rest = tuple(int(v) for v in x.shape[1:])
            if self._prod(rest) == expected_feature_dim:
                return x.reshape(1, b, expected_feature_dim).contiguous()
            if rest[-1] == expected_feature_dim:
                pos = self._prod(rest[:-1])
                return x.reshape(b, pos, expected_feature_dim).permute(1, 0, 2).contiguous()
            return None

        raise ValueError(f"Unsupported activation_layout={self.config.activation_layout}")

    def _to_positions_batch_feature(
        self,
        A: torch.Tensor,
        G: torch.Tensor,
        module: nn.Module,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            return None

        # F.linear convention: weight shape is [out_features, in_features].
        out_features = int(weight.shape[0])
        in_features = int(weight.shape[1])

        layout = self.config.activation_layout.upper()
        A2 = self._flatten_activation(A, in_features, layout)
        G2 = self._flatten_activation(G, out_features, layout)
        if A2 is None or G2 is None:
            return None
        return A2, G2


# ---------------------------------------------------------------------
# Main language-level accumulator
# ---------------------------------------------------------------------

@dataclass
class LanguageShapleyConfig:
    languages: List[str]
    target_language: str
    attribute_every: int = 100
    normalize_loss_by_tokens: bool = True
    activation_layout: str = "SBH"
    include_bias: bool = False
    reduce_tensor_parallel: bool = True
    reduce_pipeline_parallel: bool = True
    reduce_data_parallel: bool = True
    jsonl_flush_every: int = 100


class LanguageInRunShapley:
    """
    Language-level first-order In-Run Data Shapley accumulator.

    The class aggregates per-sequence increments into language buckets. For a
    multilingual pretraining run with 7 FineWeb2 languages, the output is a
    vector of length 7 for each target validation language.
    """

    def __init__(self, model: nn.Module, config: LanguageShapleyConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

        self.lang_to_id = {name: i for i, name in enumerate(config.languages)}
        if config.target_language not in self.lang_to_id:
            raise ValueError(f"target_language={config.target_language} not in languages={config.languages}")

        self.ghost = MegatronLinearGhostDot(
            model,
            GhostHookConfig(
                num_languages=len(config.languages),
                activation_layout=config.activation_layout,
                include_bias=config.include_bias,
            ),
        )
        self.ghost.install()

        n = len(config.languages)
        self.cumulative_scores = torch.zeros(n, dtype=torch.float64)
        self.cumulative_token_counts = torch.zeros(n, dtype=torch.float64)
        self.cumulative_seen_sequences = torch.zeros(n, dtype=torch.float64)
        self.num_attribute_steps = 0

    def close(self) -> None:
        self.ghost.remove()

    def should_attribute(self, iteration: int) -> bool:
        return self.config.attribute_every > 0 and iteration % self.config.attribute_every == 0

    def attribute(
        self,
        train_batch: TensorBatch,
        val_batch: TensorBatch,
        lr: float,
        iteration: int,
        forward_model_with_batch,
    ) -> Dict[str, float]:
        """
        Run one attribution micro-step.

        Args:
            train_batch:
                Megatron-style batch with at least:
                    tokens, labels, loss_mask, position_ids, attention_mask, language_id
                language_id shape: [B], int in [0, num_languages).
            val_batch:
                Same structure for target-language held-out validation data.
                language_id is not required for val_batch.
            lr:
                Current optimizer learning rate.
            iteration:
                Training iteration, for logging only.
            forward_model_with_batch:
                Callable:
                    token_losses = forward_model_with_batch(combined_batch)

                For Megatron GPT this is typically:
                    def forward_model_with_batch(batch):
                        return model(
                            batch["tokens"],
                            batch["position_ids"],
                            batch["attention_mask"],
                            labels=batch["labels"],
                            loss_mask=batch["loss_mask"],
                            packed_seq_params=None,
                        )

        Returns:
            A dict of this-step language scores and token counts after distributed reduction.
        """
        if "language_id" not in train_batch:
            raise KeyError('train_batch must contain "language_id" with shape [micro_batch_size].')

        language_id = train_batch["language_id"].to(self.device).long()
        train_loss_mask = train_batch["loss_mask"].to(self.device)
        B = int(language_id.numel())

        combined = build_combined_gpt_batch(train_batch, val_batch, batch_dim=0)
        combined = {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in combined.items()}

        combined_loss_mask = combined["loss_mask"]

        self.model.zero_grad(set_to_none=True)
        self.ghost.begin(train_batch_size=B, device=self.device)

        token_losses = forward_model_with_batch(combined)
        seq_losses = sequence_loss_from_token_losses(
            token_losses=token_losses,
            loss_mask=combined_loss_mask,
            batch_dim=0,
            normalize_by_tokens=self.config.normalize_loss_by_tokens,
        )

        # Backprop through train + target-validation losses. Hooks split the
        # combined activations and compute train_i dot val_batch.
        seq_losses.sum().backward()

        sample_dots = self.ghost.end()
        sample_increments = -float(lr) * sample_dots

        n = len(self.config.languages)
        lang_scores = torch.zeros(n, device=self.device, dtype=torch.float64)
        lang_token_counts = torch.zeros(n, device=self.device, dtype=torch.float64)
        lang_seen_sequences = torch.zeros(n, device=self.device, dtype=torch.float64)

        token_counts = train_loss_mask.view(B, -1).sum(dim=1).to(self.device).double()

        for l in range(n):
            mask = language_id == l
            if mask.any():
                lang_scores[l] = sample_increments[mask].double().sum()
                lang_token_counts[l] = token_counts[mask].sum()
                lang_seen_sequences[l] = mask.double().sum()

        reduce_scores_for_megatron_(
            lang_scores,
            lang_token_counts,
            reduce_tensor_parallel=self.config.reduce_tensor_parallel,
            reduce_pipeline_parallel=self.config.reduce_pipeline_parallel,
            reduce_data_parallel=self.config.reduce_data_parallel,
        )

        # Sequence counts should only be reduced over DP, not TP/PP.
        if _is_dist_ready() and self.config.reduce_data_parallel:
            _all_reduce_sum_(lang_seen_sequences, group=_safe_group("get_data_parallel_group"))

        self.cumulative_scores += lang_scores.detach().cpu()
        self.cumulative_token_counts += lang_token_counts.detach().cpu()
        self.cumulative_seen_sequences += lang_seen_sequences.detach().cpu()
        self.num_attribute_steps += 1

        self.model.zero_grad(set_to_none=True)

        result = {
            "iteration": int(iteration),
            "target_language": self.config.target_language,
            "lr": float(lr),
            "num_attribute_steps": int(self.num_attribute_steps),
        }
        for i, lang in enumerate(self.config.languages):
            result[f"step_score/{lang}"] = float(lang_scores[i].detach().cpu())
            result[f"cum_score/{lang}"] = float(self.cumulative_scores[i])
            result[f"cum_tokens/{lang}"] = float(self.cumulative_token_counts[i])
            result[f"cum_sequences/{lang}"] = float(self.cumulative_seen_sequences[i])
            result[f"score_per_1m_tokens/{lang}"] = (
                float(self.cumulative_scores[i])
                / max(float(self.cumulative_token_counts[i]), 1.0)
                * 1_000_000.0
            )
            result[f"helpfulness_per_1m_tokens/{lang}"] = (
                -float(self.cumulative_scores[i])
                / max(float(self.cumulative_token_counts[i]), 1.0)
                * 1_000_000.0
            )
        return result

    def summary(self) -> Dict[str, object]:
        rows = []
        for i, lang in enumerate(self.config.languages):
            tokens = float(self.cumulative_token_counts[i])
            score = float(self.cumulative_scores[i])
            rows.append({
                "language": lang,
                "target_language": self.config.target_language,
                "cumulative_inrun_shapley_loss_change": score,
                "cumulative_tokens": tokens,
                "cumulative_sequences": float(self.cumulative_seen_sequences[i]),
                "score_per_1m_tokens": score / max(tokens, 1.0) * 1_000_000.0,
                "helpfulness_per_1m_tokens": -score / max(tokens, 1.0) * 1_000_000.0,
            })
        rows.sort(key=lambda r: r["cumulative_inrun_shapley_loss_change"])
        return {
            "config": asdict(self.config),
            "num_attribute_steps": self.num_attribute_steps,
            "rows": rows,
        }

    def write_jsonl(self, path: str | Path, record: Mapping[str, object]) -> None:
        if not _is_global_rank0():
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"time": time.time(), **dict(record)}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def write_summary_json(self, path: str | Path) -> None:
        if not _is_global_rank0():
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.summary(), f, ensure_ascii=False, indent=2)


MEGATRON_INTEGRATION_EXAMPLE = """
# === 1) Add to your modified pretrain_gpt.py imports ===
from megatron_inrun_language_shapley import (
    LanguageInRunShapley,
    LanguageShapleyConfig,
)

# === 2) Build after model is created ===
languages = ["eng_Latn", "deu_Latn", "fra_Latn", "spa_Latn", "ita_Latn", "por_Latn", "zho_Hans"]
shapley = LanguageInRunShapley(
    model=model[0] if isinstance(model, list) else model,
    config=LanguageShapleyConfig(
        languages=languages,
        target_language="zho_Hans",
        attribute_every=args.shapley_attribute_every,
        activation_layout="SBH",
        include_bias=False,
    ),
    device=torch.device("cuda", torch.cuda.current_device()),
)

# === 3) Define the Megatron forward adapter ===
def forward_model_with_batch(batch):
    return model(
        batch["tokens"],
        batch["position_ids"],
        batch["attention_mask"],
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
        packed_seq_params=None,
    )

# === 4) In your training loop, every N iterations ===
# train_batch must include language_id: shape [micro_batch_size].
# val_batch comes from target-language held-out validation iterator.
if shapley.should_attribute(iteration):
    lr = optimizer.param_groups[0]["lr"]
    record = shapley.attribute(
        train_batch=train_batch,
        val_batch=next(target_language_valid_iterator),
        lr=lr,
        iteration=iteration,
        forward_model_with_batch=forward_model_with_batch,
    )
    shapley.write_jsonl(args.shapley_output_jsonl, record)

# === 5) At checkpoint/end ===
shapley.write_summary_json(args.shapley_summary_json)
"""


if __name__ == "__main__":
    print("This file is a Megatron-LM helper module, not a standalone trainer.")
    print("Copy it into your Megatron-LM root and import LanguageInRunShapley.")
    print(MEGATRON_INTEGRATION_EXAMPLE)
