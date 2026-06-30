# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT with optional language-level attribution.

This public version has been anonymized to avoid exposing user-, host-,
or environment-specific information. Runtime-specific values should be
provided through command-line arguments or environment variables.
"""

import torch
from functools import partial
from typing import List, Optional, Tuple, Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

import os

from megatron_inrun_language_shapley import (
    LanguageInRunShapley,
    LanguageShapleyConfig,
)

# ---------------------------------------------------------------------
# In-Run Data Shapley global state
# ---------------------------------------------------------------------

_SHAPLEY = None
# _TARGET_VALID_ITERATOR = None
_SHAPLEY_INITIALIZED = False


stimer = StragglerDetector()

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump
            dump(snapshot, open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'))

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te, normalization=args.normalization)
            elif args.heterogeneous_layers_config_path is not None:
                transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm,
                        normalization=args.normalization)
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine=use_te)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
            z_loss_strength=args.z_loss_strength,
        )

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # Metadata for In-Run Data Shapley.
    # Do not pass these into the model forward.
    language_id = batch.pop("language_id", None)
    dataset_id = batch.pop("dataset_id", None)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        batch["attention_mask"],
        batch["position_ids"],
        language_id,
    )


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    # loss[0] is a view of loss, so it has ._base not None, which triggers assert error
    # in core/pipeline_parallel/schedule.py::deallocate_output_tensor, calling .clone()
    # on loss[0] fixes this
    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0].clone(),
        local_num_tokens,
        {'lm loss': (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids, language_id = get_batch(data_iterator)
    timers('batch-generator').stop()

    # Keep the current training micro-batch for In-Run Data Shapley.
    # language_id is metadata only and must not be passed into the model.
    args._latest_train_batch_for_shapley = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "language_id": language_id,
    }

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(
                tokens,
                position_ids,
                attention_mask,
                labels=labels,
            )
        else:
            output_tensor = model(
                tokens,
                position_ids,
                attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )

    return output_tensor, partial(loss_func, loss_mask)


def _unwrap_model(model):
    """Megatron sometimes passes model as a one-element list."""
    return model[0] if isinstance(model, list) else model


def _get_current_lr(optimizer) -> float:
    """Best-effort extraction of the current learning rate from Megatron optimizer wrappers."""
    if optimizer is None:
        return float(os.environ.get("MEGATRON_INRUN_SHAPLEY_FALLBACK_LR", "0.0"))

    if hasattr(optimizer, "param_groups") and optimizer.param_groups:
        return float(optimizer.param_groups[0].get("lr", 0.0))

    inner = getattr(optimizer, "optimizer", None)
    if inner is not None and hasattr(inner, "param_groups") and inner.param_groups:
        return float(inner.param_groups[0].get("lr", 0.0))

    optimizers = getattr(optimizer, "optimizers", None)
    if optimizers:
        for opt in optimizers:
            if hasattr(opt, "param_groups") and opt.param_groups:
                return float(opt.param_groups[0].get("lr", 0.0))
            inner = getattr(opt, "optimizer", None)
            if inner is not None and hasattr(inner, "param_groups") and inner.param_groups:
                return float(inner.param_groups[0].get("lr", 0.0))

    return float(os.environ.get("MEGATRON_INRUN_SHAPLEY_FALLBACK_LR", "0.0"))


def maybe_init_inrun_shapley(model):
    """Initialize language-level In-Run Data Shapley once per process."""
    global _SHAPLEY, _SHAPLEY_INITIALIZED

    if _SHAPLEY_INITIALIZED:
        return _SHAPLEY

    enabled = os.environ.get("MEGATRON_INRUN_SHAPLEY_ENABLE", "0")
    if enabled != "1":
        _SHAPLEY_INITIALIZED = True
        _SHAPLEY = None
        print_rank_0("[InRunShapley] disabled.")
        return None

    args = get_args()

    if getattr(args, "pipeline_model_parallel_size", 1) != 1:
        print_rank_0(
            "[InRunShapley] WARNING: pipeline_model_parallel_size > 1 detected. "
            "This first integration is intended for PP=1. Attribution may be skipped or incorrect."
        )

    languages = [
        x.strip()
        for x in os.environ.get("MEGATRON_INRUN_SHAPLEY_LANGUAGES", "lang_a,lang_b,lang_c").split(",")
        if x.strip()
    ]
    target_language = os.environ.get("MEGATRON_INRUN_SHAPLEY_TARGET_LANGUAGE", "lang_a")
    attribute_every = int(os.environ.get("MEGATRON_INRUN_SHAPLEY_ATTRIBUTE_EVERY", "100"))
    activation_layout = os.environ.get("MEGATRON_INRUN_SHAPLEY_ACTIVATION_LAYOUT", "SBH")

    wrapped_model = _unwrap_model(model)

    _SHAPLEY = LanguageInRunShapley(
        model=wrapped_model,
        config=LanguageShapleyConfig(
            languages=languages,
            target_language=target_language,
            attribute_every=attribute_every,
            activation_layout=activation_layout,
            include_bias=False,
        ),
        device=torch.device("cuda", torch.cuda.current_device()),
    )

    _SHAPLEY_INITIALIZED = True

    print_rank_0(
        f"[InRunShapley] initialized: target={target_language}, "
        f"languages={languages}, attribute_every={attribute_every}, "
        f"activation_layout={activation_layout}"
    )

    return _SHAPLEY


def inrun_shapley_forward_model_with_batch(model):
    """Adapter used by LanguageInRunShapley.attribute()."""
    wrapped_model = _unwrap_model(model)

    def _forward(batch):
        args = get_args()

        if args.use_legacy_models:
            return wrapped_model(
                batch["tokens"],
                batch["position_ids"],
                batch["attention_mask"],
                labels=batch["labels"],
            )

        return wrapped_model(
            batch["tokens"],
            batch["position_ids"],
            batch["attention_mask"],
            labels=batch["labels"],
            loss_mask=batch["loss_mask"],
        )

    return _forward


def _build_validation_batch_for_shapley(valid_data_iterator):
    """Fetch one Megatron validation micro-batch and convert it to a Shapley batch dict."""
    if valid_data_iterator is None:
        return None

    # Reuse this file's get_batch() so TP/CP handling matches training.
    tokens, labels, loss_mask, attention_mask, position_ids, _language_id = get_batch(valid_data_iterator)

    if tokens is None:
        return None

    return {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def maybe_run_inrun_shapley(iteration, model, optimizer, valid_data_iterator):
    """Run one language-level In-Run Data Shapley attribution step if scheduled.

    This function is intentionally kept outside forward_step(). It should be called
    from Megatron's training loop after a normal training iteration is complete.

    Required training.py hook, after iteration increments:
        try:
            from <this_script_module> import maybe_run_inrun_shapley
            maybe_run_inrun_shapley(iteration, model, optimizer, valid_data_iterator)
        except Exception as e:
            print_rank_0(f"[InRunShapley] hook failed: {e}")
    """
    shapley = maybe_init_inrun_shapley(model)
    if shapley is None:
        return

    if not shapley.should_attribute(int(iteration)):
        return

    args = get_args()

    train_batch = getattr(args, "_latest_train_batch_for_shapley", None)
    if train_batch is None:
        print_rank_0(f"[InRunShapley] skipped at iteration={iteration}: no cached train batch.")
        return

    if train_batch.get("language_id", None) is None:
        print_rank_0(f"[InRunShapley] skipped at iteration={iteration}: language_id is None.")
        return

    if valid_data_iterator is None:
        print_rank_0(f"[InRunShapley] skipped at iteration={iteration}: valid_data_iterator is None.")
        return

    try:
        val_batch = _build_validation_batch_for_shapley(valid_data_iterator)
    except StopIteration:
        print_rank_0(f"[InRunShapley] skipped at iteration={iteration}: valid iterator exhausted.")
        return

    if val_batch is None:
        print_rank_0(f"[InRunShapley] skipped at iteration={iteration}: validation batch is None.")
        return

    lr = _get_current_lr(optimizer)
    if lr == 0.0:
        print_rank_0(
            f"[InRunShapley] WARNING at iteration={iteration}: current lr resolved to 0.0. "
            "Set MEGATRON_INRUN_SHAPLEY_FALLBACK_LR if this is unexpected."
        )

    try:
        record = shapley.attribute(
            train_batch=train_batch,
            val_batch=val_batch,
            lr=lr,
            iteration=int(iteration),
            forward_model_with_batch=inrun_shapley_forward_model_with_batch(model),
        )
    except Exception as e:
        print_rank_0(f"[InRunShapley] ERROR at iteration={iteration}: {type(e).__name__}: {e}")
        raise

    output_jsonl = os.environ.get("MEGATRON_INRUN_SHAPLEY_OUTPUT_JSONL", "")
    if output_jsonl:
        shapley.write_jsonl(output_jsonl, record)

    summary_json = os.environ.get("MEGATRON_INRUN_SHAPLEY_SUMMARY_JSON", "")
    if summary_json:
        shapley.write_summary_json(summary_json)

    # Print a compact rank-0 progress line. Full values are written to JSONL.
    print_rank_0(
        f"[InRunShapley] attribution written at iteration={iteration}, "
        f"target={record.get('target_language')}, lr={lr:.6e}"
    )


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path=args.s3_cache_path,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
