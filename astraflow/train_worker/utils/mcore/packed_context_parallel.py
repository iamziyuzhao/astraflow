from typing import Any

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_model_config

from astraflow.train_worker.utils.mcore.routing_replay import (
    get_replay_chunk_index,
    get_replay_context,
)


def split_packed_tensor_context_parallel(
    values: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    """Zigzag context-parallel split of a packed tensor along its token axis.

    Generalizes the split performed by `preprocess_packed_seqs_context_parallel`
    to tensors with arbitrary trailing dims (e.g. routed_experts of shape
    [total_tokens, num_moe_layers, top_k]): each sequence is cut into CP*2
    chunks and rank r keeps chunks r and 2*CP-1-r.
    """
    if cp_size <= 1:
        return values
    input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    batch_size = input_lens.shape[0]

    shape = (int(input_lens.sum().item()) // cp_size, *values.shape[1:])
    splitted = torch.zeros(shape, dtype=values.dtype, device=values.device)
    for i in range(batch_size):
        seqlen = input_lens[i] // cp_size
        half_seqlen = seqlen // 2
        start_idx = cu_seqlens[i] // cp_size
        # split to 2 chunks
        d = values[cu_seqlens[i] : cu_seqlens[i + 1]]
        splitted[start_idx : start_idx + half_seqlen] = d[
            half_seqlen * cp_rank : half_seqlen * (cp_rank + 1)
        ]

        remain_start = input_lens[i] - half_seqlen * (cp_rank + 1)
        remain_end = input_lens[i] - half_seqlen * cp_rank
        remain_end = min(remain_end, d.shape[0])
        remain_len = remain_end - remain_start
        splitted[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d[
            remain_start:remain_end
        ]
    return splitted


def sequence_parallel_chunk(
    values: torch.Tensor,
    tp_size: int,
    tp_rank: int,
) -> torch.Tensor:
    """Contiguous per-TP-rank chunk along the token axis.

    Mirrors `scatter_to_sequence_parallel_region`: with sequence parallelism
    the transformer layers (and thus the MoE routers) only see the local
    contiguous 1/tp_size chunk of the packed sequence.
    """
    total_tokens = values.shape[0]
    if total_tokens % tp_size != 0:
        raise ValueError(
            f"Packed token count {total_tokens} is not divisible by "
            f"tensor parallel size {tp_size} for sequence parallelism."
        )
    chunk_len = total_tokens // tp_size
    return values[tp_rank * chunk_len : (tp_rank + 1) * chunk_len]


def preprocess_packed_seqs_context_parallel(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, PackedSeqParams]:
    """
    Preprocess packed sequences.
    CP splits sequence into CP*2 chunks, and each GPU gets 2 chunks (GPU0 gets first and last chunks, GPU1 gets second and second last chunks, and so on),
    this is for load balancing with causal masking. See https://github.com/NVIDIA/TransformerEngine/issues/1368
    """
    input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = input_lens.max().item()

    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    align_to_multiple_of = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    # assume input_ids and cu_seqlens are already padded to align_to_multiple_of
    if any(length % align_to_multiple_of for length in input_lens) != 0:
        raise ValueError(
            f"Some of the input sequence length ({input_lens}) is not a multiple of align_to_multiple_of {align_to_multiple_of} "
            "for context/sequence parallel in Megatron."
        )

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=max_seqlen,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_kv=max_seqlen,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
    )

    if cp_size <= 1:
        return input_ids.unsqueeze(0), packed_seq_params

    splitted = split_packed_tensor_context_parallel(
        input_ids, cu_seqlens, cp_size, cp_rank
    )
    return splitted.unsqueeze(0), packed_seq_params


def postprocess_packed_seqs_context_parallel(
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    post_process: bool,
) -> torch.Tensor:
    """
    Postprocess packed sequences
    """
    cp_size = mpu.get_context_parallel_world_size()
    if not post_process:
        return output
    if cp_size <= 1:
        return output.squeeze(0)
    # shape = [batch_size, seq_len] + list(output.shape[2:])
    # [1, packed, dim] -> [batch_size, seq_len, dim]
    batch_size = cu_seqlens.shape[0] - 1
    output_len = int(cu_seqlens[-1].item())
    # output shape: [total_packed_seq_len] + list(output.shape[2:]
    output_new = torch.empty(
        (output_len, *output.shape[2:]), device=output.device, dtype=output.dtype
    )
    # all gather output across context parallel group
    # need to gather across cp group and concatenate in sequence dimension
    output_list = [torch.empty_like(output) for _ in range(cp_size)]
    dist.all_gather(
        output_list, output.detach(), group=mpu.get_context_parallel_group()
    )
    output_list[mpu.get_context_parallel_rank()] = output

    for i in range(batch_size):
        seq_len = cu_seqlens[i + 1] - cu_seqlens[i]
        splitted_seq_len = (cu_seqlens[i + 1] - cu_seqlens[i]) // cp_size
        half_splitted_seq_len = splitted_seq_len // 2

        tmp = torch.empty(
            (seq_len, *output.shape[2:]), device=output.device, dtype=output.dtype
        )
        for j in range(cp_size):
            o = output_list[j].squeeze(0)
            # split to 2 chunks
            start = cu_seqlens[i] // cp_size
            o0, o1 = (
                o[start : start + half_splitted_seq_len],
                o[start + half_splitted_seq_len : start + splitted_seq_len],
            )
            tmp[j * half_splitted_seq_len : (j + 1) * half_splitted_seq_len] = o0
            splitted_start = seq_len - (j + 1) * half_splitted_seq_len
            splitted_end = seq_len - j * half_splitted_seq_len
            tmp[splitted_start:splitted_end] = o1

        output_new[cu_seqlens[i] : cu_seqlens[i + 1]] = tmp[:seq_len]
    return output_new


def _split_routed_experts_for_model_parallel(
    model: torch.nn.Module,
    routed_experts: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Slice packed routed_experts down to the rows this rank's routers see."""
    routed_experts = split_packed_tensor_context_parallel(
        routed_experts,
        cu_seqlens,
        mpu.get_context_parallel_world_size(),
        mpu.get_context_parallel_rank(),
    )
    tp_size = mpu.get_tensor_model_parallel_world_size()
    if tp_size > 1 and get_model_config(model).sequence_parallel:
        routed_experts = sequence_parallel_chunk(
            routed_experts, tp_size, mpu.get_tensor_model_parallel_rank()
        )
    return routed_experts


def packed_context_parallel_forward(
    model: torch.nn.Module,
    input_: dict[str, Any],
):
    input_ids = input_["input_ids"]
    cu_seqlens = input_["cu_seqlens"]
    position_ids = input_["position_ids"]
    # NOTE: read, never pop. With virtual pipeline parallelism megatron hands
    # the *same* micro-batch dict to every model chunk's forward, and each
    # chunk must install the records for the layers it hosts right before its
    # own forward consumes them (install/consume lockstep). Popping would both
    # restrict the install to chunk 0 -- appending records for layers hosted on
    # later chunks that those chunks' forwards do not consume in step, which
    # silently serves the wrong micro-batch's routing under activation
    # recompute -- and permanently mutate mb_list.padded_mbs, breaking a second
    # forward_backward_batch over the same micro-batch list.
    routed_experts = input_.get("routed_experts")
    input_ids_rmpad, packed_seq_params = preprocess_packed_seqs_context_parallel(
        input_ids, cu_seqlens
    )
    input_ids_rmpad = input_ids_rmpad.contiguous()
    replay_context = get_replay_context()
    if (
        replay_context is not None
        and replay_context.is_armed
        and routed_experts is not None
    ):
        replay_context.install_packed(
            _split_routed_experts_for_model_parallel(model, routed_experts, cu_seqlens),
            chunk_index=get_replay_chunk_index(model),
        )
    try:
        output_orig = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
        )
    except Exception as e:
        raise RuntimeError(
            f"Error occurred in packed context parallel forward pass on model {model} "
            f"with input_ids shape {input_ids_rmpad.shape} and packed_seq_params {packed_seq_params}."
        ) from e

    model_vp_stage = getattr(model, "vp_stage", None)
    is_pipeline_last_stage = mpu.is_pipeline_last_stage(
        ignore_virtual=False, vp_stage=model_vp_stage
    )
    output = postprocess_packed_seqs_context_parallel(
        output_orig, cu_seqlens, is_pipeline_last_stage
    )
    return output
