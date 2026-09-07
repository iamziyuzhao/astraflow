"""CP zigzag / SP slicing of packed routed_experts (CPU, pure-torch math).

Validates that `split_packed_tensor_context_parallel` — the trailing-dims
generalization used both by `preprocess_packed_seqs_context_parallel` (for
input_ids) and by the R3 routed_experts path — slices a
[total_tokens, num_moe_layers, top_k] tensor exactly the way input_ids are
sliced, so replayed routing rows stay aligned with the tokens each rank
forwards. The megatron import inside the module under test is guarded with
importorskip; no process groups are needed (cp/tp sizes are explicit args).
"""

import pytest
import torch

pytest.importorskip("megatron.core")

from astraflow.train_worker.utils.mcore.packed_context_parallel import (  # noqa: E402
    sequence_parallel_chunk,
    split_packed_tensor_context_parallel,
)

NUM_MOE_LAYERS = 4
TOP_K = 2


def _packed_batch(cp_size: int, tp_size: int):
    # Sequence lengths aligned to tp_size * cp_size * 2, as the trainer pads.
    align = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    seqlens = [2 * align, 4 * align, 2 * align]
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seqlens), dim=0)), dtype=torch.long
    )
    total = int(cu_seqlens[-1])
    # input_ids == arange, so input_ids double as row indices into any packed
    # tensor: routed[t] belongs to token input_ids[t] by construction.
    input_ids = torch.arange(total, dtype=torch.long)
    routed = (
        input_ids.view(-1, 1, 1) * 100
        + torch.arange(NUM_MOE_LAYERS).view(1, -1, 1) * 10
        + torch.arange(TOP_K).view(1, 1, -1)
    ).to(torch.int16)
    return input_ids, routed, cu_seqlens


def test_cp1_is_identity():
    input_ids, routed, cu_seqlens = _packed_batch(cp_size=1, tp_size=1)
    assert torch.equal(
        split_packed_tensor_context_parallel(routed, cu_seqlens, 1, 0), routed
    )


@pytest.mark.parametrize("cp_size", [2, 4])
def test_trailing_dims_follow_input_ids_split(cp_size):
    input_ids, routed, cu_seqlens = _packed_batch(cp_size=cp_size, tp_size=1)
    covered = []
    for cp_rank in range(cp_size):
        ids_split = split_packed_tensor_context_parallel(
            input_ids, cu_seqlens, cp_size, cp_rank
        )
        routed_split = split_packed_tensor_context_parallel(
            routed, cu_seqlens, cp_size, cp_rank
        )
        assert ids_split.shape[0] == input_ids.shape[0] // cp_size
        assert routed_split.shape == (
            input_ids.shape[0] // cp_size,
            NUM_MOE_LAYERS,
            TOP_K,
        )
        # Row t of the split routed tensor is the record of the token that
        # ended up at position t of the split input_ids.
        assert torch.equal(routed_split, routed[ids_split])
        covered.append(ids_split)
    # Zigzag chunks partition the packed tokens exactly once across ranks.
    assert torch.equal(
        torch.cat(covered).sort().values, torch.arange(input_ids.shape[0])
    )


def test_zigzag_chunk_layout():
    # Rank r keeps chunks r and 2*cp-1-r of each sequence (load balancing for
    # causal attention).
    cp_size = 2
    input_ids, _, cu_seqlens = _packed_batch(cp_size=cp_size, tp_size=1)
    seqlen0 = int(cu_seqlens[1])
    quarter = seqlen0 // (2 * cp_size)
    rank0 = split_packed_tensor_context_parallel(input_ids, cu_seqlens, cp_size, 0)
    rank1 = split_packed_tensor_context_parallel(input_ids, cu_seqlens, cp_size, 1)
    first_seq_rank0 = rank0[: seqlen0 // cp_size]
    first_seq_rank1 = rank1[: seqlen0 // cp_size]
    assert torch.equal(
        first_seq_rank0,
        torch.cat([input_ids[:quarter], input_ids[3 * quarter : 4 * quarter]]),
    )
    assert torch.equal(
        first_seq_rank1,
        torch.cat(
            [input_ids[quarter : 2 * quarter], input_ids[2 * quarter : 3 * quarter]]
        ),
    )


def test_sequence_parallel_chunk_matches_input_ids_chunk():
    tp_size = 2
    input_ids, routed, _ = _packed_batch(cp_size=1, tp_size=tp_size)
    chunks = []
    for tp_rank in range(tp_size):
        ids_chunk = sequence_parallel_chunk(input_ids, tp_size, tp_rank)
        routed_chunk = sequence_parallel_chunk(routed, tp_size, tp_rank)
        assert torch.equal(routed_chunk, routed[ids_chunk])
        chunks.append(ids_chunk)
    assert torch.equal(torch.cat(chunks), input_ids)

    with pytest.raises(ValueError, match="not divisible"):
        sequence_parallel_chunk(input_ids[:-1], tp_size, 0)


@pytest.mark.parametrize("cp_size,tp_size", [(2, 2), (4, 2)])
def test_combined_cp_then_sp_alignment(cp_size, tp_size):
    input_ids, routed, cu_seqlens = _packed_batch(cp_size=cp_size, tp_size=tp_size)
    for cp_rank in range(cp_size):
        ids_local = split_packed_tensor_context_parallel(
            input_ids, cu_seqlens, cp_size, cp_rank
        )
        routed_local = split_packed_tensor_context_parallel(
            routed, cu_seqlens, cp_size, cp_rank
        )
        for tp_rank in range(tp_size):
            ids_view = sequence_parallel_chunk(ids_local, tp_size, tp_rank)
            routed_view = sequence_parallel_chunk(routed_local, tp_size, tp_rank)
            assert torch.equal(routed_view, routed[ids_view])
