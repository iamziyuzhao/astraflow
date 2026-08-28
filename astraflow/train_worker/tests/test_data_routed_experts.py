"""Tests for ND per-token tensors (R3 ``routed_experts``) in the data helpers.

R3 (Rollout Routing Replay) attaches an int16 tensor of shape
``[B, S, num_moe_layers, top_k]`` to every training batch. The helpers in
``astraflow/train_worker/utils/data.py`` historically selected per-token
tensors with numel-based predicates (``numel == bs * max_seqlen`` /
``numel == total_length``), which silently mishandled tensors with trailing
dims. These tests pin the shape-based behavior:

1. ``split_padded_tensor_dict_into_mb_list`` splits ``[B, S, L, K]`` tensors
   by row alongside ``input_ids`` instead of duplicating them into every
   micro-batch.
2. ``pack_tensor_dict`` -> ``pad_packed_tensor_dict`` ->
   ``unsqueeze_packed_tensor_dict`` preserves per-token ``[L, K]`` rows
   aligned to packed positions.
3. qwen2.5-vl-style ``position_ids`` still takes its special path
   (regression guard for the predicate change).
"""

from __future__ import annotations

import torch

from astraflow.train_worker.api.cli_args import MicroBatchSpec
from astraflow.train_worker.utils.data import (
    concat_padded_tensors,
    pack_tensor_dict,
    pad_packed_tensor_dict,
    split_padded_tensor_dict_into_mb_list,
    unsqueeze_packed_tensor_dict,
)

NUM_MOE_LAYERS = 4
TOP_K = 2
# Distinct value per (layer, expert-slot) cell so misaligned copies are caught.
CELL_OFFSETS = torch.arange(NUM_MOE_LAYERS * TOP_K, dtype=torch.int16).reshape(
    NUM_MOE_LAYERS, TOP_K
)


def _expected_experts(token_ids: torch.Tensor) -> torch.Tensor:
    """Expected routed_experts rows for the given token ids (any leading shape)."""
    base = token_ids.to(torch.int16) * 10
    return base.reshape(*token_ids.shape, 1, 1) + CELL_OFFSETS


def _make_padded_batch(
    lens: list[int], vl_position_ids: bool = False
) -> dict[str, torch.Tensor]:
    """Build a padded batch whose routed_experts rows are derived from input_ids."""
    bs = len(lens)
    max_seqlen = max(lens)
    attention_mask = torch.zeros(bs, max_seqlen, dtype=torch.long)
    # Unique nonzero token per valid (b, s) position.
    input_ids = torch.zeros(bs, max_seqlen, dtype=torch.long)
    for b, seq_len in enumerate(lens):
        attention_mask[b, :seq_len] = 1
        input_ids[b, :seq_len] = 7 + 100 * b + torch.arange(seq_len)
    routed_experts = _expected_experts(input_ids)
    routed_experts[attention_mask == 0] = 0
    data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "routed_experts": routed_experts,
        "rewards": torch.arange(bs, dtype=torch.float32),
    }
    if vl_position_ids:
        data["position_ids"] = (
            torch.arange(max_seqlen).view(1, max_seqlen, 1).expand(bs, max_seqlen, 3)
            + torch.arange(bs).view(bs, 1, 1) * 1000
        ).contiguous()
    return data


def _assert_rows_aligned(input_ids: torch.Tensor, routed_experts: torch.Tensor):
    """Every position with a nonzero token carries its derived expert rows;
    pad positions carry all-zero rows."""
    valid = input_ids != 0
    expected = _expected_experts(input_ids)
    assert torch.equal(routed_experts[valid], expected[valid])
    assert (routed_experts[~valid] == 0).all()


class TestSplitPaddedTensorDict:
    def test_routed_experts_split_by_row(self):
        lens = [8, 6, 4, 2]
        data = _make_padded_batch(lens)
        mb_spec = MicroBatchSpec(n_mbs=2, max_tokens_per_mb=10)
        mb_list = split_padded_tensor_dict_into_mb_list(data, mb_spec)

        assert len(mb_list.mbs) >= 2
        total_rows = 0
        for mb in mb_list.mbs:
            experts = mb["routed_experts"]
            n_rows = mb["input_ids"].shape[0]
            # Split by row alongside input_ids, not duplicated into every mb.
            assert experts.shape == (n_rows, max(lens), NUM_MOE_LAYERS, TOP_K)
            assert experts.dtype == torch.int16
            _assert_rows_aligned(mb["input_ids"], experts)
            # 1D per-sequence tensors keep the historical not-to-split path.
            assert mb["rewards"].shape == (len(lens),)
            total_rows += n_rows
        assert total_rows == len(lens)
        assert sorted(mb_list.forward_indices) == list(range(len(lens)))

    def test_qwen25_vl_position_ids_special_path(self):
        # qwen2.5-vl position_ids: [bs, max_seqlen, 3],
        # numel == bs * max_seqlen * 3.
        lens = [6, 4, 2, 2]
        data = _make_padded_batch(lens, vl_position_ids=True)
        mb_spec = MicroBatchSpec(n_mbs=2, max_tokens_per_mb=8)
        mb_list = split_padded_tensor_dict_into_mb_list(data, mb_spec)

        assert len(mb_list.mbs) >= 2
        total_rows = 0
        for mb in mb_list.mbs:
            pos = mb["position_ids"]
            n_rows = mb["input_ids"].shape[0]
            assert pos.shape == (n_rows, max(lens), 3)
            for i in range(n_rows):
                b = (pos[i, 0, 0] // 1000).item()
                # Reordered together with input_ids (row 0 of seq b is 7+100*b).
                assert mb["input_ids"][i, 0].item() == 7 + 100 * b
            total_rows += n_rows
        assert total_rows == len(lens)


class TestPackedRoundTrip:
    def test_pack_pad_unsqueeze_preserves_rows(self):
        lens = [6, 4, 2]
        total_length = sum(lens)
        data = _make_padded_batch(lens)
        data.pop("rewards")

        packed = pack_tensor_dict(data)
        experts = packed["routed_experts"]
        assert experts.shape == (total_length, NUM_MOE_LAYERS, TOP_K)
        _assert_rows_aligned(packed["input_ids"], experts)
        cu_seqlens = packed["cu_seqlens"]
        for b, seq_len in enumerate(lens):
            start = cu_seqlens[b].item()
            assert torch.equal(
                experts[start : start + seq_len],
                _expected_experts(data["input_ids"][b, :seq_len]),
            )

        pad_to_length = 16
        padded, pad_len, _, _ = pad_packed_tensor_dict(packed, pad_to_length)
        assert pad_len == pad_to_length - total_length
        experts = padded["routed_experts"]
        assert experts.shape == (pad_to_length, NUM_MOE_LAYERS, TOP_K)
        assert padded["cu_seqlens"][-1].item() == pad_to_length
        _assert_rows_aligned(padded["input_ids"], experts)
        assert (experts[total_length:] == 0).all()

        unsqueezed = unsqueeze_packed_tensor_dict(padded)
        assert unsqueezed["routed_experts"].shape == (
            1,
            pad_to_length,
            NUM_MOE_LAYERS,
            TOP_K,
        )
        assert unsqueezed["input_ids"].shape == (1, pad_to_length)
        assert torch.equal(unsqueezed["cu_seqlens"], padded["cu_seqlens"])
        _assert_rows_aligned(
            unsqueezed["input_ids"][0], unsqueezed["routed_experts"][0]
        )

    def test_pad_packed_align_sequences_preserves_rows(self):
        lens = [6, 4, 2]
        data = _make_padded_batch(lens)
        data.pop("rewards")
        packed = pack_tensor_dict(data)

        padded, _, old_cu_seqlens, align_to_length = pad_packed_tensor_dict(
            packed, pad_to_length=16, align_sequences=True, align_to_multiple_of=4
        )
        assert torch.equal(old_cu_seqlens, packed["cu_seqlens"])
        # lens 6, 4, 2 -> aligned lens 8, 4, 4.
        assert align_to_length == 16
        experts = padded["routed_experts"]
        assert experts.shape[1:] == (NUM_MOE_LAYERS, TOP_K)
        assert experts.shape[0] == padded["cu_seqlens"][-1].item()
        _assert_rows_aligned(padded["input_ids"], experts)
        # Each sequence's rows land at the aligned offsets.
        aligned_starts = [0, 8, 12]
        for b, (seq_len, start) in enumerate(zip(lens, aligned_starts)):
            assert torch.equal(
                experts[start : start + seq_len],
                _expected_experts(data["input_ids"][b, :seq_len]),
            )


class TestConcatPaddedTensors:
    def test_mixed_length_nd_concat(self):
        short, long = 3, 5
        a = _make_padded_batch([long])
        b = _make_padded_batch([short])
        out = concat_padded_tensors([a, b])
        assert out["routed_experts"].shape == (2, long, NUM_MOE_LAYERS, TOP_K)
        _assert_rows_aligned(out["input_ids"], out["routed_experts"])
        assert (out["attention_mask"][1, short:] == 0).all()
        assert (out["routed_experts"][1, short:] == 0).all()
