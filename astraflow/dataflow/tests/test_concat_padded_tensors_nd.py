"""ND trailing-dim safety of concat_padded_tensors for R3 routed_experts.

Routed-expert tensors are ``[1, seq_len, num_moe_layers, top_k]`` — the
padding path must preserve the trailing dims instead of assuming 2D.
"""

from __future__ import annotations

import torch

from astraflow.dataflow.rollout_buffer import _slice_tensor_dict
from astraflow.dataflow.utils import concat_padded_tensors

NUM_MOE_LAYERS = 4
TOP_K = 2


def _routed_experts_for(input_ids: torch.Tensor) -> torch.Tensor:
    """Derive routed_experts rows deterministically from input_ids values.

    Row ``t`` encodes ``input_ids[0, t]``, so per-position alignment can be
    re-verified after any concat / slice round trip.
    """
    seq_len = input_ids.shape[1]
    base = input_ids[0].to(torch.int16).reshape(seq_len, 1, 1)
    offsets = torch.arange(NUM_MOE_LAYERS * TOP_K, dtype=torch.int16).reshape(
        NUM_MOE_LAYERS, TOP_K
    )
    return (base * 10 + offsets).unsqueeze(0)


def _make_example(token_ids: list[int]) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([token_ids], dtype=torch.int32)
    seq_len = len(token_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones((1, seq_len), dtype=torch.bool),
        "logprobs": -0.5 * torch.ones((1, seq_len), dtype=torch.float32),
        "routed_experts": _routed_experts_for(input_ids),
        "rewards": torch.tensor([1.0], dtype=torch.float32),
    }


def _assert_positions_aligned(batch: dict[str, torch.Tensor], lengths: list[int]):
    """Every unpadded position's routed_experts row must match its token id."""
    for b, length in enumerate(lengths):
        expected = _routed_experts_for(batch["input_ids"][b : b + 1, :length])
        assert torch.equal(batch["routed_experts"][b : b + 1, :length], expected)


def test_concat_pads_nd_trailing_dims():
    lengths = [3, 5, 2]
    examples = [
        _make_example([11, 12, 13]),
        _make_example([21, 22, 23, 24, 25]),
        _make_example([31, 32]),
    ]
    batch = concat_padded_tensors(examples)

    assert batch["input_ids"].shape == (3, 5)
    assert batch["routed_experts"].shape == (3, 5, NUM_MOE_LAYERS, TOP_K)
    assert batch["routed_experts"].dtype == torch.int16
    for b, length in enumerate(lengths):
        assert bool(batch["attention_mask"][b, :length].all())
        assert not bool(batch["attention_mask"][b, length:].any())
        # Padded routed_experts positions carry the pad value (0).
        assert torch.all(batch["routed_experts"][b, length:] == 0)
    _assert_positions_aligned(batch, lengths)


def test_concat_nd_respects_pad_value():
    examples = [_make_example([11, 12]), _make_example([21, 22, 23])]
    batch = concat_padded_tensors(examples, pad_value=-1.0)
    assert torch.all(batch["routed_experts"][0, 2:] == -1)
    assert torch.all(batch["logprobs"][0, 2:] == -1.0)
    # attention_mask always pads with zeros regardless of pad_value.
    assert not bool(batch["attention_mask"][0, 2:].any())


def test_concat_slice_reconcat_round_trip():
    lengths = [3, 5, 2]
    examples = [
        _make_example([11, 12, 13]),
        _make_example([21, 22, 23, 24, 25]),
        _make_example([31, 32]),
    ]
    batch = concat_padded_tensors(examples)

    slices = [_slice_tensor_dict(batch, i, i + 1) for i in range(len(examples))]
    for i, sliced in enumerate(slices):
        assert sliced["routed_experts"].shape == (1, 5, NUM_MOE_LAYERS, TOP_K)
        assert sliced["routed_experts"].dtype == torch.int16
        assert torch.equal(sliced["input_ids"], batch["input_ids"][i : i + 1])

    rebatch = concat_padded_tensors(slices)
    for key in ("input_ids", "attention_mask", "logprobs", "routed_experts", "rewards"):
        assert torch.equal(rebatch[key], batch[key]), key
    _assert_positions_aligned(rebatch, lengths)


def test_reconcat_with_mixed_max_lengths():
    # Two independently padded sub-batches with different max lengths must
    # merge into one aligned batch (the buffer's get_batch path).
    short = concat_padded_tensors(
        [_make_example([11, 12]), _make_example([31, 32, 33])]
    )
    long = concat_padded_tensors([_make_example([21, 22, 23, 24, 25])])
    merged = concat_padded_tensors([short, long])
    assert merged["routed_experts"].shape == (3, 5, NUM_MOE_LAYERS, TOP_K)
    _assert_positions_aligned(merged, [2, 3, 5])
