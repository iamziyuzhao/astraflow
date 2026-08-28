"""RolloutBuffer put/get must be ND-safe for R3 routed_experts tensors.

Assertion-style verification (no buffer code changes expected): per-example
slicing in ``put`` and batch re-concat in ``get_batch`` must preserve the
dtype, trailing dims, and row order of ``[1, seq_len, num_moe_layers, top_k]``
tensors.
"""

from __future__ import annotations

import torch

from astraflow.dataflow.rollout_buffer import RolloutBuffer
from astraflow.dataflow.tests.test_concat_padded_tensors_nd import (
    NUM_MOE_LAYERS,
    TOP_K,
    _make_example,
    _routed_experts_for,
)
from astraflow.dataflow.utils import concat_padded_tensors


def test_put_get_preserves_routed_experts_dtype_shape_order():
    buffer = RolloutBuffer(max_size=8, queue_order="fifo")
    examples = [
        _make_example([11, 12, 13]),
        _make_example([21, 22, 23, 24]),
    ]
    for example in examples:
        assert buffer.put(example, metadata={"min_version": 1})

    result = buffer.get_batch(batch_size=2, timeout=5.0, current_version=1)
    assert result is not None
    batch, metadatas = result
    assert len(metadatas) == 2

    experts = batch["routed_experts"]
    assert experts.dtype == torch.int16
    assert experts.shape == (2, 4, NUM_MOE_LAYERS, TOP_K)
    # FIFO: arrival order preserved.
    assert batch["input_ids"][0, :3].tolist() == [11, 12, 13]
    assert batch["input_ids"][1, :4].tolist() == [21, 22, 23, 24]
    for b, length in enumerate((3, 4)):
        expected = _routed_experts_for(batch["input_ids"][b : b + 1, :length])
        assert torch.equal(experts[b : b + 1, :length], expected)
        assert torch.all(experts[b, length:] == 0)


def test_put_slices_multi_example_batch_nd():
    # A multi-sequence batch is sliced into per-example entries on put;
    # each entry must keep its full [1, S, L, K] routed_experts block.
    buffer = RolloutBuffer(max_size=8, queue_order="fifo")
    combined = concat_padded_tensors(
        [_make_example([11, 12]), _make_example([21, 22, 23])]
    )
    assert buffer.put(combined, metadata={"min_version": 1})
    assert buffer.size() == 2

    result = buffer.get_batch(batch_size=2, timeout=5.0, current_version=1)
    assert result is not None
    batch, _ = result
    assert batch["routed_experts"].shape == (2, 3, NUM_MOE_LAYERS, TOP_K)
    assert batch["routed_experts"].dtype == torch.int16
    for b, length in enumerate((2, 3)):
        expected = _routed_experts_for(batch["input_ids"][b : b + 1, :length])
        assert torch.equal(batch["routed_experts"][b : b + 1, :length], expected)


def test_state_dict_round_trip_preserves_routed_experts():
    buffer = RolloutBuffer(max_size=8, queue_order="fifo")
    buffer.put(_make_example([11, 12, 13]), metadata={"min_version": 1})
    state = buffer.state_dict()

    restored = RolloutBuffer(max_size=8, queue_order="fifo")
    restored.load_state_dict(state)
    result = restored.get_batch(batch_size=1, timeout=5.0, current_version=1)
    assert result is not None
    batch, _ = result
    assert batch["routed_experts"].dtype == torch.int16
    assert batch["routed_experts"].shape == (1, 3, NUM_MOE_LAYERS, TOP_K)
    expected = _routed_experts_for(batch["input_ids"])
    assert torch.equal(batch["routed_experts"], expected)
