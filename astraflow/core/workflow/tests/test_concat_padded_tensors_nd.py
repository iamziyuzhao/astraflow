"""Tests for ND-aware padding in ``concat_padded_tensors``.

Run:
    pytest astraflow/core/workflow/tests/test_concat_padded_tensors_nd.py -v
"""

import torch

from astraflow.core.workflow.utils.data import concat_padded_tensors

NUM_MOE_LAYERS = 4
TOP_K = 2


def _sample(seq_len: int, offset: int = 0) -> dict[str, torch.Tensor]:
    """Build a [1, seq_len, ...] padded-tensor dict with distinctive values."""
    routed = (
        torch.arange(seq_len * NUM_MOE_LAYERS * TOP_K, dtype=torch.int16) + offset
    ).reshape(1, seq_len, NUM_MOE_LAYERS, TOP_K)
    return {
        "input_ids": torch.arange(seq_len, dtype=torch.int32).unsqueeze(0) + offset,
        "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
        "loss_mask": torch.ones(1, seq_len, dtype=torch.int32),
        "rewards": torch.tensor([float(offset)], dtype=torch.float32),
        "routed_experts": routed,
    }


def test_concat_mixed_length_nd_shapes_and_rows():
    short = _sample(3, offset=1)
    long = _sample(5, offset=100)
    out = concat_padded_tensors([short, long], pad_value=0.0)

    assert out["input_ids"].shape == (2, 5)
    assert out["attention_mask"].shape == (2, 5)
    assert out["loss_mask"].shape == (2, 5)
    assert out["rewards"].shape == (2,)
    assert out["routed_experts"].shape == (2, 5, NUM_MOE_LAYERS, TOP_K)
    assert out["routed_experts"].dtype == torch.int16

    # Per-token rows land at the right positions: valid positions keep the
    # original per-token rows, padded positions are pad_value.
    torch.testing.assert_close(out["routed_experts"][0, :3], short["routed_experts"][0])
    assert torch.all(out["routed_experts"][0, 3:] == 0)
    torch.testing.assert_close(out["routed_experts"][1], long["routed_experts"][0])

    # 2D keys behave exactly as before.
    torch.testing.assert_close(out["input_ids"][0, :3], short["input_ids"][0])
    assert torch.all(out["input_ids"][0, 3:] == 0)
    assert torch.all(out["attention_mask"][0, :3])
    assert torch.all(~out["attention_mask"][0, 3:])
    torch.testing.assert_close(out["input_ids"][1], long["input_ids"][0])
    assert torch.all(out["attention_mask"][1])


def test_concat_equal_length_nd_no_padding():
    a = _sample(4, offset=0)
    b = _sample(4, offset=50)
    out = concat_padded_tensors([a, b])

    assert out["routed_experts"].shape == (2, 4, NUM_MOE_LAYERS, TOP_K)
    torch.testing.assert_close(out["routed_experts"][0], a["routed_experts"][0])
    torch.testing.assert_close(out["routed_experts"][1], b["routed_experts"][0])


def test_concat_2d_only_behavior_unchanged():
    dicts = []
    for seq_len, offset in ((2, 1), (4, 10)):
        d = _sample(seq_len, offset=offset)
        d.pop("routed_experts")
        dicts.append(d)
    out = concat_padded_tensors(dicts, pad_value=0.0)

    assert set(out) == {"input_ids", "attention_mask", "loss_mask", "rewards"}
    assert out["input_ids"].shape == (2, 4)
    torch.testing.assert_close(out["input_ids"][0, :2], dicts[0]["input_ids"][0])
    assert torch.all(out["input_ids"][0, 2:] == 0)
    assert torch.all(~out["attention_mask"][0, 2:])
