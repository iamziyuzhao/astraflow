"""Ingest-time validation of R3 routed_experts in AstraDataAcquisition.

``_ingest_structured_result`` is the single validation point: malformed
routed_experts tensors and mixed with/without batches must fail fast here
with a precise error instead of a KeyError later in concat_padded_tensors.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from astraflow.dataflow.data_acquisition import AstraDataAcquisition

NUM_MOE_LAYERS = 4
TOP_K = 2


class _DummyLoader:
    sampler = None

    def __iter__(self):
        yield []


def _make_acquisition(published: list[dict[str, Any]]) -> AstraDataAcquisition:
    def _publish(
        batch: dict[str, Any], metadata: dict[str, Any] | None, timeout: float | None
    ):
        del metadata, timeout
        published.append(batch)
        return True

    return AstraDataAcquisition(
        rollout=object(),
        rollout_dataloader=_DummyLoader(),
        workflow_spec={},
        publish_fn=_publish,
    )


_DEFAULT_EXPERTS = object()


def _make_seq(
    seq_len: int = 3,
    routed_experts: Any = _DEFAULT_EXPERTS,
) -> dict[str, Any]:
    seq: dict[str, Any] = {
        "input_ids": torch.arange(seq_len, dtype=torch.int32).unsqueeze(0),
        "attention_mask": torch.ones((1, seq_len), dtype=torch.bool),
        "rewards": torch.tensor([1.0], dtype=torch.float32),
    }
    if routed_experts is _DEFAULT_EXPERTS:
        routed_experts = torch.zeros(
            (1, seq_len, NUM_MOE_LAYERS, TOP_K), dtype=torch.int16
        )
    if routed_experts is not None:
        seq["routed_experts"] = routed_experts
    return seq


def _make_result(seqs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_trajs": len(seqs),
        "rewards": torch.tensor([1.0] * len(seqs), dtype=torch.float32),
        "trajectories": [{"sequences": [s]} for s in seqs],
    }


def test_valid_routed_experts_are_published_intact():
    published: list[dict[str, Any]] = []
    acquisition = _make_acquisition(published)
    experts = torch.arange(3 * NUM_MOE_LAYERS * TOP_K, dtype=torch.int16).reshape(
        1, 3, NUM_MOE_LAYERS, TOP_K
    )
    acquisition._ingest_structured_result(
        _make_result([_make_seq(3, routed_experts=experts)])
    )
    assert len(published) == 1
    assert torch.equal(published[0]["routed_experts"], experts)
    assert published[0]["routed_experts"].dtype == torch.int16


def test_results_without_routed_experts_still_ingest():
    published: list[dict[str, Any]] = []
    acquisition = _make_acquisition(published)
    acquisition._ingest_structured_result(
        _make_result([_make_seq(3, routed_experts=None)])
    )
    assert len(published) == 1
    assert "routed_experts" not in published[0]


def test_non_tensor_routed_experts_raises():
    published: list[dict[str, Any]] = []
    acquisition = _make_acquisition(published)
    bad = np.zeros((1, 3, NUM_MOE_LAYERS, TOP_K), dtype=np.int16)
    with pytest.raises(TypeError, match="torch tensor"):
        acquisition._ingest_structured_result(
            _make_result([_make_seq(3, routed_experts=bad)])
        )
    assert not published


def test_wrong_dtype_raises():
    published: list[dict[str, Any]] = []
    acquisition = _make_acquisition(published)
    bad = torch.zeros((1, 3, NUM_MOE_LAYERS, TOP_K), dtype=torch.int32)
    with pytest.raises(ValueError, match="int16"):
        acquisition._ingest_structured_result(
            _make_result([_make_seq(3, routed_experts=bad)])
        )
    assert not published


def test_wrong_ndim_raises():
    published: list[dict[str, Any]] = []
    acquisition = _make_acquisition(published)
    bad = torch.zeros((3, NUM_MOE_LAYERS, TOP_K), dtype=torch.int16)
    with pytest.raises(ValueError, match="ndim == 4"):
        acquisition._ingest_structured_result(
            _make_result([_make_seq(3, routed_experts=bad)])
        )
    assert not published


def test_wrong_batch_dim_raises():
    published: list[dict[str, Any]] = []
    acquisition = _make_acquisition(published)
    bad = torch.zeros((2, 3, NUM_MOE_LAYERS, TOP_K), dtype=torch.int16)
    with pytest.raises(ValueError, match="batch dim 1"):
        acquisition._ingest_structured_result(
            _make_result([_make_seq(3, routed_experts=bad)])
        )
    assert not published


def test_seq_len_mismatch_raises():
    published: list[dict[str, Any]] = []
    acquisition = _make_acquisition(published)
    bad = torch.zeros((1, 2, NUM_MOE_LAYERS, TOP_K), dtype=torch.int16)
    with pytest.raises(ValueError, match="seq_len 3"):
        acquisition._ingest_structured_result(
            _make_result([_make_seq(3, routed_experts=bad)])
        )
    assert not published


def test_mixed_sequences_within_result_raise():
    published: list[dict[str, Any]] = []
    acquisition = _make_acquisition(published)
    result = _make_result([_make_seq(3), _make_seq(3, routed_experts=None)])
    with pytest.raises(ValueError, match="Mixed rollout result"):
        acquisition._ingest_structured_result(result)
    assert not published


def test_mixed_results_across_stream_raise_with_then_without():
    published: list[dict[str, Any]] = []
    acquisition = _make_acquisition(published)
    acquisition._ingest_structured_result(_make_result([_make_seq(3)]))
    assert len(published) == 1
    with pytest.raises(ValueError, match="Mixed rollout stream"):
        acquisition._ingest_structured_result(
            _make_result([_make_seq(3, routed_experts=None)])
        )
    assert len(published) == 1


def test_mixed_results_across_stream_raise_without_then_with():
    published: list[dict[str, Any]] = []
    acquisition = _make_acquisition(published)
    acquisition._ingest_structured_result(
        _make_result([_make_seq(3, routed_experts=None)])
    )
    assert len(published) == 1
    with pytest.raises(ValueError, match="Mixed rollout stream"):
        acquisition._ingest_structured_result(_make_result([_make_seq(3)]))
    assert len(published) == 1
