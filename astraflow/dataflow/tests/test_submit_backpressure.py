"""Submission back-pressure: the fresh buffer's fill gates prompt submission.

Why this exists. Rollout ran ~4x faster than training consumed on the
Qwen3-30B-A3B runs, the fresh buffer was an open loop, and under
``queue_order=edf`` the trainer was handed the *oldest* permissible sample
every step: measured staleness climbed from ~7 to ~23 versions (the
``max_staleness`` ceiling) over the run, and MATH-500 tracked it -- rising
while staleness stayed under ~10, eroding once it passed ~17, in both runs.

``max_buffered_samples`` closes the loop: while the fresh buffer already
holds that many samples the submit tick sends nothing, and when it is
below, it submits only about enough prompts to fill the gap. Generation in
flight is bounded separately by RaaS (``max_concurrent_rollouts``), so the
two together cap how much data can age ahead of the trainer.

These tests drive the submit tick directly (no threads) with a fake RaaS
and a fake backlog reading, then check the production wiring: YAML ->
loader -> AgentConfig -> AstraFlowService -> AstraFlow -> acquisition.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from astraflow.core.config.loader import load_dataflow_config
from astraflow.dataflow.astraflow import AstraFlow
from astraflow.dataflow.data_acquisition import AstraDataAcquisition
from astraflow.dataflow.service import AstraFlowService
from astraflow.dataflow.service_config import AgentConfig, ServiceConfig

# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


class _Loader:
    sampler = None

    def __init__(self, n: int = 64):
        self.n = n

    def __iter__(self):
        yield [{"x": i} for i in range(self.n)]


class _Rollout:
    """RaaS stand-in: a fixed in-flight capacity, records every submit."""

    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self.inflight = 0
        self.submitted: list[dict[str, Any]] = []

    def get_raas_availability(self) -> dict[str, int]:
        return {
            "available": max(0, self.capacity - self.inflight),
            "inflight": self.inflight,
        }

    def submit_auto(self, data, workflow_spec=None, **kwargs):
        del workflow_spec, kwargs
        self.submitted.append(data)
        self.inflight += 1
        return len(self.submitted)

    def pull_completed(self, max_items: int = 256, timeout: float = 0.0):
        del max_items, timeout
        return []


def _acquisition(
    rollout: _Rollout,
    max_buffered: int | None,
    backlog: dict[str, Any],
) -> AstraDataAcquisition:
    return AstraDataAcquisition(
        rollout=rollout,
        rollout_dataloader=_Loader(),
        workflow_spec={},
        publish_fn=lambda *a, **k: True,
        max_buffered_samples=max_buffered,
        buffered_fn=(lambda: backlog["n"]) if max_buffered is not None else None,
    )


# ----------------------------------------------------------------------
# The gate itself
# ----------------------------------------------------------------------


def test_closed_gate_submits_nothing_and_counts_the_tick():
    rollout = _Rollout(capacity=8)
    backlog = {"n": 4}
    acq = _acquisition(rollout, max_buffered=4, backlog=backlog)

    n, info = acq._submit_tick_debug(256)

    assert n == 0
    assert info["gated"] is True
    assert info["buffered"] == 4
    assert rollout.submitted == []
    assert acq.get_stats()["submit_gated_ticks"] == 1


def test_open_gate_submits_only_the_headroom():
    """Buffer holds 1 of 4: three prompts go out, not the full capacity."""
    rollout = _Rollout(capacity=8)
    backlog = {"n": 1}
    acq = _acquisition(rollout, max_buffered=4, backlog=backlog)

    n, info = acq._submit_tick_debug(256)

    assert n == 3
    assert len(rollout.submitted) == 3
    assert "gated" not in info

    backlog["n"] = 4
    n, info = acq._submit_tick_debug(256)
    assert n == 0 and info["gated"] is True
    assert len(rollout.submitted) == 3


def test_headroom_is_divided_by_observed_samples_per_prompt():
    """80 sequences over 10 ingested results -> 8 per prompt -> ceil(20/8)=3."""
    rollout = _Rollout(capacity=64)
    backlog = {"n": 0}
    acq = _acquisition(rollout, max_buffered=20, backlog=backlog)
    with acq._stats_lock:
        acq._ingest_stats["total"] = 80
        acq._stats["producer_batches"] = 10

    n, _ = acq._submit_tick_debug(256)

    assert n == 3


def test_headroom_still_submits_at_least_one_prompt():
    rollout = _Rollout(capacity=64)
    backlog = {"n": 19}
    acq = _acquisition(rollout, max_buffered=20, backlog=backlog)
    with acq._stats_lock:
        acq._ingest_stats["total"] = 800
        acq._stats["producer_batches"] = 100

    n, _ = acq._submit_tick_debug(256)

    assert n == 1


def test_raas_capacity_still_binds_when_below_headroom():
    rollout = _Rollout(capacity=2)
    backlog = {"n": 0}
    acq = _acquisition(rollout, max_buffered=1000, backlog=backlog)

    n, _ = acq._submit_tick_debug(256)

    assert n == 2


def test_unconfigured_gate_keeps_the_open_loop():
    rollout = _Rollout(capacity=8)
    acq = _acquisition(rollout, max_buffered=None, backlog={})

    n, info = acq._submit_tick_debug(256)

    assert n == 8
    assert info["buffered"] == -1
    assert "gated" not in info
    assert acq.get_stats()["submit_gated_ticks"] == 0


def test_plain_tick_honours_the_gate_too():
    rollout = _Rollout(capacity=8)
    backlog = {"n": 4}
    acq = _acquisition(rollout, max_buffered=4, backlog=backlog)

    assert acq._submit_tick(256) == 0
    assert rollout.submitted == []

    backlog["n"] = 2
    assert acq._submit_tick(256) == 2
    assert len(rollout.submitted) == 2


def test_backlog_read_failure_fails_open():
    """A broken backlog reading must not silently stop all rollout."""
    rollout = _Rollout(capacity=8)
    acq = AstraDataAcquisition(
        rollout=rollout,
        rollout_dataloader=_Loader(),
        workflow_spec={},
        publish_fn=lambda *a, **k: True,
        max_buffered_samples=4,
        buffered_fn=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    n, info = acq._submit_tick_debug(256)

    assert n == 8
    assert info["buffered"] == -1


def test_gate_transitions_are_printed_once(capsys):
    rollout = _Rollout(capacity=8)
    backlog = {"n": 4}
    acq = _acquisition(rollout, max_buffered=4, backlog=backlog)

    acq._submit_tick_debug(256)
    acq._submit_tick_debug(256)
    backlog["n"] = 0
    acq._submit_tick_debug(256)
    acq._submit_tick_debug(256)

    out = capsys.readouterr().out
    assert out.count("[AstraFlow-submit-gate] CLOSED") == 1
    assert out.count("[AstraFlow-submit-gate] open") == 1


@pytest.mark.parametrize("bad", [0, -5])
def test_non_positive_cap_is_rejected(bad):
    with pytest.raises(ValueError, match="positive"):
        _acquisition(_Rollout(), max_buffered=bad, backlog={"n": 0})


def test_cap_without_backlog_reader_is_rejected():
    with pytest.raises(ValueError, match="buffered_fn"):
        AstraDataAcquisition(
            rollout=_Rollout(),
            rollout_dataloader=_Loader(),
            workflow_spec={},
            publish_fn=lambda *a, **k: True,
            max_buffered_samples=4,
        )


# ----------------------------------------------------------------------
# Production wiring
# ----------------------------------------------------------------------


def _batch(n: int) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.zeros(n, 2, dtype=torch.long),
        "attention_mask": torch.ones(n, 2, dtype=torch.long),
        "rewards": torch.zeros(n, dtype=torch.float32),
    }


def test_astraflow_reads_the_backlog_from_its_own_serving_buffer():
    rollout = _Rollout(capacity=8)
    flow = AstraFlow(
        rollout=rollout,
        rollout_dataloader=_Loader(),
        workflow_spec={},
        max_buffered_samples=3,
    )
    acq = flow.data_acquisition
    assert acq._max_buffered_samples == 3
    assert acq._buffered_fn() == 0

    assert flow.data_serving.put(_batch(3), {"min_version": 0}, 0.1)
    assert acq._buffered_fn() == 3

    n, info = acq._submit_tick_debug(256)
    assert n == 0 and info["gated"] is True
    assert rollout.submitted == []


def test_service_forwards_the_cap_from_agent_config():
    config = ServiceConfig(agent=AgentConfig(max_buffered_samples=64))
    service = AstraFlowService(config)
    try:
        service.register_agent("default", config.agent)
        acq = service.flows["default"].data_acquisition
        assert acq._max_buffered_samples == 64
        assert acq._buffered_fn is not None
        assert acq._buffered_fn() == 0
    finally:
        service.raas_pool.shutdown()


def test_service_default_leaves_the_gate_off():
    config = ServiceConfig(agent=AgentConfig())
    service = AstraFlowService(config)
    try:
        service.register_agent("default", config.agent)
        assert service.flows["default"].data_acquisition._max_buffered_samples is None
    finally:
        service.raas_pool.shutdown()


def _raw(buffer: dict, train_batch_size: int | None = 256) -> dict:
    """A merged-config dict shaped like load_and_merge_configs' output."""
    raw = {
        "experiment": {"experiment_name": "r3", "trial_name": "t0"},
        "dataflow": {"host": "127.0.0.1", "port": 18123, "buffer": dict(buffer)},
    }
    if train_batch_size is not None:
        raw["trainer_base"] = {"train_batch_size": train_batch_size}
    return raw


def test_buffer_yaml_key_reaches_the_agent_config():
    raw = _raw({"max_staleness": 4, "max_buffered_samples": 512})
    agent = load_dataflow_config(raw)["agent"]
    assert agent["max_buffered_samples"] == 512
    assert agent["max_staleness"] == 4
    assert AgentConfig(**{k: v for k, v in agent.items() if k in AgentConfig.__dataclass_fields__}).max_buffered_samples == 512


def test_loader_rejects_a_cap_below_the_training_batch():
    raw = _raw({"max_buffered_samples": 100}, train_batch_size=256)
    with pytest.raises(ValueError, match="below"):
        load_dataflow_config(raw)


def test_loader_accepts_a_cap_equal_to_the_training_batch():
    raw = _raw({"max_buffered_samples": 256}, train_batch_size=256)
    assert load_dataflow_config(raw)["agent"]["max_buffered_samples"] == 256


def test_loader_rejects_a_non_positive_cap():
    raw = _raw({"max_buffered_samples": 0})
    with pytest.raises(ValueError, match="positive"):
        load_dataflow_config(raw)


def test_loader_without_a_trainer_section_only_checks_positivity():
    raw = _raw({"max_buffered_samples": 5}, train_batch_size=None)
    assert load_dataflow_config(raw)["agent"]["max_buffered_samples"] == 5


def test_loader_leaves_the_gate_unset_by_default():
    raw = _raw({"max_staleness": 8})
    assert load_dataflow_config(raw)["agent"].get("max_buffered_samples") is None
