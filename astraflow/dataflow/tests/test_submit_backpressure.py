"""Submission back-pressure: the fresh buffer's fill gates prompt submission.

Why this exists. Rollout ran ~4x faster than training consumed on the
Qwen3-30B-A3B runs, the fresh buffer was an open loop, and under
``queue_order=edf`` the trainer was handed the *oldest* permissible sample
every step: measured staleness climbed from ~7 to ~23 versions (the
``max_staleness`` ceiling) over the run, and MATH-500 tracked it -- rising
while staleness stayed under ~10, eroding once it passed ~17, in both runs.

``max_buffered_samples`` closes the loop: ``_submit_budget`` caps each
submit tick so the fullest fresh buffer (the per-model max) stays under the
cap, dividing the headroom by the accepted-samples-per-prompt estimate
(``accepted / producer_batches``). Generation in flight is bounded
separately by RaaS (``max_concurrent_rollouts``), so the two together cap
how much data can age ahead of the trainer.

These tests drive the submit tick directly (no threads) with a fake RaaS
and a fake serving layer, check the production wiring (YAML -> loader ->
AgentConfig -> AstraFlowService -> AstraFlow -> acquisition), and run the
whole loop tick by tick against the real AstraFlow objects.
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


class _Serving:
    """The two things the gate reads from MultiModelDataServing."""

    def __init__(self, sizes: dict[str, int]):
        self.sizes = sizes

    @property
    def model_ids(self) -> list[str]:
        return list(self.sizes)

    def size(self, model_id: str | None = None) -> int:
        return self.sizes[model_id] if model_id is not None else sum(self.sizes.values())

    # stats sinks the ingest path pushes to; irrelevant to the gate
    def accumulate_acquisition_stats(self, stats):
        pass

    def accumulate_agent_metrics(self, metrics):
        pass


def _acquisition(
    rollout: _Rollout,
    max_buffered: int | None,
    serving: _Serving | None = None,
    **kwargs: Any,
) -> AstraDataAcquisition:
    return AstraDataAcquisition(
        rollout=rollout,
        rollout_dataloader=_Loader(),
        workflow_spec={},
        publish_fn=lambda *a, **k: True,
        data_serving=serving,
        max_buffered_samples=max_buffered,
        **kwargs,
    )


@pytest.fixture
def no_tick_sleep(monkeypatch):
    """The submit tick sleeps 0.1 s when gated or idle; not in a unit test."""
    from astraflow.dataflow import data_acquisition as da_mod

    monkeypatch.setattr(da_mod.time, "sleep", lambda _s: None)


# ----------------------------------------------------------------------
# The gate itself
# ----------------------------------------------------------------------


def test_closed_gate_submits_nothing():
    rollout = _Rollout(capacity=8)
    serving = _Serving({"model0": 4})
    acq = _acquisition(rollout, max_buffered=4, serving=serving)

    n, _ = acq._submit_tick_debug(256)

    assert n == 0
    assert rollout.submitted == []


def test_open_gate_submits_only_the_headroom():
    """Buffer holds 1 of 4: three prompts go out, not the full capacity."""
    rollout = _Rollout(capacity=8)
    serving = _Serving({"model0": 1})
    acq = _acquisition(rollout, max_buffered=4, serving=serving)

    n, _ = acq._submit_tick_debug(256)

    assert n == 3
    assert len(rollout.submitted) == 3

    serving.sizes["model0"] = 4
    n, _ = acq._submit_tick_debug(256)
    assert n == 0
    assert len(rollout.submitted) == 3


def test_headroom_is_divided_by_observed_samples_per_prompt():
    """80 accepted sequences over 10 prompts -> 8 per prompt -> ceil(20/8)=3."""
    rollout = _Rollout(capacity=64)
    acq = _acquisition(rollout, max_buffered=20, serving=_Serving({"model0": 0}))
    with acq._stats_lock:
        acq._ingest_stats["accepted"] = 80
        acq._stats["producer_batches"] = 10

    n, _ = acq._submit_tick_debug(256)

    assert n == 3


def test_headroom_still_submits_at_least_one_prompt():
    rollout = _Rollout(capacity=64)
    acq = _acquisition(rollout, max_buffered=20, serving=_Serving({"model0": 19}))
    with acq._stats_lock:
        acq._ingest_stats["accepted"] = 800
        acq._stats["producer_batches"] = 100

    n, _ = acq._submit_tick_debug(256)

    assert n == 1


def test_a_low_acceptance_rate_never_inflates_the_budget():
    """Fewer than one accepted sample per prompt counts as one (never more
    prompts than headroom)."""
    acq = _acquisition(_Rollout(capacity=64), max_buffered=8, serving=_Serving({"model0": 0}))
    with acq._stats_lock:
        acq._ingest_stats["accepted"] = 3
        acq._stats["producer_batches"] = 10
    assert acq._submit_budget(64) == 8


def test_raas_capacity_still_binds_when_below_headroom():
    rollout = _Rollout(capacity=2)
    acq = _acquisition(rollout, max_buffered=1000, serving=_Serving({"model0": 0}))

    n, _ = acq._submit_tick_debug(256)

    assert n == 2


def test_unconfigured_gate_keeps_the_open_loop():
    rollout = _Rollout(capacity=8)
    acq = _acquisition(rollout, max_buffered=None, serving=None)

    n, _ = acq._submit_tick_debug(256)

    assert n == 8
    assert acq._submit_budget(123) == 123


def test_plain_tick_honours_the_gate_too(no_tick_sleep):
    rollout = _Rollout(capacity=8)
    serving = _Serving({"model0": 4})
    acq = _acquisition(rollout, max_buffered=4, serving=serving)

    assert acq._submit_tick(256) == 0
    assert rollout.submitted == []

    serving.sizes["model0"] = 2
    assert acq._submit_tick(256) == 2
    assert len(rollout.submitted) == 2


def test_multi_model_backlog_is_the_most_backed_up_buffer():
    """The per-model max gates, not the sum: 1 + 3 of cap 4 still has headroom 1."""
    acq = _acquisition(
        _Rollout(capacity=8), max_buffered=4, serving=_Serving({"model0": 1, "model1": 3})
    )
    assert acq._submit_budget(8) == 1
    acq._data_serving.sizes["model1"] = 4
    assert acq._submit_budget(8) == 0


def _result(n: int, version: int, reward_pattern: str = "mixed") -> dict[str, Any]:
    """One rollout result: ``n`` sequences stamped with the weight version
    that generated them. ``mixed`` rewards survive filter_zero_adv; ``flat``
    ones (all equal) do not."""
    rewards = [float(i % 2) if reward_pattern == "mixed" else 0.0 for i in range(n)]
    seqs = [
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "rewards": torch.tensor([r], dtype=torch.float32),
            "versions": torch.tensor([[version, version, version]], dtype=torch.long),
        }
        for r in rewards
    ]
    return {
        "n_trajs": n,
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "trajectories": [{"sequences": seqs}],
    }


def test_samples_per_prompt_is_learned_from_accepted_ingestion():
    """The estimate divides ACCEPTED sequences by ingested prompts, so a
    filter that rejects whole groups lowers it, and an upstream-rejected
    None result counts as a prompt that yielded nothing."""
    acq = _acquisition(
        _Rollout(capacity=64),
        max_buffered=64,
        serving=_Serving({"model0": 0}),
        filter_fn="filter_zero_adv",
    )

    acq._ingest_one_result(_result(8, version=1, reward_pattern="mixed"))
    assert acq._submit_budget(64) == 8  # 8 accepted / 1 prompt -> 64 / 8

    acq._ingest_one_result(_result(8, version=1, reward_pattern="flat"))
    assert acq._submit_budget(64) == 16  # 8 accepted / 2 prompts -> 64 / 4

    acq._ingest_one_result(None)
    assert acq._submit_budget(64) == 24  # 8 accepted / 3 prompts -> ceil(64 / 2.67)

    stats = acq.get_ingest_stats()
    assert stats["accepted"] == 8 and stats["total"] == 16
    assert acq.get_stats()["producer_batches"] == 3


# ----------------------------------------------------------------------
# Production wiring
# ----------------------------------------------------------------------


def _batch(n: int) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.zeros(n, 2, dtype=torch.long),
        "attention_mask": torch.ones(n, 2, dtype=torch.long),
        "rewards": torch.zeros(n, dtype=torch.float32),
    }


def test_astraflow_gates_on_its_own_serving_buffer():
    rollout = _Rollout(capacity=8)
    flow = AstraFlow(
        rollout=rollout,
        rollout_dataloader=_Loader(),
        workflow_spec={},
        max_buffered_samples=3,
    )
    acq = flow.data_acquisition
    assert acq._max_buffered_samples == 3
    assert acq._data_serving is flow.data_serving
    assert acq._submit_budget(8) == 3

    assert flow.data_serving.put(_batch(3), {"min_version": 0}, 0.1)
    n, _ = acq._submit_tick_debug(256)
    assert n == 0
    assert rollout.submitted == []


def test_service_forwards_the_cap_from_agent_config():
    config = ServiceConfig(agent=AgentConfig(max_buffered_samples=64))
    service = AstraFlowService(config)
    try:
        service.register_agent("default", config.agent)
        flow = service.flows["default"]
        acq = flow.data_acquisition
        assert acq._max_buffered_samples == 64
        assert acq._data_serving is flow.data_serving
        assert acq._submit_budget(8) == 8
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
    assert (
        AgentConfig(
            **{k: v for k, v in agent.items() if k in AgentConfig.__dataclass_fields__}
        ).max_buffered_samples
        == 512
    )


def test_loader_leaves_the_gate_unset_by_default():
    raw = _raw({"max_staleness": 8})
    assert load_dataflow_config(raw)["agent"].get("max_buffered_samples") is None


def test_multi_model_flow_gates_on_the_most_backed_up_buffer():
    flow = AstraFlow(
        rollout=_Rollout(),
        rollout_dataloader=_Loader(),
        workflow_spec={},
        expected_model_ids=["model0", "model1"],
        max_buffered_samples=8,
    )
    acq = flow.data_acquisition
    assert acq._submit_budget(8) == 8
    flow.data_serving.buffers["model1"].put(_batch(3), {"min_version": 1}, 0.1)
    assert acq._submit_budget(8) == 5


# ----------------------------------------------------------------------
# The loop itself: submit -> generate -> ingest -> train -> resubmit
# ----------------------------------------------------------------------


class _LatencyRaaS:
    """RaaS stand-in with a bounded in-flight pool and a fixed generation
    latency in ticks. Each completed prompt yields ``n`` sequences stamped
    with the weight version current when it was submitted."""

    def __init__(self, capacity: int, n_samples: int, latency: int):
        self.capacity, self.n, self.latency = capacity, n_samples, latency
        self.version = 1
        self.tick = 0
        self.pending: list[tuple[int, int, int]] = []
        self.next_id = 1

    def get_raas_availability(self) -> dict[str, int]:
        return {
            "available": max(0, self.capacity - len(self.pending)),
            "inflight": len(self.pending),
        }

    def submit_auto(self, data, workflow_spec=None, **kwargs):
        del data, workflow_spec, kwargs
        tid = self.next_id
        self.next_id += 1
        self.pending.append((self.tick + self.latency, tid, self.version))
        return tid

    def advance(self) -> None:
        self.tick += 1

    def pull_completed(self, max_items: int = 512, timeout: float = 0.0):
        del timeout
        done = [p for p in self.pending if p[0] <= self.tick][:max_items]
        done_ids = {p[1] for p in done}
        self.pending = [p for p in self.pending if p[1] not in done_ids]
        return [
            {"task_id": tid, "ok": True, "result": _result(self.n, v), "error": None}
            for (_, tid, v) in done
        ]


def _simulate(
    cap: int | None,
    *,
    ticks: int = 300,
    trainer_period: int = 6,
    tbs: int = 16,
    n: int = 4,
    capacity: int = 8,
    latency: int = 2,
) -> dict[str, int]:
    """Drive the real AstraFlow objects tick by tick.

    Supply is capacity*n/latency = 16 samples per tick; the trainer takes
    tbs=16 every ``trainer_period`` ticks, so rollout out-supplies training
    6x -- the shape of the 30B runs.
    """
    raas = _LatencyRaaS(capacity, n, latency)
    flow = AstraFlow(
        rollout=raas,
        rollout_dataloader=_Loader(n=100000),
        workflow_spec={},
        buffer_size=100000,
        max_buffered_samples=cap,
    )
    acq = flow.data_acquisition
    version = 1
    out = {"max_buffered": 0, "max_staleness": 0, "waits": 0, "steps": 0}
    for t in range(ticks):
        acq._submit_tick_debug(256)
        raas.advance()
        for item in raas.pull_completed():
            acq._ingest_one_result(item["result"])
        out["max_buffered"] = max(out["max_buffered"], flow.size())
        if t % trainer_period == trainer_period - 1:
            if flow.size() < tbs:
                out["waits"] += 1
                continue
            _, metas = flow.get_training_batch(
                expected_sample_count=tbs,
                replay_ratio=0.0,
                timeout=0.5,
                current_version=version,
            )
            stale = max(version - int(m["min_version"]) for m in metas)
            out["max_staleness"] = max(out["max_staleness"], stale)
            version += 1
            raas.version = version
            acq.notify_version_changed(version)
            out["steps"] += 1
    return out


def test_closed_loop_bounds_outstanding_samples_and_staleness(no_tick_sleep):
    tbs, n, capacity = 16, 4, 8
    cap = 2 * tbs
    closed = _simulate(cap, tbs=tbs, n=n, capacity=capacity)

    assert closed["steps"] >= 40
    assert closed["waits"] == 0
    # Buffered never exceeds the cap plus what was already in flight.
    assert closed["max_buffered"] <= cap + capacity * n
    # Staleness is bounded by the outstanding data, in batches, plus one.
    bound = -(-(cap + capacity * n) // tbs) + 1
    assert closed["max_staleness"] <= bound


def test_open_loop_staleness_grows_without_bound(no_tick_sleep):
    """The regression the gate exists for: with no cap the buffer fills and
    edf hands the trainer ever-older samples."""
    tbs, n, capacity = 16, 4, 8
    opened = _simulate(None, tbs=tbs, n=n, capacity=capacity)
    closed = _simulate(2 * tbs, tbs=tbs, n=n, capacity=capacity)

    assert opened["max_buffered"] > 10 * (2 * tbs + capacity * n)
    assert opened["max_staleness"] > 4 * closed["max_staleness"]


def test_gate_reopens_when_the_trainer_drains_the_buffer(no_tick_sleep):
    rollout = _Rollout(capacity=8)
    flow = AstraFlow(
        rollout=rollout,
        rollout_dataloader=_Loader(),
        workflow_spec={},
        max_buffered_samples=4,
    )
    acq = flow.data_acquisition
    assert flow.data_serving.put(_batch(4), {"min_version": 1}, 0.1)

    n, _ = acq._submit_tick_debug(256)
    assert n == 0

    out = flow.get_training_batch(
        expected_sample_count=4, replay_ratio=0.0, timeout=0.5, current_version=2
    )
    assert out is not None
    assert flow.size() == 0

    n, _ = acq._submit_tick_debug(256)
    assert n == 4  # headroom 4 / (no ingest yet -> 1 per prompt)
