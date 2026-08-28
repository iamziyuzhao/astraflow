"""Collect-path tunables must be reachable from YAML and must actually bite.

Regression coverage for two coupled bugs:

1. ``pull_timeout`` / ``max_collect_per_tick`` / ``collect_timeout`` existed
   on the engine and on ``AstraDataAcquisition`` but nothing in the
   production construction path ever passed them, so they were dead knobs.
2. ``RaaSPool.pull_completed`` waited on its collect futures with a
   hardcoded ``max(timeout + 5.0, 10.0)``.  That is <= the engine's own
   10s HTTP timeout, so the future always expired first and raising
   ``pull_timeout`` could not help.  A future expiry marks the instance
   suspect and permanently drops the rollouts the server already popped.
"""

from __future__ import annotations

import dataclasses
import inspect
import textwrap

import pytest

from astraflow.dataflow import raas_pool as raas_pool_mod
from astraflow.dataflow.astraflow import AstraFlow
from astraflow.dataflow.data_acquisition import AstraDataAcquisition
from astraflow.dataflow.raas2_engine import RaaS2InferenceEngine
from astraflow.dataflow.raas_pool import (
    _COLLECT_FUTURE_GRACE_SEC,
    _COLLECT_FUTURE_MIN_SEC,
    RaaSPool,
)
from astraflow.dataflow.service import AstraFlowService
from astraflow.dataflow.service_config import AgentConfig, ServiceConfig

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class _DummyLoader:
    sampler = None

    def __iter__(self):
        yield []


class _FakeEngine:
    """Minimal stand-in exposing only what the collect path touches."""

    def __init__(self, pull_timeout: float | None = None):
        self.pull_timeout = pull_timeout
        self.calls: list[tuple[int, float]] = []

    def pull_completed(self, max_items, timeout):
        self.calls.append((max_items, timeout))
        return []


class _RecordingFuture:
    def __init__(self, fn, args, sink):
        self._fn = fn
        self._args = args
        self._sink = sink

    def result(self, timeout=None):
        self._sink.append(timeout)
        return self._fn(*self._args)


class _RecordingExecutor:
    """Executor stub that records the timeout passed to ``future.result``."""

    def __init__(self):
        self.result_timeouts: list[float] = []

    def submit(self, fn, *args):
        return _RecordingFuture(fn, args, self.result_timeouts)

    def shutdown(self, wait=True):
        pass


def _default_of(func, name):
    return inspect.signature(func).parameters[name].default


@pytest.fixture
def pool_factory():
    """Build RaaSPools and guarantee their heartbeat threads are stopped."""
    created: list[RaaSPool] = []

    def _make(**kwargs) -> RaaSPool:
        pool = RaaSPool(**kwargs)
        created.append(pool)
        return pool

    yield _make

    for pool in created:
        pool.shutdown()


# ----------------------------------------------------------------------
# (1) The tunables reach the engine / acquisition objects from a config
#     the way production builds them.
# ----------------------------------------------------------------------


def test_agent_config_exposes_the_collect_knobs():
    names = {f.name for f in dataclasses.fields(AgentConfig)}
    assert {"raas_pull_timeout", "max_collect_per_tick", "collect_timeout"} <= names


def test_dataflow_yaml_section_reaches_agent_config(tmp_path):
    """The production YAML -> ServiceConfig path, not a hand-built config.

    ``__main__._parse_config`` filters the ``dataflow:`` section against
    ``dataclasses.fields(AgentConfig)``, so a knob only becomes settable
    from YAML once it is a real AgentConfig field.
    """
    from astraflow.dataflow.__main__ import _parse_config

    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        textwrap.dedent(
            """\
            experiment:
              experiment_name: r3
              trial_name: t0
            dataflow:
              host: 127.0.0.1
              port: 18123
              raas_pull_timeout: 240.0
              max_collect_per_tick: 48
              collect_timeout: 2.0
            """
        )
    )

    config = _parse_config(str(config_path))
    assert config.agent.raas_pull_timeout == 240.0
    assert config.agent.max_collect_per_tick == 48
    assert config.agent.collect_timeout == 2.0


def test_service_forwards_knobs_to_pool_and_acquisition():
    """End-to-end through the objects ``python -m astraflow`` builds."""
    config = ServiceConfig(
        agent=AgentConfig(
            raas_pull_timeout=240.0,
            max_collect_per_tick=48,
            collect_timeout=2.0,
        )
    )
    service = AstraFlowService(config)
    try:
        assert service.raas_pool._pull_timeout == 240.0

        service.register_agent("default", config.agent)
        acquisition = service.flows["default"].data_acquisition
        assert acquisition._max_collect_per_tick == 48
        assert acquisition._collect_timeout == 2.0
    finally:
        service.raas_pool.shutdown()


def test_pool_hands_pull_timeout_to_every_engine_it_creates(pool_factory, monkeypatch):
    """``register`` used to call ``RaaS2InferenceEngine(service_url=...)`` only."""
    captured: dict[str, object] = {}

    class _StubEngine(_FakeEngine):
        def __init__(self, *, service_url, pull_timeout=None):
            super().__init__(pull_timeout=pull_timeout)
            captured["service_url"] = service_url
            captured["pull_timeout"] = pull_timeout

        def initialize(self, **kwargs):
            captured["initialized"] = True

    monkeypatch.setattr(raas_pool_mod, "RaaS2InferenceEngine", _StubEngine)

    pool = pool_factory(pull_timeout=240.0)
    pool.register("raas-0", "http://localhost:1")

    assert captured["service_url"] == "http://localhost:1"
    assert captured["pull_timeout"] == 240.0
    assert pool._engines["raas-0"].pull_timeout == 240.0


def test_astraflow_forwards_collect_knobs_to_acquisition():
    class _Serving:
        buffer = None

        def put(self, *args, **kwargs):
            return True

    flow = AstraFlow(
        rollout=object(),
        rollout_dataloader=_DummyLoader(),
        workflow_spec={},
        data_serving=_Serving(),
        max_collect_per_tick=17,
        collect_timeout=0.25,
    )
    assert flow.data_acquisition._max_collect_per_tick == 17
    assert flow.data_acquisition._collect_timeout == 0.25


# ----------------------------------------------------------------------
# (2) The pool's future timeout tracks a raised pull timeout.
# ----------------------------------------------------------------------


def test_collect_future_timeout_tracks_raised_pull_timeout():
    engines = [("raas-0", _FakeEngine(pull_timeout=240.0))]
    collect = RaaSPool._collect_future_timeout(0.1, engines)

    # The old hardcoded formula: max(0.1 + 5.0, 10.0) == 10.0.
    assert collect != 10.0
    assert collect == 240.0 + _COLLECT_FUTURE_GRACE_SEC
    # The future must outlast the engine's own HTTP timeout, otherwise it
    # fires first and the popped results are lost.
    assert collect > engines[0][1].pull_timeout


def test_collect_future_timeout_uses_the_slowest_engine():
    engines = [
        ("raas-0", _FakeEngine(pull_timeout=30.0)),
        ("raas-1", _FakeEngine(pull_timeout=240.0)),
    ]
    assert RaaSPool._collect_future_timeout(0.1, engines) == (
        240.0 + _COLLECT_FUTURE_GRACE_SEC
    )


def test_pull_completed_waits_on_the_configured_pull_timeout(pool_factory):
    """The knob reaches the real ``fut.result(timeout=...)`` call."""
    pool = pool_factory()
    executor = _RecordingExecutor()
    pool._executor = executor

    engine = _FakeEngine(pull_timeout=240.0)
    with pool._lock:
        pool._engines["raas-0"] = engine

    assert pool.pull_completed(max_items=8, timeout=0.1) == []
    assert executor.result_timeouts == [240.0 + _COLLECT_FUTURE_GRACE_SEC]
    assert engine.calls == [(8, 0.1)]


# ----------------------------------------------------------------------
# (3) Defaults are unchanged for a config that does not set them.
# ----------------------------------------------------------------------


def test_agent_config_defaults_are_unset_sentinels():
    """Unset knobs stay ``None`` so AstraDataAcquisition owns the defaults."""
    config = AgentConfig()
    assert config.raas_pull_timeout is None
    assert config.max_collect_per_tick is None
    assert config.collect_timeout is None


def test_service_with_default_config_keeps_historical_collect_behavior():
    """A YAML that sets none of the knobs must behave exactly as before."""
    config = ServiceConfig()
    service = AstraFlowService(config)
    try:
        assert service.raas_pool._pull_timeout is None

        service.register_agent("default", config.agent)
        acquisition = service.flows["default"].data_acquisition
        assert acquisition._max_collect_per_tick == _default_of(
            AstraDataAcquisition.__init__, "max_collect_per_tick"
        )
        assert acquisition._collect_timeout == _default_of(
            AstraDataAcquisition.__init__, "collect_timeout"
        )
        # Pin the historical literals so a drift is caught.
        assert acquisition._max_collect_per_tick == 512
        assert acquisition._collect_timeout == 0.1
    finally:
        service.raas_pool.shutdown()


def test_unset_pull_timeout_leaves_engine_behavior_identical():
    """``pull_timeout=None`` keeps the engine sharing ``request_timeout``."""
    engine = RaaS2InferenceEngine(service_url="http://localhost:1")
    assert engine.pull_timeout == engine.request_timeout == 10.0


def test_pool_default_pull_timeout_is_none(pool_factory):
    assert pool_factory()._pull_timeout is None


def test_astraflow_without_knobs_keeps_acquisition_defaults():
    class _Serving:
        buffer = None

        def put(self, *args, **kwargs):
            return True

    flow = AstraFlow(
        rollout=object(),
        rollout_dataloader=_DummyLoader(),
        workflow_spec={},
        data_serving=_Serving(),
    )
    assert flow.data_acquisition._max_collect_per_tick == _default_of(
        AstraDataAcquisition.__init__, "max_collect_per_tick"
    )
    assert flow.data_acquisition._collect_timeout == _default_of(
        AstraDataAcquisition.__init__, "collect_timeout"
    )


def test_collect_future_timeout_floor_is_preserved():
    """With no engine pull_timeout configured, the historical floor holds."""
    engines = [("raas-0", _FakeEngine(pull_timeout=None))]
    assert RaaSPool._collect_future_timeout(0.1, engines) == _COLLECT_FUTURE_MIN_SEC
    assert _COLLECT_FUTURE_MIN_SEC == 10.0
    # A long server-side long-poll still dominates, as before.
    assert RaaSPool._collect_future_timeout(30.0, engines) == 35.0


def test_default_engine_future_timeout_outlasts_its_http_timeout():
    """Deliberate behavior change: 10.0 -> 15.0 for an unconfigured engine.

    Waiting exactly as long as the engine's own HTTP timeout guaranteed the
    future lost the race, which is the data-loss bug.  The pool is now
    strictly more patient than the engine — never less.
    """
    engine = RaaS2InferenceEngine(service_url="http://localhost:1")
    collect = RaaSPool._collect_future_timeout(0.1, [("raas-0", engine)])
    assert collect == 15.0
    assert collect > engine.pull_timeout
