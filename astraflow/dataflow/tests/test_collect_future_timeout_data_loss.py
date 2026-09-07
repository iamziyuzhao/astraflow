"""Behavioral proof that the collect-future fix stops losing rollouts.

Companion to ``test_collect_tunables_plumbing.py``, which pins the config
plumbing and the arithmetic of ``RaaSPool._collect_future_timeout``.  This
file asserts the *consequence* instead: with the pre-fix formula a pull
that is merely slow — well inside the engine's own configured
``pull_timeout`` — has its future killed by the pool, and because
``RaaS3Manager._drain_completed`` pops results out of
``_completed_results`` before writing the response, every rollout in that
tick is gone for good and the instance is wrongly marked suspect.

The module constants are scaled down so the timing runs sub-second; only
the magnitudes change, not the shape of the arithmetic (whether the
engine's ``pull_timeout`` participates at all).

CPU-only, no services, no GPUs.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from astraflow.dataflow import raas_pool as raas_pool_mod
from astraflow.dataflow.raas_pool import RaaSPool

# The engine tolerates a 1.0s pull and answers in 0.4s — comfortably
# inside its own budget, and exactly the case the pool used to kill.
_ENGINE_PULL_TIMEOUT = 1.0
_ENGINE_DELAY = 0.4
_SCALED_MIN = 0.2
_SCALED_GRACE = 0.05


class _SlowEngine:
    """A registered engine whose pull is slow but well within its budget."""

    def __init__(self) -> None:
        self.pull_timeout = _ENGINE_PULL_TIMEOUT
        self.service_url = "http://localhost:1"
        self._initialized = True

    def pull_completed(
        self, max_items: int = 256, timeout: float = 0.0
    ) -> list[dict[str, Any]]:
        time.sleep(_ENGINE_DELAY)
        return [{"task_id": 1, "ok": True, "result": {}}]


def _legacy_collect_future_timeout(timeout, engines):
    """The pre-fix formula: a bare floor that ignores ``engine.pull_timeout``.

    Mirrors ``collect_timeout = max(timeout + 5.0, 10.0)`` from before the
    fix, expressed against the scaled constants so the test stays fast.
    """
    return max(
        timeout + raas_pool_mod._COLLECT_FUTURE_GRACE_SEC,
        raas_pool_mod._COLLECT_FUTURE_MIN_SEC,
    )


@pytest.fixture()
def pool(monkeypatch):
    monkeypatch.setattr(raas_pool_mod, "_COLLECT_FUTURE_MIN_SEC", _SCALED_MIN)
    monkeypatch.setattr(raas_pool_mod, "_COLLECT_FUTURE_GRACE_SEC", _SCALED_GRACE)

    pool = RaaSPool(heartbeat_interval=3600.0, pull_timeout=_ENGINE_PULL_TIMEOUT)
    # Stop the heartbeat thread (it would probe /health over HTTP against a
    # RaaS that does not exist) but keep the executor — pull_completed fans
    # out through it.  Its only tick ran against an empty pool.
    pool._stop_event.set()
    pool._suspect_event.set()
    with pool._lock:
        pool._engines["raas-0"] = _SlowEngine()
    try:
        yield pool
    finally:
        pool.shutdown()


def test_slow_pull_is_lost_under_the_legacy_future_timeout(pool, monkeypatch):
    """The bug: the pool's own future kills a slow-but-healthy pull."""
    monkeypatch.setattr(
        RaaSPool,
        "_collect_future_timeout",
        staticmethod(_legacy_collect_future_timeout),
    )

    results = pool.pull_completed(max_items=8, timeout=0.0)

    # The server already popped this rollout; nothing hands it back.
    assert results == []
    assert "raas-0" in pool._suspect_uids


def test_slow_pull_survives_with_the_derived_future_timeout(pool):
    """The fix: the future is sized from the engine's own pull timeout."""
    results = pool.pull_completed(max_items=8, timeout=0.0)

    assert len(results) == 1
    assert results[0]["_raas_uid"] == "raas-0"
    assert "raas-0" not in pool._suspect_uids
