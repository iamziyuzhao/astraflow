"""Queued weight updates coalesce through the sender's newest buffer, and the
engine is labelled with the version the sender actually served.

At 30B one pull+pause+load cycle takes ~76 s and a training step can be
shorter, so per-step notify_version calls queue up behind the per-model
lock. The sender always serves its *latest* buffer, so the first queued
request that pulls gets newer weights than it asked for; labelling that load
with the *requested* version would stamp every rollout token with a version
that is too low and inflate measured staleness -- which a staleness cap then
acts on. Two things are pinned against the real manager methods with a
stubbed pull and a fake engine:

1. the engine is labelled with the served version
   (``requested v=N, sender served v=M``), and
2. the requests queued behind it then find their version already loaded
   after acquiring the lock and skip the pull.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from astraflow.raas.server import manager as manager_mod
from astraflow.raas.server.manager import RaaS3Manager


class _Engine:
    def __init__(self, load_seconds: float = 0.05):
        self.load_seconds = load_seconds
        self.loaded: list[str] = []
        self.versions: list[int] = []

    def pause_generation(self):
        pass

    def load_weights_from_path(self, path, use_lora=False):
        del use_lora
        time.sleep(self.load_seconds)
        self.loaded.append(path)

    def continue_generation(self):
        pass

    def set_version(self, v: int):
        self.versions.append(v)


def _manager(engine: _Engine, served_version: int | None) -> RaaS3Manager:
    """A bare manager: only the attributes the weight-update path touches.

    ``served_version`` is what the stubbed pull reports the sender served:
    an int, or ``None`` for a sender that does not report one (the requested
    version is then assumed).
    """
    m = object.__new__(RaaS3Manager)
    m._weight_versions = {}
    m._weight_update_locks = {}
    m._engines = {"model0": engine}
    m._eval_engines = {}
    m._metrics_cache = (0.0, [])
    m._metrics_cache_ok = False
    m._last_good_snapshot = None
    m._last_good_snapshot_at = 0.0
    m._weight_update_in_progress = False
    m._weight_update_started_at = 0.0
    m._engine_id = "test"  # _tag (print prefix) derives from it
    m.pulls: list[int] = []

    def _pull(endpoint, model_id="default"):
        del endpoint, model_id
        m.pulls.append(len(m.pulls) + 1)
        result = {
            "ok": True,
            "shm_path": f"/dev/shm/x{len(m.pulls)}",
            "use_lora": False,
        }
        if served_version is not None:
            result["version"] = served_version
        return result

    m._pull_weights_to_disk = _pull
    return m


def _run(coro):
    return asyncio.run(coro)


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def manager_log():
    handler = _ListHandler()
    logger = manager_mod._base_logger
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def _gather_three(m: RaaS3Manager):
    async def go():
        return await asyncio.gather(
            m.notify_version("model0", 1, "h:1"),
            m.notify_version("model0", 2, "h:1"),
            m.notify_version("model0", 3, "h:1"),
        )

    return _run(go())


def test_queued_requests_coalesce_to_the_version_the_sender_served(manager_log):
    """v1 pulls first; the sender's newest buffer is already v3, so v2 and
    v3 find their version loaded when they get the lock. One pull, not three."""
    engine = _Engine()
    m = _manager(engine, served_version=3)

    r1, r2, r3 = _gather_three(m)

    assert r1["ok"] and r1["version"] == 3 and r1.get("pull_result") is not None
    for r in (r2, r3):
        assert r["ok"] and r["pulled"] is False
        assert "(after lock)" in r["reason"]
    assert len(m.pulls) == 1
    assert m._weight_versions["model0"] == 3
    assert engine.versions == [3]
    assert engine.loaded == ["/dev/shm/x1"]

    assert any("requested v=1, sender served v=3" in msg for msg in manager_log.messages)
    skipped = [msg for msg in manager_log.messages if "already loaded" in msg and "after acquiring lock" in msg]
    assert len(skipped) == 2


def test_queued_requests_without_a_newer_buffer_load_in_order():
    """A sender that serves exactly what was asked coalesces nothing."""
    engine = _Engine()
    m = _manager(engine, served_version=None)

    r1, r2, r3 = _gather_three(m)

    assert [r["version"] for r in (r1, r2, r3)] == [1, 2, 3]
    assert all(r.get("pull_result") is not None for r in (r1, r2, r3))
    assert len(m.pulls) == 3
    assert m._weight_versions["model0"] == 3
    assert engine.versions == [1, 2, 3]


def test_a_served_version_never_lowers_the_label():
    engine = _Engine()
    m = _manager(engine, served_version=2)
    r = _run(m.notify_version("model0", 5, "h:1"))
    assert r["ok"] and r["version"] == 5
    assert m._weight_versions["model0"] == 5
    assert engine.versions == [5]


def test_an_already_loaded_version_is_not_pulled_again():
    engine = _Engine()
    m = _manager(engine, served_version=None)
    assert _run(m.notify_version("model0", 4, "h:1"))["ok"]
    r = _run(m.notify_version("model0", 4, "h:1"))
    assert r["pulled"] is False and "<= local" in r["reason"]
    assert len(m.pulls) == 1
