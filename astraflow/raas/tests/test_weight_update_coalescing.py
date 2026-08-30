"""Queued weight updates coalesce to the newest version and the engine is
labelled with the version the sender actually served.

At 30B one pull+pause+load cycle takes ~76 s and a training step can be
shorter, so per-step notify_version calls queue up behind the per-model
lock. Loading every queued version in turn puts the engine a version
further behind the trainer on every step; labelling each load with the
*requested* version, while the sender serves its *latest* buffer, stamps
every rollout token with a version that is too low and inflates measured
staleness -- which a staleness cap then acts on. Both are exercised here
against the real manager methods with a stubbed pull and a fake engine.
"""

from __future__ import annotations

import asyncio
import time

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


def _manager(engine: _Engine, served_version) -> RaaS3Manager:
    """A bare manager: only the attributes the weight-update path touches.

    ``served_version`` is what the stubbed pull reports the sender served:
    an int, or ``None`` to echo the requested version.
    """
    m = object.__new__(RaaS3Manager)
    m._weight_versions = {}
    m._weight_update_locks = {}
    m._newest_requested_version = {}
    m._engines = {"model0": engine}
    m._eval_engines = {}
    m._metrics_cache = (0.0, [])
    m._metrics_cache_ok = False
    m._last_good_snapshot = None
    m._last_good_snapshot_at = 0.0
    m._weight_update_in_progress = False
    m._weight_update_started_at = 0.0
    m._engine_id = "test"
    m.pulls: list[int] = []

    def _pull(endpoint, model_id="default"):
        del endpoint, model_id
        m.pulls.append(len(m.pulls) + 1)
        v = served_version if served_version is not None else None
        result = {"ok": True, "shm_path": f"/dev/shm/x{len(m.pulls)}", "use_lora": False}
        if v is not None:
            result["version"] = v
        return result

    m._pull_weights_to_disk = _pull
    return m


def _run(coro):
    return asyncio.run(coro)


def test_queued_requests_coalesce_to_the_newest_version():
    engine = _Engine()
    m = _manager(engine, served_version=None)

    async def go():
        return await asyncio.gather(
            m.notify_version("model0", 1, "h:1"),
            m.notify_version("model0", 2, "h:1"),
            m.notify_version("model0", 3, "h:1"),
        )

    r1, r2, r3 = _run(go())
    # v1 held the lock first and loaded; v2 was superseded by v3 while it
    # queued; v3 loaded. Two pulls, not three.
    assert r1["ok"] and r1.get("pull_result") is not None
    assert r2 == {"ok": True, "model_id": "model0", "pulled": False,
                  "reason": "version=2 superseded by 3"}
    assert r3["ok"] and r3.get("pull_result") is not None
    assert len(m.pulls) == 2
    assert m._weight_versions["model0"] == 3
    assert engine.versions == [1, 3]


def test_engine_is_labelled_with_the_version_the_sender_served():
    """The request asked for v1; the sender's latest buffer was v3."""
    engine = _Engine()
    m = _manager(engine, served_version=3)

    async def go():
        return await asyncio.gather(
            m.notify_version("model0", 1, "h:1"),
            m.notify_version("model0", 2, "h:1"),
            m.notify_version("model0", 3, "h:1"),
        )

    r1, r2, r3 = _run(go())
    assert r1["ok"] and r1["version"] == 3
    # After v1's load labelled the engine v3, v2 and v3 are already loaded.
    assert r2["pulled"] is False and "superseded" in r2["reason"]
    assert r3["pulled"] is False and "<= local" in r3["reason"]
    assert len(m.pulls) == 1
    assert m._weight_versions["model0"] == 3
    assert engine.versions == [3]


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
