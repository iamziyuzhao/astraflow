"""Tunable pull timeout / collect tick size for large R3 pull payloads."""

from __future__ import annotations

import time
from typing import Any

from astraflow.dataflow.data_acquisition import AstraDataAcquisition
from astraflow.dataflow.raas2_engine import (
    MAX_REQUEST_TIMEOUT_SEC,
    RaaS2InferenceEngine,
    dumps_object,
)


class _DummyLoader:
    sampler = None

    def __iter__(self):
        yield []


def _publish(batch, metadata, timeout):
    del batch, metadata, timeout
    return True


def test_pull_timeout_defaults_to_request_timeout():
    engine = RaaS2InferenceEngine(service_url="http://localhost:1", request_timeout=7.0)
    assert engine.pull_timeout == 7.0


def test_pull_timeout_capped_at_max():
    engine = RaaS2InferenceEngine(service_url="http://localhost:1", pull_timeout=1e9)
    assert engine.pull_timeout == MAX_REQUEST_TIMEOUT_SEC


def test_pull_completed_uses_pull_timeout(monkeypatch):
    engine = RaaS2InferenceEngine(
        service_url="http://localhost:1",
        request_timeout=7.0,
        pull_timeout=123.0,
    )
    assert engine.pull_timeout == 123.0

    captured: dict[str, Any] = {}

    class _Response:
        status_code = 200
        content = dumps_object({"ok": True, "result": []})

    def _fake_post(url, data=None, timeout=None, headers=None):
        captured["url"] = url
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(
        "astraflow.dataflow.raas2_engine.requests.post",
        _fake_post,
    )
    items = engine.pull_completed(max_items=4, timeout=0.5)
    assert items == []
    assert captured["url"].endswith("/pull")
    assert captured["timeout"] == 123.0


def test_data_acquisition_collect_knob_defaults():
    acquisition = AstraDataAcquisition(
        rollout=object(),
        rollout_dataloader=_DummyLoader(),
        workflow_spec={},
        publish_fn=_publish,
    )
    assert acquisition._max_collect_per_tick == 512
    assert acquisition._collect_timeout == 0.1


def test_data_acquisition_collect_knobs_plumbed_to_pull():
    class _Rollout:
        def __init__(self):
            self.pull_calls: list[tuple[int, float]] = []

        def get_raas_availability(self):
            return {"available": 0}

        def submit_auto(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("no availability, submit_auto must not be called")

        def pull_completed(self, max_items: int = 256, timeout: float = 0.0):
            self.pull_calls.append((max_items, timeout))
            return []

    rollout = _Rollout()
    acquisition = AstraDataAcquisition(
        rollout=rollout,
        rollout_dataloader=_DummyLoader(),
        workflow_spec={},
        publish_fn=_publish,
        max_collect_per_tick=17,
        collect_timeout=0.25,
    )
    acquisition.start()
    try:
        deadline = time.time() + 5.0
        while not rollout.pull_calls and time.time() < deadline:
            time.sleep(0.02)
    finally:
        acquisition.stop(timeout=5.0)

    assert rollout.pull_calls, "collect loop never called pull_completed"
    assert rollout.pull_calls[0] == (17, 0.25)
