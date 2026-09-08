"""Regression tests for the shared weight-sync stall budget.

The weight-sync waits were widened for a 30B MoE (a full sync takes
76-130 s; the old 60 s was sized for ~8B dense models). Pinned here:

1. the value is defined once (``WEIGHT_SYNC_TIMEOUT_SEC``), the trainer-side
   waits reference it, and the RaaS-side grace window is kept in step with
   it, so they cannot drift apart;
2. a missed ``buffer_ready`` ack and a late delta each warn (once, with the
   budget in the message) and carry on -- the double-buffer flip still
   happens, nothing raises.

NOTE ON LOCATION: these cover ``astraflow/core/weight_manager`` and
``astraflow/raas/server``, not dataflow.  They live here because
``astraflow/dataflow/tests/`` is the test location this change was scoped
to; they would sit more naturally in
``astraflow/core/weight_manager/tests/``.

All CPU, no GPUs, no subprocesses.
"""

from __future__ import annotations

import inspect
import logging
import queue
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import astraflow.core.weight_manager.weight_manager as weight_manager_mod  # noqa: E402
from astraflow.core.weight_manager.weight_manager import (  # noqa: E402
    WEIGHT_SYNC_TIMEOUT_SEC,
    WeightManager,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
WEIGHT_MANAGER_PY = (
    REPO_ROOT / "astraflow" / "core" / "weight_manager" / "weight_manager.py"
)
RAAS_MANAGER_PY = REPO_ROOT / "astraflow" / "raas" / "server" / "manager.py"


# ---------------------------------------------------------------------------
# 1. One definition, referenced everywhere
# ---------------------------------------------------------------------------


def test_shared_budget_has_the_expected_value():
    assert WEIGHT_SYNC_TIMEOUT_SEC == 300.0


def test_wait_delta_ready_default_references_the_shared_budget():
    default = (
        inspect.signature(WeightManager.wait_delta_ready).parameters["timeout"].default
    )
    assert default is WEIGHT_SYNC_TIMEOUT_SEC


def test_raas_grace_window_is_kept_in_step_with_the_shared_budget():
    """The RaaS-side monitor's grace window must not lag the trainer's budget:
    a legitimate 30B pull would otherwise be force-probed mid-update."""
    manager_mod = pytest.importorskip("astraflow.raas.server.manager")
    assert manager_mod.RaaS3Manager._WEIGHT_UPDATE_GRACE_SEC == WEIGHT_SYNC_TIMEOUT_SEC


@pytest.mark.parametrize(
    "source_path", [WEIGHT_MANAGER_PY, RAAS_MANAGER_PY], ids=lambda p: p.name
)
def test_no_bare_300_literal_remains(source_path):
    """Guard against a future edit re-introducing an uncoupled ``300.0``.

    Only the definitions of ``WEIGHT_SYNC_TIMEOUT_SEC`` and of the RaaS grace
    window (kept in step by the test above) may spell the number; anything
    else must reference the name.
    """
    assert source_path.exists(), source_path
    offenders = []
    for lineno, line in enumerate(source_path.read_text().splitlines(), start=1):
        code = line.split("#", 1)[0]
        if "=" in code and (
            "WEIGHT_SYNC_TIMEOUT_SEC" in code or "_WEIGHT_UPDATE_GRACE_SEC" in code
        ):
            continue  # the definitions
        if re.search(r"(?<![\w.])300\.0(?![\w.])", code):
            offenders.append(f"{source_path.name}:{lineno}: {line.strip()}")
    assert not offenders, (
        "re-literalled weight-sync timeout; reference "
        "WEIGHT_SYNC_TIMEOUT_SEC instead:\n" + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# 2. Expiry paths warn with the budget and carry on
# ---------------------------------------------------------------------------


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)

    def messages(self, level: int) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno == level]


@pytest.fixture
def wm_log():
    """Capture the weight manager's logger directly (no reliance on propagation)."""
    handler = _ListHandler()
    logger = weight_manager_mod.logger
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


class _StubInputQueue:
    def __init__(self):
        self.messages = []

    def put(self, msg):
        self.messages.append(msg)


class _StubOutputQueue:
    """Returns queued acks; raises ``queue.Empty`` once they run out."""

    def __init__(self, acks=()):
        self._acks = list(acks)
        self.timeouts: list[float | None] = []

    def get(self, timeout=None):
        self.timeouts.append(timeout)
        if not self._acks:
            raise queue.Empty
        return self._acks.pop(0)


def _make_weight_manager(acks=()) -> WeightManager:
    """A WeightManager wired only for ``_notify_buffer_ready``.

    ``__init__`` starts a sender subprocess and allocates shared memory,
    neither of which this path needs, so the object is built directly.
    """
    wm = WeightManager.__new__(WeightManager)
    wm._local_rank = 0
    wm._sender_process = object()  # only asserted non-None
    wm._input_queue = _StubInputQueue()
    wm._output_queue = _StubOutputQueue(acks)
    wm._delta_done_event = None
    wm._inactive_buf_idx = 0
    return wm


def test_buffer_ready_waits_the_shared_budget_for_the_ack():
    wm = _make_weight_manager(acks=[{"status": "ok"}])
    assert wm._notify_buffer_ready(3) == {"status": "ok"}
    assert wm._output_queue.timeouts == [WEIGHT_SYNC_TIMEOUT_SEC]
    assert wm._input_queue.messages == ["buffer_ready:3:0"]


def test_missed_ack_warns_with_the_budget_and_flips_the_buffer_half(wm_log):
    wm = _make_weight_manager()
    assert wm._inactive_buf_idx == 0

    assert wm._notify_buffer_ready(0) is None

    assert wm._inactive_buf_idx == 1
    warnings = wm_log.messages(logging.WARNING)
    assert len(warnings) == 1
    assert "did not acknowledge" in warnings[0]
    assert f"{WEIGHT_SYNC_TIMEOUT_SEC:.0f}s" in warnings[0]


def test_repeated_misses_keep_warning_without_raising(wm_log):
    """No escalation: every miss is one warning and one buffer flip."""
    wm = _make_weight_manager()
    for version in range(4):
        assert wm._notify_buffer_ready(version) is None
        assert wm._inactive_buf_idx == (version + 1) % 2
    assert len(wm_log.messages(logging.WARNING)) == 4


def test_late_delta_warns_with_the_budget_and_carries_on(wm_log):
    class _NeverSetEvent:
        def __init__(self):
            self.timeouts: list[float] = []

        def wait(self, timeout=None):
            self.timeouts.append(timeout)
            return False

    wm = _make_weight_manager()
    event = _NeverSetEvent()
    wm._delta_done_event = event

    wm.wait_delta_ready()  # default timeout: the shared budget

    assert event.timeouts == [WEIGHT_SYNC_TIMEOUT_SEC]
    warnings = wm_log.messages(logging.WARNING)
    assert len(warnings) == 1
    assert "delta not ready" in warnings[0]
    assert f"{WEIGHT_SYNC_TIMEOUT_SEC:.0f}s" in warnings[0]
