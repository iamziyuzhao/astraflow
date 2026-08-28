"""Regression tests for the shared weight-sync stall budget.

This branch widened the weight-sync timeouts 3-5x (RaaS-side grace window
90 -> 300, ``buffer_ready`` ack 60 -> 300, ``wait_delta_ready`` default
60 -> 300) by writing the literal ``300.0`` into four separate files
coupled only by "keep in sync" comments, and every expiry path still only
warned and carried on.  Two things are pinned here:

1. the value is defined once (``WEIGHT_SYNC_TIMEOUT_SEC``) and the other
   sites reference it, so they cannot drift apart;
2. a missed ``buffer_ready`` ack escalates to a hard failure after a
   bounded number of consecutive misses instead of warning forever while
   the trainer keeps producing versions RaaS will never receive.

NOTE ON LOCATION: these cover ``astraflow/core/weight_manager`` and
``astraflow/raas/server``, not dataflow.  They live here because
``astraflow/dataflow/tests/`` is the test location this change was scoped
to; they would sit more naturally in
``astraflow/core/weight_manager/tests/``.

All CPU, no GPUs, no subprocesses.
"""

from __future__ import annotations

import inspect
import queue
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

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


def test_buffer_ready_ack_timeout_references_the_shared_budget():
    assert WeightManager._BUFFER_READY_ACK_TIMEOUT_SEC is WEIGHT_SYNC_TIMEOUT_SEC


def test_wait_delta_ready_default_references_the_shared_budget():
    default = (
        inspect.signature(WeightManager.wait_delta_ready).parameters["timeout"].default
    )
    assert default is WEIGHT_SYNC_TIMEOUT_SEC


def test_raas_grace_window_references_the_shared_budget():
    """The RaaS-side monitor must use the same budget object, not a copy."""
    manager_mod = pytest.importorskip("astraflow.raas.server.manager")
    assert manager_mod.RaaS3Manager._WEIGHT_UPDATE_GRACE_SEC is WEIGHT_SYNC_TIMEOUT_SEC


@pytest.mark.parametrize(
    "source_path", [WEIGHT_MANAGER_PY, RAAS_MANAGER_PY], ids=lambda p: p.name
)
def test_no_bare_300_literal_remains(source_path):
    """Guard against a future edit re-introducing an uncoupled ``300.0``.

    Only the single definition of ``WEIGHT_SYNC_TIMEOUT_SEC`` may spell
    the number; anything else must reference the name.
    """
    assert source_path.exists(), source_path
    offenders = []
    for lineno, line in enumerate(source_path.read_text().splitlines(), start=1):
        code = line.split("#", 1)[0]
        if "WEIGHT_SYNC_TIMEOUT_SEC" in code and "=" in code:
            continue  # the one definition
        if re.search(r"(?<![\w.])300\.0(?![\w.])", code):
            offenders.append(f"{source_path.name}:{lineno}: {line.strip()}")
    assert not offenders, (
        "re-literalled weight-sync timeout; reference "
        "WEIGHT_SYNC_TIMEOUT_SEC instead:\n" + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# 2. A missed buffer_ready ack escalates instead of warning forever
# ---------------------------------------------------------------------------


class _StubInputQueue:
    def __init__(self):
        self.messages = []

    def put(self, msg):
        self.messages.append(msg)


class _StubOutputQueue:
    """Returns queued acks; raises ``queue.Empty`` once they run out."""

    def __init__(self, acks=()):
        self._acks = list(acks)

    def get(self, timeout=None):
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
    wm._consecutive_ack_timeouts = 0
    return wm


def test_ack_timeouts_below_the_limit_only_warn():
    wm = _make_weight_manager()
    limit = WeightManager._MAX_CONSECUTIVE_ACK_TIMEOUTS
    assert limit >= 2

    for version in range(limit - 1):
        assert wm._notify_buffer_ready(version) is None

    assert wm._consecutive_ack_timeouts == limit - 1


def test_ack_timeouts_escalate_at_the_limit():
    """Warning forever would stall the trainer ~5 min/step, silently."""
    wm = _make_weight_manager()
    limit = WeightManager._MAX_CONSECUTIVE_ACK_TIMEOUTS

    for version in range(limit - 1):
        wm._notify_buffer_ready(version)

    with pytest.raises(RuntimeError, match="presumed dead"):
        wm._notify_buffer_ready(limit - 1)


def test_successful_ack_resets_the_escalation_counter():
    """A slow-but-alive agent must never accumulate toward the limit."""
    limit = WeightManager._MAX_CONSECUTIVE_ACK_TIMEOUTS
    # limit-1 misses, then a real ack, then limit-1 misses again.
    wm = WeightManager.__new__(WeightManager)
    wm._local_rank = 0
    wm._sender_process = object()
    wm._input_queue = _StubInputQueue()
    wm._delta_done_event = None
    wm._inactive_buf_idx = 0
    wm._consecutive_ack_timeouts = 0

    class _AlternatingQueue:
        def __init__(self):
            self.calls = 0

        def get(self, timeout=None):
            self.calls += 1
            # The limit-th call succeeds; every other call times out.
            if self.calls == limit:
                return {"status": "ok"}
            raise queue.Empty

    wm._output_queue = _AlternatingQueue()

    for version in range(limit - 1):
        wm._notify_buffer_ready(version)
    assert wm._consecutive_ack_timeouts == limit - 1

    # The successful ack clears the counter ...
    assert wm._notify_buffer_ready(limit - 1) == {"status": "ok"}
    assert wm._consecutive_ack_timeouts == 0

    # ... so the next run of misses starts over rather than tripping.
    for version in range(limit - 1):
        wm._notify_buffer_ready(version)
    assert wm._consecutive_ack_timeouts == limit - 1


def test_buffer_half_still_flips_on_a_tolerated_miss():
    """The existing double-buffer flip is unchanged for a single miss."""
    wm = _make_weight_manager()
    assert wm._inactive_buf_idx == 0
    wm._notify_buffer_ready(0)
    assert wm._inactive_buf_idx == 1
