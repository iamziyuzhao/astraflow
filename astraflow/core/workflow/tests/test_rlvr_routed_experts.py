"""Tests for R3 routed-experts emission in the RLVR workflow.

Run:
    pytest astraflow/core/workflow/tests/test_rlvr_routed_experts.py -v
"""

import asyncio
from typing import Any

import numpy as np
import pytest
import torch

from astraflow.core.workflow.api.cli_args import GenerationHyperparameters
from astraflow.core.workflow.api.io_struct import ModelResponse
from astraflow.core.workflow.impl import rlvr as rlvr_mod
from astraflow.core.workflow.impl.rlvr import RLVRWorkflow

NUM_MOE_LAYERS = 4
TOP_K = 2

PROMPT_IDS = [5, 6, 7]
OUTPUT_IDS = [8, 9]


class _StubTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        return " ".join(str(i) for i in ids)


class _StubEngine:
    def __init__(self, resps: list[ModelResponse]):
        self._resps = list(resps)

    def get_version(self) -> int:
        return 0

    async def agenerate(self, req: Any) -> ModelResponse:
        return self._resps.pop(0)


def _reward_fn(*args: Any, **kwargs: Any) -> float:
    return 1.0


async def _stub_async_reward(*args: Any, **kwargs: Any) -> float:
    return 1.0


def _make_workflow(n_samples: int) -> RLVRWorkflow:
    gconfig = GenerationHyperparameters(n_samples=n_samples, return_routed_experts=True)
    wf = RLVRWorkflow(
        reward_fn=_reward_fn,
        gconfig=gconfig,
        tokenizer=_StubTokenizer(),
        get_input_ids_fn=lambda data, tokenizer, enable_thinking: list(PROMPT_IDS),
        data_extract_prompt_fn=lambda data: data,
    )
    # Avoid spinning up the reward ProcessPoolExecutor in unit tests.
    wf.async_reward_fn = _stub_async_reward
    return wf


def _routed_rows(seq_len: int) -> np.ndarray:
    """[seq_len - 1, L, K] int16 rows where row t is filled with value t."""
    rows = np.arange(seq_len - 1, dtype=np.int16).reshape(-1, 1, 1)
    return np.broadcast_to(rows, (seq_len - 1, NUM_MOE_LAYERS, TOP_K)).copy()


def _make_response(
    routed: np.ndarray | None,
    stop_reason: str = "stop",
    output_tokens: list[int] | None = None,
) -> ModelResponse:
    output_tokens = list(OUTPUT_IDS) if output_tokens is None else list(output_tokens)
    return ModelResponse(
        input_tokens=list(PROMPT_IDS),
        output_tokens=output_tokens,
        output_logprobs=[-0.1 * (i + 1) for i in range(len(output_tokens))],
        output_versions=[0] * len(output_tokens),
        output_routed_experts=routed,
        stop_reason=stop_reason,
    )


class _RecordingLogger:
    """Stand-in for the module logger; collects warning messages."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.warnings.append(msg % args if args else msg)

    def __getattr__(self, _name: str) -> Any:
        return lambda *args, **kwargs: None


@pytest.fixture
def warn_log(monkeypatch: pytest.MonkeyPatch) -> _RecordingLogger:
    """Capture rlvr's warnings without fighting its custom logging config."""
    recorder = _RecordingLogger()
    monkeypatch.setattr(rlvr_mod, "logger", recorder)
    return recorder


def test_routed_experts_emission_aligned_with_seq():
    seq_len = len(PROMPT_IDS) + len(OUTPUT_IDS)
    resp = _make_response(_routed_rows(seq_len))
    wf = _make_workflow(n_samples=1)
    result = asyncio.run(wf.arun_episode(_StubEngine([resp]), {"query_id": "q0"}))

    assert result["n_trajs"] == 1
    res = result["trajectories"][0]["sequences"][0]

    routed = res["routed_experts"]
    assert routed.shape == (1, seq_len, NUM_MOE_LAYERS, TOP_K)
    assert routed.dtype == torch.int16

    # Alignment with the seq/logprobs layout: [1, seq_len] rows built as
    # prompt + output concat.
    assert res["input_ids"].shape == (1, seq_len)
    torch.testing.assert_close(
        res["input_ids"][0], torch.tensor(PROMPT_IDS + OUTPUT_IDS, dtype=torch.int32)
    )
    torch.testing.assert_close(
        res["logprobs"][0],
        torch.tensor([0.0] * len(PROMPT_IDS) + [-0.1, -0.2], dtype=torch.float32),
    )

    # Position t of routed_experts is the record for the forward that consumed
    # token t (rows were filled with value t).
    for t in range(seq_len - 1):
        assert torch.all(routed[0, t] == t), f"row {t} misaligned"

    # Synthetic final row: arange(top_k) per MoE layer.
    expected_final = torch.arange(TOP_K, dtype=torch.int16).expand(
        NUM_MOE_LAYERS, TOP_K
    )
    torch.testing.assert_close(routed[0, seq_len - 1], expected_final)


def test_missing_routed_experts_drops_sample(warn_log):
    seq_len = len(PROMPT_IDS) + len(OUTPUT_IDS)
    good = _make_response(_routed_rows(seq_len))
    aborted = _make_response(None, stop_reason="interrupt")
    wf = _make_workflow(n_samples=2)
    result = asyncio.run(
        wf.arun_episode(_StubEngine([good, aborted]), {"query_id": "q1"})
    )

    # Only the sample with a routing record survives; nothing is zero-filled.
    assert result["n_trajs"] == 1
    res = result["trajectories"][0]["sequences"][0]
    assert res["routed_experts"].shape == (1, seq_len, NUM_MOE_LAYERS, TOP_K)
    assert torch.any(res["routed_experts"] != 0)
    assert any("dropping sample" in w for w in warn_log.warnings)


def test_no_routing_payload_at_all_raises_naming_both_flags():
    """Server launched without enable_return_routed_experts: fail fast.

    Such a server accepts ``return_routed_experts`` on the request, answers
    successfully, and simply omits ``meta_info['routed_experts']``. Silently
    dropping every sample empties the rollout buffer and hangs training until
    the batch timeout, so the very first such response must raise.
    """
    seq_len = len(PROMPT_IDS) + len(OUTPUT_IDS)
    good = _make_response(_routed_rows(seq_len))
    no_routing = _make_response(None)  # normal completion, no payload
    wf = _make_workflow(n_samples=2)

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(
            wf.arun_episode(_StubEngine([good, no_routing]), {"query_id": "q4"})
        )

    msg = str(excinfo.value)
    # Both flag names must appear: the request-side one and the server-side one.
    assert "GenerationHyperparameters.return_routed_experts" in msg, msg
    assert "enable_return_routed_experts" in msg, msg
    assert "sglang" in msg.lower(), msg


def test_no_routing_payload_raises_on_the_first_bad_response():
    """The raise happens on the first offending response, not after the batch."""
    seen = _make_response(None)
    never_generated = _make_response(_routed_rows(5))
    engine = _StubEngine([seen, never_generated])
    wf = _make_workflow(n_samples=1)

    with pytest.raises(RuntimeError):
        asyncio.run(wf.arun_episode(engine, {"query_id": "q5"}))

    # The second stub response was never consumed by this single-sample episode.
    assert len(engine._resps) == 1


def test_aborted_response_drops_sample_without_raising(warn_log):
    """An aborted/interrupted request stays a per-sample drop."""
    for stop_reason in ("abort", "interrupt"):
        wf = _make_workflow(n_samples=1)
        aborted = _make_response(None, stop_reason=stop_reason)
        result = asyncio.run(
            wf.arun_episode(_StubEngine([aborted]), {"query_id": "q6"})
        )
        assert result["n_trajs"] == 0, stop_reason
        assert result["trajectories"] == []

    assert len(warn_log.warnings) == 2
    assert all("dropping sample" in w for w in warn_log.warnings)


def test_empty_generation_drops_sample_without_raising(warn_log):
    """A response that generated nothing could not have recorded routing."""
    wf = _make_workflow(n_samples=1)
    empty = _make_response(None, output_tokens=[])
    result = asyncio.run(wf.arun_episode(_StubEngine([empty]), {"query_id": "q7"}))

    assert result["n_trajs"] == 0
    assert result["trajectories"] == []
    assert any("dropping sample" in w for w in warn_log.warnings)


def test_mismatched_routed_experts_rowcount_drops_sample(warn_log):
    seq_len = len(PROMPT_IDS) + len(OUTPUT_IDS)
    bad = _make_response(_routed_rows(seq_len)[:-1])  # one row short
    wf = _make_workflow(n_samples=1)
    result = asyncio.run(wf.arun_episode(_StubEngine([bad]), {"query_id": "q2"}))

    assert result["n_trajs"] == 0
    assert result["trajectories"] == []
    assert any("does not match" in w for w in warn_log.warnings)


def test_mismatched_routed_experts_ndim_drops_sample(warn_log):
    """A 2D payload is unusable for this sample but says nothing systemic."""
    seq_len = len(PROMPT_IDS) + len(OUTPUT_IDS)
    bad = _make_response(_routed_rows(seq_len).reshape(seq_len - 1, -1))
    wf = _make_workflow(n_samples=1)
    result = asyncio.run(wf.arun_episode(_StubEngine([bad]), {"query_id": "q8"}))

    assert result["n_trajs"] == 0
    assert any("does not match" in w for w in warn_log.warnings)


def test_flag_off_emits_no_routed_experts():
    resp = _make_response(None)
    gconfig = GenerationHyperparameters(n_samples=1)
    wf = RLVRWorkflow(
        reward_fn=_reward_fn,
        gconfig=gconfig,
        tokenizer=_StubTokenizer(),
        get_input_ids_fn=lambda data, tokenizer, enable_thinking: list(PROMPT_IDS),
        data_extract_prompt_fn=lambda data: data,
    )
    wf.async_reward_fn = _stub_async_reward
    result = asyncio.run(wf.arun_episode(_StubEngine([resp]), {"query_id": "q3"}))

    assert result["n_trajs"] == 1
    res = result["trajectories"][0]["sequences"][0]
    assert "routed_experts" not in res
