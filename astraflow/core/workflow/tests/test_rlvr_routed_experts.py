"""R3 routed-experts emission in the RLVR workflow.

The workflow's whole job here is layout: the engine hands back int32
``[seq_len - 1, num_hidden_layers * top_k]`` (one row per forwarded
position), and the trainer needs one row per ``input_ids`` position. The
rollout never forwards the final token, so its row is a copy of the last
recorded one (that position is loss-masked anyway). A response without a
payload emits no key at all: the trainer's ``KeyError`` is the single
contract point for a server that does not capture.

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
from astraflow.core.workflow.impl.rlvr import RLVRWorkflow

NUM_LAYERS = 4
TOP_K = 2
NUM_COLUMNS = NUM_LAYERS * TOP_K

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


def _make_workflow(n_samples: int, return_routed_experts: bool = True) -> RLVRWorkflow:
    gconfig = GenerationHyperparameters(
        n_samples=n_samples, return_routed_experts=return_routed_experts
    )
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
    """int32 ``[seq_len - 1, C]`` rows where row t is filled with value t."""
    rows = np.arange(seq_len - 1, dtype=np.int32).reshape(-1, 1)
    return np.broadcast_to(rows, (seq_len - 1, NUM_COLUMNS)).copy()


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


def _episode(wf: RLVRWorkflow, resps: list[ModelResponse], query_id: str):
    return asyncio.run(wf.arun_episode(_StubEngine(resps), {"query_id": query_id}))


def test_routed_experts_emission_aligned_with_seq():
    seq_len = len(PROMPT_IDS) + len(OUTPUT_IDS)
    resp = _make_response(_routed_rows(seq_len))
    wf = _make_workflow(n_samples=1)
    result = _episode(wf, [resp], "q0")

    assert result["n_trajs"] == 1
    res = result["trajectories"][0]["sequences"][0]

    routed = res["routed_experts"]
    assert routed.shape == (1, seq_len, NUM_COLUMNS)
    assert routed.dtype == torch.int32

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

    # The never-forwarded final position repeats the last recorded row.
    torch.testing.assert_close(routed[0, seq_len - 1], routed[0, seq_len - 2])
    assert torch.all(routed[0, seq_len - 1] == seq_len - 2)


def test_emitted_tensor_does_not_alias_the_response_array():
    seq_len = len(PROMPT_IDS) + len(OUTPUT_IDS)
    rows = _routed_rows(seq_len)
    wf = _make_workflow(n_samples=1)
    result = _episode(wf, [_make_response(rows)], "q1")
    routed = result["trajectories"][0]["sequences"][0]["routed_experts"]
    rows[:] = -1
    assert torch.all(routed >= 0)


@pytest.mark.parametrize("stop_reason", ["stop", "interrupt", "length"])
def test_missing_payload_emits_no_key_and_does_not_raise(stop_reason):
    """No per-sample drop, no request-side raise: the trainer's KeyError is
    the single contract point for a server that does not capture."""
    seq_len = len(PROMPT_IDS) + len(OUTPUT_IDS)
    with_payload = _make_response(_routed_rows(seq_len))
    without = _make_response(None, stop_reason=stop_reason)
    wf = _make_workflow(n_samples=2)
    result = _episode(wf, [with_payload, without], "q2")

    assert result["n_trajs"] == 2
    seqs = [t["sequences"][0] for t in result["trajectories"]]  # one trajectory per sample
    assert len(seqs) == 2
    assert "routed_experts" in seqs[0]
    assert "routed_experts" not in seqs[1]
    assert seqs[1]["input_ids"].shape == (1, seq_len)


def test_flag_off_emits_no_routed_experts():
    resp = _make_response(None)
    wf = _make_workflow(n_samples=1, return_routed_experts=False)
    result = _episode(wf, [resp], "q3")

    assert result["n_trajs"] == 1
    res = result["trajectories"][0]["sequences"][0]
    assert "routed_experts" not in res
    assert set(res) >= {"input_ids", "loss_mask", "logprobs", "versions", "attention_mask", "rewards"}
