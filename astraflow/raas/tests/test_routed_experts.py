"""CPU-only tests for R3 routed-expert capture consumption on the RaaS side.

Covers:
1. ``SGLangBackend.parse_generation_response`` against canned base64 payloads:
   a flat int32 view of the bytes, ``None`` when the key is absent or the
   request was aborted before prefill.
2. The interrupt/resume accumulation in ``RemoteInfEngine.agenerate`` with a
   mocked server: three interrupted chunks' concatenated rows must equal one
   uninterrupted generation's rows (the off-by-one killer), the
   ``routed_experts_start_len`` handed to each iteration is the number of
   rows captured so far, and a short payload fails at the final reshape.
3. The launch-time guards in ``SGLangConfig.build_args``: an explicit
   capturing ``moe_runner_backend`` and a positive ``chunked_prefill_size``.
"""

import asyncio
import base64

import numpy as np
import pytest

import astraflow.raas.engine.remote_inf_engine as remote_inf_engine_mod
from astraflow.raas.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    SGLangConfig,
)
from astraflow.raas.api.io_struct import ModelRequest
from astraflow.raas.engine.remote_inf_engine import RemoteInfEngine
from astraflow.raas.engine.sglang_remote import SGLangBackend
from astraflow.raas.engine.vllm_remote import VLLMBackend

# Fake tiny MoE model: 4 decoder layers, top-2 routing, 8 logical experts.
NUM_LAYERS = 4
TOP_K = 2
NUM_COLUMNS = NUM_LAYERS * TOP_K
NUM_EXPERTS = 8


def expert_rows(positions: range) -> np.ndarray:
    """Deterministic ground-truth expert ids, ``[rows, num_layers, top_k]`` int32."""
    return np.array(
        [
            [
                [(pos + 3 * layer + k) % NUM_EXPERTS for k in range(TOP_K)]
                for layer in range(NUM_LAYERS)
            ]
            for pos in positions
        ],
        dtype=np.int32,
    ).reshape(len(positions), NUM_LAYERS, TOP_K)


def encode_experts(rows: np.ndarray) -> str:
    """Encode int32 expert rows the way SGLang does: base64 of LE int32 bytes."""
    return base64.b64encode(rows.astype("<i4").tobytes()).decode("ascii")


def make_response(
    output_tokens: list[int],
    finish_type: str = "stop",
    routed_experts: str | None = None,
    finish_message: str | None = None,
) -> dict:
    finish_reason: dict = {"type": finish_type}
    if finish_message is not None:
        finish_reason["message"] = finish_message
    meta_info = {
        "finish_reason": finish_reason,
        "output_token_logprobs": [[-0.1, tok] for tok in output_tokens],
    }
    if routed_experts is not None:
        meta_info["routed_experts"] = routed_experts
    return {"meta_info": meta_info}


# ---------------------------------------------------------------------------
# parse_generation_response
# ---------------------------------------------------------------------------


class TestParseGenerationResponse:
    def test_valid_payload_is_a_flat_int32_view(self):
        backend = SGLangBackend()
        rows = expert_rows(range(5))
        result = backend.parse_generation_response(
            make_response([7, 8, 9], routed_experts=encode_experts(rows))
        )
        assert result.routed_experts is not None
        assert result.routed_experts.dtype == np.int32
        assert result.routed_experts.shape == (5 * NUM_COLUMNS,)
        np.testing.assert_array_equal(result.routed_experts, rows.reshape(-1))
        assert result.output_tokens == [7, 8, 9]

    def test_missing_key_yields_none(self):
        backend = SGLangBackend()
        result = backend.parse_generation_response(make_response([7, 8, 9]))
        assert result.routed_experts is None

    def test_abort_before_prefill_yields_none(self):
        backend = SGLangBackend()
        result = backend.parse_generation_response(
            make_response(
                [],
                finish_type="abort",
                finish_message="Abort before prefill",
            )
        )
        assert result.output_tokens == []
        assert result.routed_experts is None

    def test_bytes_are_little_endian_int32(self):
        backend = SGLangBackend()
        raw = base64.b64encode(np.array([1, 256, -1], dtype="<i4").tobytes()).decode()
        result = backend.parse_generation_response(make_response([1], routed_experts=raw))
        assert result.routed_experts.tolist() == [1, 256, -1]


# ---------------------------------------------------------------------------
# build_generation_request
# ---------------------------------------------------------------------------


def test_sglang_payload_keys_present_when_enabled():
    backend = SGLangBackend()
    req = ModelRequest(
        input_ids=[1, 2, 3],
        gconfig=GenerationHyperparameters(return_routed_experts=True),
    )
    http_req = backend.build_generation_request(
        req, with_lora=False, routed_experts_start_len=7
    )
    assert http_req.payload["return_routed_experts"] is True
    assert http_req.payload["routed_experts_start_len"] == 7


def test_sglang_payload_keys_absent_when_disabled():
    backend = SGLangBackend()
    req = ModelRequest(input_ids=[1, 2, 3])
    http_req = backend.build_generation_request(req, with_lora=False)
    assert "return_routed_experts" not in http_req.payload
    assert "routed_experts_start_len" not in http_req.payload


def test_vllm_backend_rejects_routed_experts():
    req = ModelRequest(
        input_ids=[1, 2, 3],
        gconfig=GenerationHyperparameters(return_routed_experts=True),
    )
    with pytest.raises(NotImplementedError):
        VLLMBackend().build_generation_request(req, with_lora=False)


# ---------------------------------------------------------------------------
# Interrupt/resume accumulation (the off-by-one killer)
# ---------------------------------------------------------------------------


class FakeSGLangServer:
    """Simulates SGLang /generate with native routed-experts capture.

    Each call pops one ``(num_tokens, finish_type)`` step from the plan,
    extends the sequence, and returns expert rows for the half-open position
    range ``[routed_experts_start_len, new_total_len - 1)`` -- exactly the
    capture-side contract. ``capture=False`` models a server launched
    without ``enable_return_routed_experts``: it accepts the request keys and
    simply omits the payload.
    """

    def __init__(self, plan: list[tuple[int, str]], capture: bool = True):
        self.plan = list(plan)
        self.capture = capture
        self.next_token = 100
        self.start_lens: list[int] = []

    def rows_for(self, start_len: int, total_len: int) -> np.ndarray:
        return expert_rows(range(start_len, total_len - 1))

    async def __call__(
        self, session, addr, endpoint, payload, method, max_retries, timeout
    ):
        assert endpoint == "/generate"
        assert payload["return_routed_experts"] is True
        num_tokens, finish_type = self.plan.pop(0)
        start_len = payload["routed_experts_start_len"]
        self.start_lens.append(start_len)
        total_len = len(payload["input_ids"]) + num_tokens
        assert 0 <= start_len <= len(payload["input_ids"])
        output_tokens = list(range(self.next_token, self.next_token + num_tokens))
        self.next_token += num_tokens
        routed = None
        if self.capture:
            routed = encode_experts(self.rows_for(start_len, total_len))
        return make_response(output_tokens, finish_type=finish_type, routed_experts=routed)


PROMPT_LEN = 5


def run_agenerate(monkeypatch, server: FakeSGLangServer):
    engine = RemoteInfEngine(InferenceEngineConfig(), SGLangBackend())
    engine.addresses = ["fake-server:0"]
    monkeypatch.setattr(remote_inf_engine_mod, "arequest_with_retry", server)
    req = ModelRequest(
        input_ids=list(range(PROMPT_LEN)),
        gconfig=GenerationHyperparameters(
            max_new_tokens=100, return_routed_experts=True
        ),
    )
    return asyncio.run(engine.agenerate(req))


def test_interrupted_chunks_equal_uninterrupted(monkeypatch):
    total_output = 12

    plain = FakeSGLangServer([(total_output, "stop")])
    uninterrupted = run_agenerate(monkeypatch, plain)
    resumed = FakeSGLangServer([(3, "abort"), (4, "abort"), (5, "stop")])
    interrupted = run_agenerate(monkeypatch, resumed)

    assert uninterrupted.output_tokens == interrupted.output_tokens
    # start_len is 0 first, then len(input_ids) - 1 after every chunk.
    assert plain.start_lens == [0]
    assert resumed.start_lens == [0, PROMPT_LEN + 3 - 1, PROMPT_LEN + 3 + 4 - 1]

    num_rows = PROMPT_LEN + total_output - 1
    expected = expert_rows(range(num_rows)).reshape(num_rows, NUM_COLUMNS)
    for response in (uninterrupted, interrupted):
        assert response.output_routed_experts is not None
        assert response.output_routed_experts.dtype == np.int32
        assert response.output_routed_experts.shape == (num_rows, NUM_COLUMNS)
        np.testing.assert_array_equal(response.output_routed_experts, expected)
    np.testing.assert_array_equal(
        interrupted.output_routed_experts, uninterrupted.output_routed_experts
    )


def test_incomplete_capture_fails_at_the_reshape(monkeypatch):
    """A chunk whose rows do not cover all forwarded positions cannot be
    reshaped to one row per position: numpy raises, nothing is silently padded."""

    class TruncatingServer(FakeSGLangServer):
        def rows_for(self, start_len, total_len):
            return super().rows_for(start_len, total_len)[:-1]

    with pytest.raises(ValueError, match="reshape"):
        run_agenerate(monkeypatch, TruncatingServer([(6, "stop")]))


def test_server_that_does_not_capture_yields_none(monkeypatch):
    """No chunk ever arrives: the response carries None (the trainer's KeyError
    is the contract point for this case), nothing is invented."""
    response = run_agenerate(
        monkeypatch, FakeSGLangServer([(3, "abort"), (4, "stop")], capture=False)
    )
    assert response.output_tokens == list(range(100, 107))
    assert response.output_routed_experts is None


# ---------------------------------------------------------------------------
# Launch-time guards in build_args
# ---------------------------------------------------------------------------


def _build_args(**overrides):
    pytest.importorskip("sglang")
    config = SGLangConfig(model_path="dummy-model", **overrides)
    return SGLangConfig.build_args(sglang_config=config, tp_size=1, base_gpu_id=0)


class TestBuildArgsGuards:
    def test_capturing_backend_with_positive_chunked_prefill_is_accepted(self):
        args = _build_args(
            enable_return_routed_experts=True,
            moe_runner_backend="triton",
            chunked_prefill_size=32768,
        )
        assert args["enable_return_routed_experts"] is True
        assert args["moe_runner_backend"] == "triton"
        assert args["chunked_prefill_size"] == 32768

    @pytest.mark.parametrize(
        "backend",
        [
            None,  # sglang's "auto": flashinfer_trtllm on sm_100, which captures nothing
            "",  # dropped by get_py_cmd, so also "auto"
            "auto",
            "flashinfer_trtllm",
            "experimental_sgl_trtllm",
            "flashinfer_mxfp4",
            "triton_kernel",
        ],
    )
    def test_non_capturing_backend_is_rejected(self, backend):
        with pytest.raises(ValueError, match="explicit capturing"):
            _build_args(
                enable_return_routed_experts=True,
                moe_runner_backend=backend,
                chunked_prefill_size=32768,
            )

    @pytest.mark.parametrize("chunked_prefill_size", [-1, 0, None])
    def test_non_positive_chunked_prefill_is_rejected(self, chunked_prefill_size):
        """sglang sizes the capturer buffer from it; the AstraFlow default is -1."""
        assert SGLangConfig.chunked_prefill_size == -1
        with pytest.raises(ValueError, match="chunked_prefill_size > 0"):
            _build_args(
                enable_return_routed_experts=True,
                moe_runner_backend="triton",
                chunked_prefill_size=chunked_prefill_size,
            )

    def test_any_positive_chunked_prefill_is_preserved(self):
        args = _build_args(
            enable_return_routed_experts=True,
            moe_runner_backend="triton",
            chunked_prefill_size=4096,
            max_prefill_tokens=32768,
        )
        assert args["chunked_prefill_size"] == 4096

    def test_guards_are_inactive_without_r3(self):
        args = _build_args(moe_runner_backend="flashinfer_trtllm")
        assert args["enable_return_routed_experts"] is False
        assert args["moe_runner_backend"] == "flashinfer_trtllm"
        assert args["chunked_prefill_size"] == -1
