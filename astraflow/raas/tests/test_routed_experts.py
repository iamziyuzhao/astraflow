"""CPU-only tests for R3 routed-expert capture consumption on the RaaS side.

Covers:
0. Model-identity discovery for engines that never launched a server (the
   eval engine): the backend asks the running server via ``/model_info``.
1. ``SGLangBackend.parse_generation_response`` against canned base64 payloads
   (valid / missing / corrupt-length / out-of-range must raise).
2. The interrupt/resume accumulation in ``RemoteInfEngine.agenerate`` with a
   mocked backend: three interrupted chunks' concatenated rows must equal one
   uninterrupted generation's rows (the off-by-one killer), using a fake
   model with L=4 MoE layers and top_k=2.
3. The chunked-prefill pairing guard in ``SGLangConfig.build_args``.
"""

import asyncio
import base64
import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

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
from astraflow.raas.engine.sglang_remote import (
    SGLangBackend,
    SGLangEngine,
    derive_moe_dims,
)
from astraflow.raas.engine.vllm_remote import VLLMBackend

# Fake tiny MoE model: 4 MoE layers, top-2 routing, 8 logical experts.
NUM_MOE_LAYERS = 4
TOP_K = 2
NUM_EXPERTS = 8


@pytest.fixture(scope="module")
def tiny_moe_model_dir(tmp_path_factory) -> str:
    """A local HF config dir describing a tiny Qwen3-MoE model."""
    model_dir = tmp_path_factory.mktemp("tiny_qwen3_moe")
    config = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": NUM_MOE_LAYERS,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": TOP_K,
        "mlp_only_layers": [],
        "decoder_sparse_step": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "moe_intermediate_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 128,
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    return str(model_dir)


def expert_rows(positions: range) -> np.ndarray:
    """Deterministic ground-truth expert ids for the given positions."""
    return np.array(
        [
            [
                [(pos + 3 * layer + k) % NUM_EXPERTS for k in range(TOP_K)]
                for layer in range(NUM_MOE_LAYERS)
            ]
            for pos in positions
        ],
        dtype=np.int32,
    ).reshape(len(positions), NUM_MOE_LAYERS, TOP_K)


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
    def test_valid_payload(self, tiny_moe_model_dir):
        backend = SGLangBackend(model_path=tiny_moe_model_dir)
        rows = expert_rows(range(5))
        result = backend.parse_generation_response(
            make_response([7, 8, 9], routed_experts=encode_experts(rows))
        )
        assert result.routed_experts is not None
        assert result.routed_experts.shape == (5, NUM_MOE_LAYERS, TOP_K)
        assert result.routed_experts.dtype == np.int16
        np.testing.assert_array_equal(result.routed_experts, rows)
        assert result.output_tokens == [7, 8, 9]
        # HF config was read and cached.
        assert backend._moe_dims == (NUM_MOE_LAYERS, TOP_K, NUM_EXPERTS)

    def test_missing_key_yields_none(self, tiny_moe_model_dir):
        backend = SGLangBackend(model_path=tiny_moe_model_dir)
        result = backend.parse_generation_response(make_response([7, 8, 9]))
        assert result.routed_experts is None

    def test_abort_before_prefill_yields_none(self, tiny_moe_model_dir):
        backend = SGLangBackend(model_path=tiny_moe_model_dir)
        result = backend.parse_generation_response(
            make_response(
                [],
                finish_type="abort",
                finish_message="Abort before prefill",
            )
        )
        assert result.output_tokens == []
        assert result.routed_experts is None

    def test_corrupt_length_raises(self, tiny_moe_model_dir):
        backend = SGLangBackend(model_path=tiny_moe_model_dir)
        # 7 int32 values: not divisible by NUM_MOE_LAYERS * TOP_K = 8.
        corrupt = base64.b64encode(np.arange(7, dtype="<i4").tobytes()).decode("ascii")
        with pytest.raises(ValueError, match="not divisible"):
            backend.parse_generation_response(
                make_response([7], routed_experts=corrupt)
            )

    def test_out_of_range_expert_id_raises(self, tiny_moe_model_dir):
        backend = SGLangBackend(model_path=tiny_moe_model_dir)
        rows = expert_rows(range(2))
        rows[0, 0, 0] = NUM_EXPERTS  # invalid logical expert id
        with pytest.raises(ValueError, match="expert ids outside"):
            backend.parse_generation_response(
                make_response([7], routed_experts=encode_experts(rows))
            )

    def test_unknown_model_path_raises(self):
        backend = SGLangBackend()
        rows = expert_rows(range(1))
        with pytest.raises(RuntimeError, match="model path"):
            backend.parse_generation_response(
                make_response([7], routed_experts=encode_experts(rows))
            )


def test_derive_moe_dims_respects_sparse_layers():
    config = SimpleNamespace(
        num_hidden_layers=8,
        num_experts=16,
        num_experts_per_tok=4,
        mlp_only_layers=[0, 1],
        decoder_sparse_step=2,
    )
    # MoE layers: idx not in {0,1} and (idx+1) % 2 == 0 -> {3, 5, 7}.
    assert derive_moe_dims(config) == (3, 4, 16)


def test_derive_moe_dims_rejects_dense_model():
    config = SimpleNamespace(num_hidden_layers=4)
    with pytest.raises(ValueError, match="MoE"):
        derive_moe_dims(config)


# ---------------------------------------------------------------------------
# build_generation_request
# ---------------------------------------------------------------------------


def test_sglang_payload_keys_present_when_enabled():
    pytest.importorskip("sglang")
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
    range ``[routed_experts_start_len, new_total_len - 1)`` — exactly the
    capture-side contract.
    """

    def __init__(self, plan: list[tuple[int, str]]):
        self.plan = list(plan)
        self.next_token = 100

    async def __call__(
        self, session, addr, endpoint, payload, method, max_retries, timeout
    ):
        assert endpoint == "/generate"
        assert payload["return_routed_experts"] is True
        num_tokens, finish_type = self.plan.pop(0)
        start_len = payload["routed_experts_start_len"]
        total_len = len(payload["input_ids"]) + num_tokens
        assert 0 <= start_len <= len(payload["input_ids"])
        output_tokens = list(range(self.next_token, self.next_token + num_tokens))
        self.next_token += num_tokens
        rows = expert_rows(range(start_len, total_len - 1))
        return make_response(
            output_tokens,
            finish_type=finish_type,
            routed_experts=encode_experts(rows),
        )


def run_agenerate(monkeypatch, tiny_moe_model_dir: str, plan: list[tuple[int, str]]):
    backend = SGLangBackend(model_path=tiny_moe_model_dir)
    engine = RemoteInfEngine(InferenceEngineConfig(), backend)
    engine.addresses = ["fake-server:0"]
    monkeypatch.setattr(
        remote_inf_engine_mod, "arequest_with_retry", FakeSGLangServer(plan)
    )
    req = ModelRequest(
        input_ids=list(range(5)),
        gconfig=GenerationHyperparameters(
            max_new_tokens=100, return_routed_experts=True
        ),
    )
    return asyncio.run(engine.agenerate(req))


def test_interrupted_chunks_equal_uninterrupted(monkeypatch, tiny_moe_model_dir):
    prompt_len = 5
    total_output = 12

    uninterrupted = run_agenerate(
        monkeypatch, tiny_moe_model_dir, plan=[(total_output, "stop")]
    )
    interrupted = run_agenerate(
        monkeypatch,
        tiny_moe_model_dir,
        plan=[(3, "abort"), (4, "abort"), (5, "stop")],
    )

    assert uninterrupted.output_tokens == interrupted.output_tokens
    expected = expert_rows(range(prompt_len + total_output - 1)).astype(np.int16)
    for response in (uninterrupted, interrupted):
        assert response.output_routed_experts is not None
        assert response.output_routed_experts.shape == (
            prompt_len + total_output - 1,
            NUM_MOE_LAYERS,
            TOP_K,
        )
        assert response.output_routed_experts.dtype == np.int16
        np.testing.assert_array_equal(response.output_routed_experts, expected)
    np.testing.assert_array_equal(
        interrupted.output_routed_experts, uninterrupted.output_routed_experts
    )


def test_incomplete_capture_fails_fast(monkeypatch, tiny_moe_model_dir):
    """A chunk whose rows do not cover all forwarded positions must raise."""

    class TruncatingServer(FakeSGLangServer):
        async def __call__(
            self, session, addr, endpoint, payload, method, max_retries, timeout
        ):
            response = await super().__call__(
                session, addr, endpoint, payload, method, max_retries, timeout
            )
            rows = np.frombuffer(
                base64.b64decode(response["meta_info"]["routed_experts"]),
                dtype="<i4",
            ).reshape(-1, NUM_MOE_LAYERS, TOP_K)
            response["meta_info"]["routed_experts"] = encode_experts(rows[:-1])
            return response

    backend = SGLangBackend(model_path=tiny_moe_model_dir)
    engine = RemoteInfEngine(InferenceEngineConfig(), backend)
    engine.addresses = ["fake-server:0"]
    monkeypatch.setattr(
        remote_inf_engine_mod, "arequest_with_retry", TruncatingServer([(6, "stop")])
    )
    req = ModelRequest(
        input_ids=list(range(5)),
        gconfig=GenerationHyperparameters(
            max_new_tokens=100, return_routed_experts=True
        ),
    )
    with pytest.raises(RuntimeError, match="routed-expert rows"):
        asyncio.run(engine.agenerate(req))


# ---------------------------------------------------------------------------
# Chunked-prefill pairing guard
# ---------------------------------------------------------------------------


class TestChunkedPrefillPairingGuard:
    def test_guard_fires_when_enabled(self):
        pytest.importorskip("sglang")
        config = SGLangConfig(
            model_path="dummy-model", enable_return_routed_experts=True
        )
        assert config.chunked_prefill_size == -1  # AstraFlow default
        args = SGLangConfig.build_args(sglang_config=config, tp_size=1, base_gpu_id=0)
        assert args["chunked_prefill_size"] == config.max_prefill_tokens
        assert args["enable_return_routed_experts"] is True

    def test_default_untouched_when_disabled(self):
        pytest.importorskip("sglang")
        config = SGLangConfig(model_path="dummy-model")
        args = SGLangConfig.build_args(sglang_config=config, tp_size=1, base_gpu_id=0)
        assert args["chunked_prefill_size"] == -1


# ---------------------------------------------------------------------------
# Model identity for engines that never launched a server (eval path)
# ---------------------------------------------------------------------------


class FakeSGLangHTTPServer:
    """Minimal stand-in for a running SGLang server: /health + model info."""

    def __init__(
        self,
        model_path: str | None,
        model_info_endpoints: tuple[str, ...] = ("/model_info", "/get_model_info"),
    ):
        self.model_path = model_path
        self.model_info_endpoints = set(model_info_endpoints)
        self.requested_paths: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
                outer.requested_paths.append(self.path)
                if self.path == "/health":
                    body = b"{}"
                elif self.path in outer.model_info_endpoints:
                    info = {"is_generation": True}
                    if outer.model_path is not None:
                        info["model_path"] = outer.model_path
                    body = json.dumps(info).encode()
                else:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # keep pytest output clean
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.addr = f"127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "FakeSGLangHTTPServer":
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def test_eval_style_engine_decodes_routed_experts(tiny_moe_model_dir):
    """An engine initialized with an existing address can decode.

    Regression: the eval engine is built as ``SGLangEngine(config)`` and
    initialized with the training engine's addresses, so ``launch_server``
    never runs and the served model used to be unknown — every eval response
    carrying routed experts raised.
    """
    with FakeSGLangHTTPServer(tiny_moe_model_dir) as server:
        engine = SGLangEngine(InferenceEngineConfig())
        try:
            engine.initialize(engine_id="eval", addr=[server.addr])
            backend = engine._backend
            # Resolved eagerly at initialize (keeps the blocking GET off the
            # rollout event loop).
            assert backend._model_path == tiny_moe_model_dir
            assert "/model_info" in server.requested_paths

            rows = expert_rows(range(4))
            result = engine._backend.parse_generation_response(
                make_response([1, 2, 3], routed_experts=encode_experts(rows))
            )
            assert result.routed_experts is not None
            np.testing.assert_array_equal(result.routed_experts, rows)
            assert backend._moe_dims == (NUM_MOE_LAYERS, TOP_K, NUM_EXPERTS)
        finally:
            engine.destroy()


def test_model_path_resolved_lazily_at_decode(tiny_moe_model_dir):
    """A backend bound to an engine resolves at first decode if not earlier."""
    with FakeSGLangHTTPServer(tiny_moe_model_dir) as server:
        backend = SGLangBackend()
        engine = RemoteInfEngine(InferenceEngineConfig(), backend)
        backend.bind_engine(engine)
        engine.addresses = [server.addr]
        assert backend._model_path is None

        rows = expert_rows(range(2))
        result = backend.parse_generation_response(
            make_response([1], routed_experts=encode_experts(rows))
        )
        np.testing.assert_array_equal(result.routed_experts, rows)
        assert backend._model_path == tiny_moe_model_dir


def test_model_path_falls_back_to_deprecated_endpoint(tiny_moe_model_dir):
    """Servers that only expose the deprecated /get_model_info still work."""
    with FakeSGLangHTTPServer(
        tiny_moe_model_dir, model_info_endpoints=("/get_model_info",)
    ) as server:
        backend = SGLangBackend()
        engine = RemoteInfEngine(InferenceEngineConfig(), backend)
        backend.bind_engine(engine)
        engine.addresses = [server.addr]
        assert backend.resolve_model_path() == tiny_moe_model_dir
        assert server.requested_paths == ["/model_info", "/get_model_info"]


def test_launched_model_path_wins_over_server_query(tiny_moe_model_dir):
    """An explicitly known model path is never overridden by a server query."""
    with FakeSGLangHTTPServer("/nonexistent/other-model") as server:
        backend = SGLangBackend(model_path=tiny_moe_model_dir)
        engine = RemoteInfEngine(InferenceEngineConfig(), backend)
        backend.bind_engine(engine)
        engine.addresses = [server.addr]
        assert backend.resolve_model_path() == tiny_moe_model_dir
        assert server.requested_paths == []


def test_unresolvable_model_path_fails_loudly():
    """A server that cannot name its model must raise, never decode garbage."""
    with FakeSGLangHTTPServer(None) as server:
        backend = SGLangBackend()
        engine = RemoteInfEngine(InferenceEngineConfig(), backend)
        backend.bind_engine(engine)
        engine.addresses = [server.addr]
        assert backend.resolve_model_path() is None
        with pytest.raises(RuntimeError, match="model path"):
            backend.parse_generation_response(
                make_response([1], routed_experts=encode_experts(expert_rows(range(1))))
            )


def test_engine_initialize_survives_unanswerable_model_info(tiny_moe_model_dir):
    """Eager resolution is best-effort: initialize must not fail on it."""
    with FakeSGLangHTTPServer(None, model_info_endpoints=()) as server:
        engine = SGLangEngine(InferenceEngineConfig())
        try:
            engine.initialize(engine_id="eval", addr=[server.addr])
            assert engine._backend._model_path is None
        finally:
            engine.destroy()


# ---------------------------------------------------------------------------
# derive_moe_dims key-spelling parity with the trainer side
# ---------------------------------------------------------------------------


def test_derive_moe_dims_accepts_num_local_experts():
    """transformers>=5 serializes Qwen3-MoE with ``num_local_experts``."""
    config = SimpleNamespace(
        num_hidden_layers=4,
        num_local_experts=16,
        num_experts_per_tok=4,
        mlp_only_layers=[],
        decoder_sparse_step=1,
    )
    assert derive_moe_dims(config) == (4, 4, 16)


def test_derive_moe_dims_accepts_n_routed_experts_and_moe_topk():
    config = SimpleNamespace(
        num_hidden_layers=6,
        n_routed_experts=32,
        moe_topk=6,
        mlp_only_layers=[],
        decoder_sparse_step=2,
    )
    assert derive_moe_dims(config) == (3, 6, 32)


def test_derive_moe_dims_rejects_unmodeled_dense_layer_placement():
    """DeepSeek-style dense-layer keys are not modeled: fail, do not guess."""
    config = SimpleNamespace(
        num_hidden_layers=8,
        n_routed_experts=64,
        num_experts_per_tok=8,
        first_k_dense_replace=3,
    )
    with pytest.raises(ValueError, match="first_k_dense_replace"):
        derive_moe_dims(config)

    config = SimpleNamespace(
        num_hidden_layers=8,
        num_experts=64,
        num_experts_per_tok=8,
        moe_layer_freq=[0, 0, 1, 1, 1, 1, 1, 1],
    )
    with pytest.raises(ValueError, match="moe_layer_freq"):
        derive_moe_dims(config)


def test_derive_moe_dims_matches_trainer_layer_rule():
    """The rollout MoE-layer count must equal the trainer's layer-map length.

    ``derive_moe_dims`` (here) and ``hf_moe_layer_indices`` (trainer) are two
    implementations of one topology rule; a config only one of them accepts —
    or that they read differently — breaks R3.
    """
    pytest.importorskip("torch")
    from astraflow.train_worker.utils.mcore.routing_replay import hf_moe_layer_indices

    for config in (
        SimpleNamespace(
            num_hidden_layers=8,
            num_experts=16,
            num_experts_per_tok=4,
            mlp_only_layers=[0, 1],
            decoder_sparse_step=2,
        ),
        SimpleNamespace(
            num_hidden_layers=4,
            num_local_experts=16,
            num_experts_per_tok=4,
            mlp_only_layers=[],
            decoder_sparse_step=1,
        ),
    ):
        assert derive_moe_dims(config)[0] == len(hf_moe_layer_indices(config))


# ---------------------------------------------------------------------------
# Launch-time validation of capture-hostile server flags
# ---------------------------------------------------------------------------


class TestPrefixCacheGuards:
    def test_radix_cache_rejected(self):
        pytest.importorskip("sglang")
        config = SGLangConfig(
            model_path="dummy-model",
            enable_return_routed_experts=True,
            disable_radix_cache=False,
        )
        with pytest.raises(ValueError, match="radix"):
            SGLangConfig.build_args(sglang_config=config, tp_size=1, base_gpu_id=0)

    def test_radix_cache_allowed_without_r3(self):
        pytest.importorskip("sglang")
        config = SGLangConfig(model_path="dummy-model", disable_radix_cache=False)
        args = SGLangConfig.build_args(sglang_config=config, tp_size=1, base_gpu_id=0)
        assert args["disable_radix_cache"] is False

    def test_hierarchical_cache_rejected(self):
        pytest.importorskip("sglang")

        @dataclass
        class _HiCacheSGLangConfig(SGLangConfig):
            enable_hierarchical_cache: bool = True

        config = _HiCacheSGLangConfig(
            model_path="dummy-model", enable_return_routed_experts=True
        )
        with pytest.raises(ValueError, match="hierarchical"):
            SGLangConfig.build_args(sglang_config=config, tp_size=1, base_gpu_id=0)

    def test_pd_disaggregation_rejected(self):
        pytest.importorskip("sglang")

        @dataclass
        class _DisaggSGLangConfig(SGLangConfig):
            disaggregation_mode: str = "prefill"

        config = _DisaggSGLangConfig(
            model_path="dummy-model", enable_return_routed_experts=True
        )
        with pytest.raises(ValueError, match="disaggregation"):
            SGLangConfig.build_args(sglang_config=config, tp_size=1, base_gpu_id=0)


class TestChunkedPrefillPolicy:
    def test_unset_value_is_forced(self):
        pytest.importorskip("sglang")
        config = SGLangConfig(
            model_path="dummy-model",
            enable_return_routed_experts=True,
            chunked_prefill_size=None,
        )
        args = SGLangConfig.build_args(sglang_config=config, tp_size=1, base_gpu_id=0)
        assert args["chunked_prefill_size"] == config.max_prefill_tokens

    def test_explicit_too_small_value_raises(self):
        pytest.importorskip("sglang")
        config = SGLangConfig(
            model_path="dummy-model",
            enable_return_routed_experts=True,
            chunked_prefill_size=4096,
            max_prefill_tokens=32768,
        )
        with pytest.raises(ValueError, match="chunked_prefill_size") as excinfo:
            SGLangConfig.build_args(sglang_config=config, tp_size=1, base_gpu_id=0)
        # Both knobs are named so the operator can resolve it either way.
        assert "max_prefill_tokens" in str(excinfo.value)
        assert "4096" in str(excinfo.value) and "32768" in str(excinfo.value)

    def test_explicit_large_enough_value_is_preserved(self):
        pytest.importorskip("sglang")
        config = SGLangConfig(
            model_path="dummy-model",
            enable_return_routed_experts=True,
            chunked_prefill_size=65536,
            max_prefill_tokens=32768,
        )
        args = SGLangConfig.build_args(sglang_config=config, tp_size=1, base_gpu_id=0)
        # The operator's memory/latency tuning survives R3.
        assert args["chunked_prefill_size"] == 65536

    def test_explicit_equal_value_is_preserved(self):
        pytest.importorskip("sglang")
        config = SGLangConfig(
            model_path="dummy-model",
            enable_return_routed_experts=True,
            chunked_prefill_size=32768,
            max_prefill_tokens=32768,
        )
        args = SGLangConfig.build_args(sglang_config=config, tp_size=1, base_gpu_id=0)
        assert args["chunked_prefill_size"] == 32768
