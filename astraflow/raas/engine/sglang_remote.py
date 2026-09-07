import os
import shlex
import subprocess
import sys
import uuid
import weakref
from typing import Any

import numpy as np
import pybase64
import requests

from astraflow.raas.api.cli_args import InferenceEngineConfig, SGLangConfig
from astraflow.raas.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    ModelRequest,
)
from astraflow.raas.engine import RemoteInfEngine
from astraflow.raas.utils import logging, pkg_version, stats_tracker
from astraflow.raas.utils.launcher import TRITON_CACHE_PATH

logger = logging.getLogger(__name__)


# Accepted HF spellings for the expert count and the routing top-k.
#
# ``transformers>=5`` serializes Qwen3-MoE with ``num_local_experts`` (the
# ``num_experts`` name survives only as an ``attribute_map`` alias on the
# typed config class, so a plain/remote-code config object may expose just
# the serialized spelling), Mixtral has only ever used ``num_local_experts``,
# and DeepSeek-style configs use ``n_routed_experts``.
_NUM_EXPERTS_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")
_TOP_K_KEYS = (
    "num_experts_per_tok",
    "num_experts_per_token",
    "moe_topk",
    "moe_top_k",
    "num_selected_experts",
)


def _first_positive_int(hf_config: Any, keys: tuple[str, ...]) -> int | None:
    """Return the first positive int value among ``keys`` on ``hf_config``."""
    for key in keys:
        value = getattr(hf_config, key, None)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if value > 0:
            return value
    return None


def derive_moe_dims(hf_config: Any) -> tuple[int, int, int]:
    """Derive ``(num_moe_layers, top_k, num_experts)`` from an HF model config.

    Mirrors the HF Qwen-MoE rule: a decoder layer is MoE iff it is not in
    ``mlp_only_layers`` and ``(layer_idx + 1) % decoder_sparse_step == 0``.
    For Qwen3-MoE (empty ``mlp_only_layers``, ``decoder_sparse_step == 1``)
    every decoder layer is MoE.

    NOTE: this topology rule is duplicated on the trainer side by
    ``astraflow.train_worker.utils.mcore.routing_replay.hf_moe_layer_indices``,
    which turns the same HF config into the *list* of MoE layer indices used
    to build the megatron layer map. The rollout rows decoded here are packed
    along exactly that MoE-layer axis, so the two implementations must stay in
    sync — accepted key spellings included. A config one side accepts and the
    other rejects makes every rollout response raise; a config the two read
    differently would silently misalign layers.
    """
    num_experts = _first_positive_int(hf_config, _NUM_EXPERTS_KEYS)
    top_k = _first_positive_int(hf_config, _TOP_K_KEYS)
    if not num_experts or not top_k:
        raise ValueError(
            f"Model config {type(hf_config).__name__} does not describe a "
            f"MoE model (found none of {_NUM_EXPERTS_KEYS} / {_TOP_K_KEYS}); "
            "cannot decode routed experts."
        )
    # Topologies that place their dense layers with DeepSeek-style keys are
    # not modeled by the mlp_only_layers/decoder_sparse_step rule above (nor
    # by the trainer-side twin), and guessing would silently shift every MoE
    # layer ordinal. Fail loudly instead.
    # TODO(agent): support first_k_dense_replace / moe_layer_freq here *and*
    # in routing_replay.hf_moe_layer_indices, in one shared helper, if R3 is
    # ever run on a DeepSeek-style checkpoint.
    first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", None)
    moe_layer_freq = getattr(hf_config, "moe_layer_freq", None)
    if first_k_dense_replace or (moe_layer_freq is not None and moe_layer_freq != 1):
        raise ValueError(
            f"Model config {type(hf_config).__name__} places its MoE layers "
            f"with first_k_dense_replace={first_k_dense_replace!r} / "
            f"moe_layer_freq={moe_layer_freq!r}, which the Qwen-style "
            "mlp_only_layers/decoder_sparse_step rule used here (and by the "
            "trainer-side routing replay) does not model; refusing to guess "
            "the MoE layer count."
        )
    mlp_only_layers = getattr(hf_config, "mlp_only_layers", None) or []
    decoder_sparse_step = getattr(hf_config, "decoder_sparse_step", None) or 1
    num_moe_layers = sum(
        1
        for layer_idx in range(hf_config.num_hidden_layers)
        if layer_idx not in mlp_only_layers
        and (layer_idx + 1) % decoder_sparse_step == 0
    )
    if num_moe_layers == 0:
        raise ValueError(
            f"Model config {type(hf_config).__name__} has no MoE layers; "
            "cannot decode routed experts."
        )
    return num_moe_layers, top_k, num_experts


# Endpoints exposed by sglang that report the served model. ``/model_info``
# is the current name; ``/get_model_info`` is its deprecated alias, kept as a
# fallback for servers older than the rename.
MODEL_INFO_ENDPOINTS = ("/model_info", "/get_model_info")
MODEL_INFO_TIMEOUT = 10.0


class SGLangBackend:
    """Backend that translates engine operations into SGLang HTTP API calls."""

    def __init__(self, model_path: str | None = None):
        # The served model, needed to derive MoE dimensions when decoding
        # routed experts. Known from three sources, in priority order:
        # explicitly passed here, captured from server_args when this backend
        # launches the server, or — for servers this process did not launch
        # (eval engines, ASTRAFLOW_LLM_SERVER_ADDRS, name_resolve) — asked of
        # the running server itself via ``/model_info``.
        self._model_path = model_path
        self._moe_dims: tuple[int, int, int] | None = None
        self._engine_ref: weakref.ReferenceType | None = None

    def bind_engine(self, engine: Any) -> None:
        """Let the backend read server addresses off its owning engine.

        Held weakly: the engine owns the backend, so a strong reference back
        would make every engine an uncollectable cycle.
        """
        self._engine_ref = weakref.ref(engine)

    def _server_addresses(self) -> list[str]:
        engine = self._engine_ref() if self._engine_ref is not None else None
        return [addr for addr in (getattr(engine, "addresses", None) or []) if addr]

    def resolve_model_path(self) -> str | None:
        """Return the served model path, asking the servers if needed.

        Returns ``None`` (without raising) when no server can answer, so
        callers that resolve eagerly — before knowing whether routed experts
        will ever be requested — can treat it as best-effort. The result is
        cached; the query is a single localhost-ish GET.
        """
        if self._model_path is not None:
            return self._model_path
        addresses = self._server_addresses()
        if not addresses:
            logger.warning(
                "Cannot resolve the served model path: SGLangBackend has no "
                "server addresses yet."
            )
            return None
        failures: list[str] = []
        for addr in addresses:
            for endpoint in MODEL_INFO_ENDPOINTS:
                url = f"http://{addr}{endpoint}"
                try:
                    response = requests.get(url, timeout=MODEL_INFO_TIMEOUT)
                    response.raise_for_status()
                    model_path = response.json().get("model_path")
                except Exception as e:  # noqa: BLE001 - reported, then retried
                    failures.append(f"{url}: {type(e).__name__}: {e}")
                    continue
                if model_path:
                    logger.info(
                        "Resolved served model path %r from %s", model_path, url
                    )
                    self._model_path = model_path
                    return model_path
                failures.append(f"{url}: response carries no 'model_path'")
        logger.warning(
            "Could not resolve the served model path from any server: %s",
            "; ".join(failures),
        )
        return None

    def _get_moe_dims(self) -> tuple[int, int, int]:
        """Return cached ``(num_moe_layers, top_k, num_experts)`` of the served model."""
        if self._moe_dims is None:
            model_path = self.resolve_model_path()
            if model_path is None:
                raise RuntimeError(
                    "Cannot decode routed experts: the served model path is "
                    "unknown to SGLangBackend. It is captured from server_args "
                    "when this backend launches the server, and otherwise "
                    "queried from the running servers via "
                    f"{' or '.join(MODEL_INFO_ENDPOINTS)} (addresses tried: "
                    f"{self._server_addresses() or 'none'}). Check that the "
                    "servers are reachable, or pass model_path to "
                    "SGLangBackend."
                )
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            self._moe_dims = derive_moe_dims(hf_config)
        return self._moe_dims

    def build_generation_request(
        self, req: ModelRequest, with_lora: bool, routed_experts_start_len: int = 0
    ) -> HttpRequest:
        """Convert a ModelRequest into an SGLang /generate HTTP request."""
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids
        stop = gconfig.stop

        if gconfig.use_beam_search:
            raise NotImplementedError(
                "Currently Beam search is not supported in SGLang backend."
            )

        sample_params = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
            "ignore_eos": gconfig.ignore_eos,
            "skip_special_tokens": gconfig.skip_special_tokens,
            "frequency_penalty": gconfig.frequency_penalty,
        }
        if stop:
            sample_params["stop"] = stop

        payload = {
            "input_ids": req.input_ids.copy(),
            "image_data": req.image_data,
            "sampling_params": sample_params,
            "return_logprob": True,
            "stream": False,
        }

        if gconfig.return_routed_experts:
            if pkg_version.is_available(
                "sglang"
            ) and not pkg_version.is_version_greater_or_equal("sglang", "0.5.13"):
                raise RuntimeError(
                    "return_routed_experts requires sglang>=0.5.13 "
                    "(native routed-experts capture)."
                )
            payload["return_routed_experts"] = True
            payload["routed_experts_start_len"] = routed_experts_start_len

        if with_lora:
            payload["lora_path"] = "lora_1"

        return HttpRequest(endpoint="/generate", payload=payload)

    def _decode_routed_experts(self, encoded: str) -> np.ndarray:
        """Decode base64 little-endian int32 expert ids into int16 ``[rows, L, K]``.

        Uses ``pybase64`` (what SGLang encodes with) rather than stdlib
        ``base64``: this runs on the shared rollout event loop, and payloads
        reach ~8 MB per sequence, where the stdlib decoder costs ~6x more.
        """
        flat = np.frombuffer(pybase64.b64decode(encoded), dtype=np.dtype("<i4"))
        num_moe_layers, top_k, num_experts = self._get_moe_dims()
        row_elems = num_moe_layers * top_k
        if flat.size % row_elems != 0:
            raise ValueError(
                f"Corrupt routed_experts payload: {flat.size} int32 values "
                f"is not divisible by num_moe_layers * top_k = {row_elems}."
            )
        if flat.size > 0 and (flat.min() < 0 or flat.max() >= num_experts):
            raise ValueError(
                f"Corrupt routed_experts payload: expert ids outside "
                f"[0, {num_experts})."
            )
        return flat.reshape(-1, num_moe_layers, top_k).astype(np.int16)

    def parse_generation_response(
        self, response: dict[str, Any]
    ) -> HttpGenerationResult:
        """Extract tokens, logprobs, and stop reason from an SGLang response."""
        meta_info = response["meta_info"]
        finish_reason = meta_info["finish_reason"]
        stop_reason = finish_reason["type"]
        stop_message = finish_reason.get("message", "")
        if stop_reason == "abort" and stop_message.startswith("Abort before prefill"):
            return HttpGenerationResult(
                output_tokens=[],
                output_logprobs=[],
                stop_reason=stop_reason,
                routed_experts=None,
            )

        output_tokens = [x[1] for x in meta_info["output_token_logprobs"]]
        output_logprobs = [x[0] for x in meta_info["output_token_logprobs"]]

        routed_experts = None
        encoded_experts = meta_info.get("routed_experts")
        if encoded_experts is not None:
            routed_experts = self._decode_routed_experts(encoded_experts)

        return HttpGenerationResult(
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            stop_reason=stop_reason,
            routed_experts=routed_experts,
        )

    def get_pause_request(self) -> HttpRequest:
        """Build request to pause generation on the server."""
        return HttpRequest(endpoint="/pause_generation", payload={})

    def get_resume_request(self) -> HttpRequest:
        """Build request to resume generation on the server."""
        return HttpRequest(endpoint="/continue_generation", payload={})

    def get_health_check_request(self) -> HttpRequest:
        """Build request to check server health."""
        return HttpRequest(endpoint="/health", payload={}, method="GET")

    def get_metrics_request(self) -> HttpRequest:
        """Build request to fetch Prometheus metrics from sglang.

        sglang serves ``/metrics`` directly from its multiprocess Prometheus
        registry, which is backed by mmap files in
        ``$PROMETHEUS_MULTIPROC_DIR``. The HTTP handler reads those files
        in-process without any ZMQ round-trip to the scheduler, so this
        endpoint stays responsive even when the scheduler is saturated.

        Replaces the old ``/get_load`` path which required a scheduler RPC
        and would time out under heavy generation load. Requires sglang to
        be launched with ``--enable-metrics`` (the default in this repo via
        ``SGLangConfig.enable_metrics=True``).
        """
        return HttpRequest(endpoint="/metrics", payload={}, method="GET")

    def get_offload_request(self) -> HttpRequest:
        """Build request to offload model from GPU to CPU."""
        return HttpRequest(endpoint="/release_memory_occupation", payload={})

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        """Build request to reload model from CPU to GPU."""
        payload = {"tags": tags} if tags is not None else {}
        return HttpRequest(endpoint="/resume_memory_occupation", payload=payload)

    def launch_server(self, server_args: dict[str, Any]) -> subprocess.Popen:
        """Spawn an SGLang server subprocess and return its Popen handle."""
        if server_args.get("model_path"):
            self._model_path = server_args["model_path"]
        launch_env = server_args.pop("__launch_env__", None)
        autopatch = (launch_env or {}).get("ASTRAFLOW_AUTOPATCH", "false").lower() in (
            "true",
            "1",
        )
        if autopatch:
            from astraflow.raas.api.cli_args import get_py_cmd

            cmd = get_py_cmd("astraflow.raas.entrypoint", server_args)
        else:
            cmd = SGLangConfig.build_cmd_from_args(server_args)
        logger.info(
            "Launching SGLang server command: %s", shlex.join([str(c) for c in cmd])
        )

        _env = os.environ.copy()
        triton_cache_path = _env.get("TRITON_CACHE_PATH", TRITON_CACHE_PATH)
        _env["TRITON_CACHE_PATH"] = os.path.join(triton_cache_path, str(uuid.uuid4()))
        if launch_env:
            _env.update({str(k): str(v) for k, v in launch_env.items()})

        return subprocess.Popen(
            cmd,
            env=_env,
            stdout=sys.stdout,
            stderr=sys.stdout,
        )


class SGLangEngine:
    """Inference engine backed by remote SGLang servers.

    All methods are delegated to the underlying RemoteInfEngine via __getattr__.
    """

    def __init__(self, config: InferenceEngineConfig):
        self.config = config
        self._backend = SGLangBackend()
        self._engine = RemoteInfEngine(config, self._backend)
        # The backend needs the server addresses to ask a server it did not
        # launch which model it serves (routed-expert decoding).
        self._backend.bind_engine(self._engine)

    def __getattr__(self, name: str):
        return getattr(self._engine, name)

    def initialize(self, *args: Any, **kwargs: Any):
        """Initialize the engine, then resolve the served model eagerly.

        Resolving here (rather than only lazily at the first routed-expert
        decode) keeps the blocking HTTP query off the rollout event loop and
        surfaces an unreachable/odd server at startup. Best-effort: a failure
        is logged, not raised, because most runs never ask for routed experts
        — the lazy path in ``SGLangBackend._get_moe_dims`` retries and raises
        for the runs that do.
        """
        result = self._engine.initialize(*args, **kwargs)
        self._backend.resolve_model_path()
        return result

    def export_stats(self) -> dict[str, float]:
        """Export workflow execution statistics without distributed reduction."""
        return stats_tracker.export_all(reduce_group=None)
