import asyncio
import os
import random
import uuid
from collections.abc import Callable
from typing import Any

import aiofiles
import aiofiles.os
import colorama
import torch
from transformers import PreTrainedTokenizerFast

from astraflow.core.workflow.api.cli_args import GenerationHyperparameters
from astraflow.core.workflow.api.engine_api import InferenceEngine
from astraflow.core.workflow.api.io_struct import ModelRequest, ModelResponse
from astraflow.core.workflow.api.reward_api import AsyncRewardWrapper
from astraflow.core.workflow.api.workflow_api import RolloutWorkflow
from astraflow.core.workflow.registry import register_workflow
from astraflow.core.workflow.utils import logging, stats_tracker
from astraflow.core.workflow.utils.data import resolve_prompt_id, results_to_structured
from astraflow.core.workflow.utils.dynamic_import import import_from_string
from astraflow.core.workflow.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)

logger = logging.getLogger("RLVR workflow")


def default_get_input_ids_fn(
    data: Any,
    tokenizer: PreTrainedTokenizerFast,
    enable_thinking: bool,
) -> list[int]:
    input_ids = tokenizer.apply_chat_template(
        data,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_dict=False,
    )
    return list(input_ids)


SOLVER_PROMPT_SUFFIX = (
    "\nLet's think step by step. Please put your final answer within \\boxed{}."
)


def default_data_extract_prompt_fn(data: dict[str, Any]) -> Any:
    messages = data["messages"]
    # Append solve instruction suffix to user messages
    return [
        {**m, "content": m["content"] + SOLVER_PROMPT_SUFFIX}
        if m["role"] == "user"
        else m
        for m in messages
    ]


# R3: stop reasons that mark a request the rollout engine killed mid-flight.
# Such a response may legitimately carry no routed-experts record — nothing
# recordable was ever forwarded — so it is dropped per-sample instead of
# failing the whole run.
ABORTED_STOP_REASONS = frozenset({"abort", "interrupt"})


def _routing_miss_is_per_sample(resp: ModelResponse) -> bool:
    """Whether an absent routed-experts payload is this one sample's accident.

    A response that was aborted/interrupted, or that generated no tokens at
    all, never ran a forward the rollout engine could have recorded, so an
    empty record says nothing about how the server was configured. Any other
    response — one that completed normally and produced tokens — coming back
    with no payload whatsoever means the engine never captured routing for
    *any* request, which is systemic rather than per-sample.
    """
    return resp.stop_reason in ABORTED_STOP_REASONS or resp.output_len == 0


def _missing_routing_payload_error(resp: ModelResponse) -> RuntimeError:
    """Build the fail-fast error for a systemically absent routing payload."""
    return RuntimeError(
        "R3 is enabled on this workflow "
        "(GenerationHyperparameters.return_routed_experts=True) but the "
        "rollout engine returned a normally-completed response carrying no "
        "routed-experts payload at all (stop_reason="
        f"{resp.stop_reason!r}, output_len={resp.output_len}). The SGLang "
        "server was very likely launched WITHOUT "
        "enable_return_routed_experts: such a server still accepts "
        "return_routed_experts on the request and answers successfully, but "
        "omits meta_info['routed_experts'] from the reply. Every sample would "
        "then be silently dropped, the rollout buffer would never fill, and "
        "training would hang until the batch timeout with a traceback "
        "pointing nowhere near the missing flag. Relaunch the inference "
        "server with SGLangConfig.enable_return_routed_experts=True "
        "(requires sglang>=0.5.13), or set return_routed_experts=False to "
        "train without routing replay."
    )


@register_workflow("rlvr")
class RLVRWorkflow(RolloutWorkflow):
    """Single-turn reward learning workflow supporting optional thinking tokens."""

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        get_input_ids_fn: Callable[
            [Any, PreTrainedTokenizerFast, bool], list[int]
        ] = default_get_input_ids_fn,
        data_extract_prompt_fn: Callable[
            [dict[str, Any]], Any
        ] = default_data_extract_prompt_fn,
    ):
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            from astraflow.core.workflow.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(self.tokenizer)
            self.tokenizer = tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking
        self.dump_dir = dump_dir
        self.rollout_stat_scope = rollout_stat_scope
        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        self.get_input_ids_fn = get_input_ids_fn
        self.data_extract_prompt_fn = data_extract_prompt_fn
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    def _build_routed_experts(
        self, resp: ModelResponse, seq_len: int
    ) -> torch.Tensor | None:
        """Build the per-position routed-experts tensor for one sequence.

        Returns an int16 tensor of shape ``[seq_len, num_moe_layers, top_k]``:
        rows 0..seq_len-2 are the expert ids recorded by the rollout engine
        and row seq_len-1 is synthetic (the final position is never forwarded
        during rollout; its output receives no loss gradient). Returns None
        when *this* response carries no usable record — an aborted request, or
        a row-count/shape mismatch — such samples must be dropped, never
        zero-filled.

        Raises
        ------
        RuntimeError
            When a normally-completed response carries no routing payload at
            all. That is a systemic misconfiguration (almost always an SGLang
            server launched without ``enable_return_routed_experts``), not a
            per-sample accident, and dropping such samples would empty the
            rollout buffer and hang training for an hour.
        """
        recorded = resp.output_routed_experts
        if recorded is None:
            if not _routing_miss_is_per_sample(resp):
                # TODO(agent): the complete fix is a pre-flight capability
                # handshake — when return_routed_experts is requested, have
                # the inference engine query the server (e.g. /get_server_info)
                # for enable_return_routed_experts and refuse to start the
                # rollout at all. That check belongs in raas/engine/, not in a
                # workflow, so this fails on the first response received rather
                # than before the first request is sent.
                raise _missing_routing_payload_error(resp)
            logger.warning(
                "return_routed_experts is enabled but this response has no "
                f"routed experts (stop_reason={resp.stop_reason}, "
                f"output_len={resp.output_len}); dropping sample."
            )
            return None
        if recorded.ndim != 3 or recorded.shape[0] != seq_len - 1:
            logger.warning(
                f"Routed experts shape {tuple(recorded.shape)} does not match "
                f"expected [{seq_len - 1}, num_moe_layers, top_k]; dropping sample."
            )
            return None
        recorded_t = torch.tensor(recorded, dtype=torch.int16)
        num_moe_layers, top_k = recorded_t.shape[1], recorded_t.shape[2]
        # Synthetic row for the never-forwarded final position. The model
        # config (num_experts) is not available in this workflow, but
        # arange(top_k) % num_experts == arange(top_k) for every valid config
        # since top_k <= num_experts, so ids 0..top_k-1 are always legal
        # logical expert ids.
        final_row = torch.arange(top_k, dtype=torch.int16).expand(
            1, num_moe_layers, top_k
        )
        return torch.cat([recorded_t, final_row], dim=0)

    @trace_session("reward")
    async def _compute_rewards(
        self,
        resp: ModelResponse,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[float, str]:
        completions_str = self.tokenizer.decode(resp.output_tokens)
        reward = await self.async_reward_fn(
            prompt_str,
            completions_str,
            resp.input_tokens,
            resp.output_tokens,
            **task_data,
        )

        return reward, completions_str

    @session_context()
    async def _collect_samples(
        self,
        engine: InferenceEngine,
        req: ModelRequest,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[ModelResponse, float, str]:
        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)

        reward, completions_str = await self._compute_rewards(
            resp, prompt_str, task_data
        )

        stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)

        return resp, reward, completions_str

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        # NOTE: load reward function dynamically if given as string
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        input_ids = self.get_input_ids_fn(
            self.data_extract_prompt_fn(data),
            self.tokenizer,
            self.enable_thinking,
        )
        n_samples = self.gconfig.n_samples
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )

        version = engine.get_version()
        prompt_str = self.tokenizer.decode(input_ids)
        prompt_strs = [prompt_str] * n_samples

        # Generate responses and collect rewards
        sample_results = await asyncio.gather(
            *[
                self._collect_samples(engine, req, prompt_str, data)
                for _ in range(n_samples)
            ]
        )
        if sample_results:
            resps, rewards, completions_strs = map(list, zip(*sample_results))
        else:
            resps, rewards, completions_strs = [], [], []

        # Build result tensors
        results = []
        for resp, reward in zip(resps, rewards):
            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions

            res = {
                "input_ids": torch.tensor(seq, dtype=torch.int32),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
                "logprobs": torch.tensor(logprobs, dtype=torch.float32),
                "versions": torch.tensor(versions, dtype=torch.int32),
                "attention_mask": torch.ones(len(seq), dtype=torch.bool),
                "rewards": torch.tensor(reward, dtype=torch.float32),
            }
            if self.gconfig.return_routed_experts:
                routed_experts = self._build_routed_experts(resp, len(seq))
                if routed_experts is None:
                    continue
                res["routed_experts"] = routed_experts
            res = {k: v.unsqueeze(0) for k, v in res.items()}
            results.append(res)

        # save rollout to file with a 1 in 128 random chance
        if self.dump_dir is not None and random.random() < 1 / 128:
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)

            # Get the unique identifier for this prompt
            qid = resolve_prompt_id(data) or uuid.uuid4().hex

            # Dump rollout to file
            file_path = os.path.join(dump_path, f"{qid}.txt")
            seqlens = [
                len(resp.input_tokens) + len(resp.output_tokens) for resp in resps
            ]
            async with aiofiles.open(file_path, "a") as f:
                answer = data.get("answer")
                if answer is not None:
                    await f.write(
                        "answer is: "
                        f"{colorama.Fore.YELLOW + colorama.Style.DIM}{answer}{colorama.Style.RESET_ALL}\n"
                    )
                for i, (prompt, completion, reward, seqlen) in enumerate(
                    zip(prompt_strs, completions_strs, rewards, seqlens)
                ):
                    info_lines = [
                        f"idx: {i + 1} / {n_samples}, seqlen: {seqlen}, reward is {reward}.",
                        f"prompt is \n{colorama.Fore.YELLOW + colorama.Style.DIM}{prompt}{colorama.Style.RESET_ALL}",
                        f"sequence is: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{completion}{colorama.Style.RESET_ALL}",
                    ]
                    info = "\n".join(info_lines)
                    await f.write(info + "\n")

        return results_to_structured(results, prompt_id=resolve_prompt_id(data))
