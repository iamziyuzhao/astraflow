"""R3 acceptance property on a tiny HF Qwen3-MoE model (CPU only).

Record the top-k expert choices made by ``Qwen3MoeForCausalLM``'s own routers
on a random input, then re-run the model with the routers replaced by
forced-routing reimplementations that are handed those recorded indices.
Replaying the model's own choices must be a no-op:

- with the HF router math (gather the fp32 softmax at the recorded indices
  and renormalize), the outputs are bit-identical;
- with the shipped Megatron replay path -- the recorded ids packed into the
  ``[tokens, num_hidden_layers * top_k]`` int32 contract tensor, installed
  into a real :class:`RoutingReplayContext`, and served to the real patched
  ``TopKRouter.forward`` (fp32 masked-fill softmax) -- the outputs match to
  numerical tolerance (the two formulations are mathematically identical,
  proven exactly in ``test_routing_replay_math.py``).

The HF model serves as the reference router stack; the megatron router is
driven with a duck-typed ``self`` because a real one needs process groups.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")

from astraflow.train_worker.utils.mcore.routing_replay import (  # noqa: E402
    RoutingReplayContext,
    install_topk_router_patch,
)

TINY_MODEL_PATH = Path("/home/haizhonz/albz/models/tiny-qwen3moe")

pytestmark = pytest.mark.skipif(
    not TINY_MODEL_PATH.exists(),
    reason=f"tiny Qwen3-MoE checkpoint not found at {TINY_MODEL_PATH}",
)


@pytest.fixture(scope="module")
def model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(TINY_MODEL_PATH))
    return model.float().eval()


@pytest.fixture(scope="module")
def input_ids(model):
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, model.config.vocab_size, (1, 32), generator=generator)


def _sparse_blocks(model):
    blocks = [
        (layer_idx, layer.mlp)
        for layer_idx, layer in enumerate(model.model.layers)
        if hasattr(layer.mlp, "gate")
    ]
    assert blocks, "tiny model has no sparse MoE blocks"
    return blocks


def _num_experts(model) -> int:
    num_experts = getattr(model.config, "num_experts", None)
    if num_experts is None:
        num_experts = model.config.num_local_experts
    return num_experts


def _run(model, input_ids, blocks, router_overrides=None):
    """Forward the model, returning (logits, per-layer MoE block outputs)."""
    originals = {layer_idx: mlp.gate.forward for layer_idx, mlp in blocks}
    layer_outputs: dict[int, torch.Tensor] = {}
    hooks = [
        mlp.register_forward_hook(
            lambda module, args, output, _i=layer_idx: layer_outputs.__setitem__(
                _i, output.detach().clone()
            )
        )
        for layer_idx, mlp in blocks
    ]
    try:
        if router_overrides is not None:
            for layer_idx, mlp in blocks:
                mlp.gate.forward = router_overrides[layer_idx]
        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits
    finally:
        for hook in hooks:
            hook.remove()
        for layer_idx, mlp in blocks:
            mlp.gate.forward = originals[layer_idx]
    return logits, layer_outputs


def _recording_router(gate, sink: dict, layer_idx: int):
    original = gate.forward

    def forward(hidden_states):
        router_logits, router_scores, router_indices = original(hidden_states)
        sink[layer_idx] = router_indices.detach().clone()
        return router_logits, router_scores, router_indices

    return forward


def _forced_router_hf_math(gate, forced_indices: torch.Tensor):
    """HF ``Qwen3MoeTopKRouter`` math with the top-k choice forced."""

    def forward(hidden_states):
        hidden_states = hidden_states.reshape(-1, gate.hidden_dim)
        router_logits = F.linear(hidden_states, gate.weight)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value = router_probs.gather(-1, forced_indices)
        if gate.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(
                dim=-1, keepdim=True
            )
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, forced_indices

    return forward


class _PatchTarget:
    """Duck-typed megatron ``TopKRouter`` over the HF gate weight."""

    def __init__(self, gate, layer_number: int):
        self.gate = gate
        self.layer_number = layer_number
        self.config = SimpleNamespace(num_moe_experts=gate.weight.shape[0])

    def _maintain_float32_expert_bias(self):
        pass

    def apply_input_jitter(self, x):
        return x

    def gating(self, x):
        return F.linear(x, self.gate.weight)


def _forced_router_real_patch(gate, layer_idx: int, forced_indices, patched_forward):
    """The shipped replay path: real patched forward fed by the real context."""
    target = _PatchTarget(gate, layer_number=layer_idx + 1)

    def forward(hidden_states):
        hidden_states = hidden_states.reshape(-1, gate.hidden_dim)
        probs, routing_map = patched_forward(target, hidden_states)
        # the context served exactly the recorded set for this layer
        assert torch.equal(
            routing_map, torch.zeros_like(routing_map).scatter_(1, forced_indices, True)
        )
        router_logits = F.linear(hidden_states, gate.weight)
        router_top_value = probs.gather(-1, forced_indices).to(router_logits.dtype)
        return router_logits, router_top_value, forced_indices

    return forward


def _pack_contract_tensor(model, recorded: dict[int, torch.Tensor], num_tokens: int):
    """``[tokens, num_hidden_layers * top_k]`` int32, columns by HF decoder layer."""
    num_layers = model.config.num_hidden_layers
    top_k = model.config.num_experts_per_tok
    packed = torch.zeros(num_tokens, num_layers * top_k, dtype=torch.int32)
    view = packed.view(num_tokens, num_layers, top_k)
    for layer_idx, ids in recorded.items():
        view[:, layer_idx, :] = ids.to(torch.int32)
    return packed


def _record_own_choices(model, input_ids, blocks):
    recorded: dict[int, torch.Tensor] = {}
    logits_ref, layer_outputs_ref = _run(
        model,
        input_ids,
        blocks,
        router_overrides={
            layer_idx: _recording_router(mlp.gate, recorded, layer_idx)
            for layer_idx, mlp in blocks
        },
    )
    num_tokens = input_ids.numel()
    top_k = model.config.num_experts_per_tok
    for layer_idx, _ in blocks:
        assert recorded[layer_idx].shape == (num_tokens, top_k)
        assert recorded[layer_idx].min() >= 0
        assert recorded[layer_idx].max() < _num_experts(model)
    return recorded, logits_ref, layer_outputs_ref


def test_replaying_own_choices_through_hf_math_is_bitwise_noop(model, input_ids):
    blocks = _sparse_blocks(model)
    recorded, logits_ref, layer_outputs_ref = _record_own_choices(model, input_ids, blocks)

    logits_forced, layer_outputs_forced = _run(
        model,
        input_ids,
        blocks,
        router_overrides={
            layer_idx: _forced_router_hf_math(mlp.gate, recorded[layer_idx])
            for layer_idx, mlp in blocks
        },
    )
    for layer_idx, _ in blocks:
        assert torch.equal(layer_outputs_forced[layer_idx], layer_outputs_ref[layer_idx])
    assert torch.equal(logits_forced, logits_ref)


def test_replaying_own_choices_through_the_real_patch_is_a_noop(model, input_ids):
    pytest.importorskip("megatron.core")
    from megatron.core.transformer.moe.router import TopKRouter

    install_topk_router_patch()
    patched_forward = TopKRouter.forward

    blocks = _sparse_blocks(model)
    recorded, logits_ref, layer_outputs_ref = _record_own_choices(model, input_ids, blocks)
    num_tokens = input_ids.numel()

    # The trainer side of the contract: one chunk, {layer_number: hf index}
    # for every sparse layer, the packed tensor installed right before the
    # forward, every router fetching exactly once.
    ctx = RoutingReplayContext(
        [{layer_idx + 1: layer_idx for layer_idx, _ in blocks}],
        model.config.num_hidden_layers,
        model.config.num_experts_per_tok,
    )
    packed = _pack_contract_tensor(model, recorded, num_tokens)
    ctx.begin_pass(forward_only=True)
    try:
        ctx.install_packed(packed, 0)
        logits_replay, layer_outputs_replay = _run(
            model,
            input_ids,
            blocks,
            router_overrides={
                layer_idx: _forced_router_real_patch(
                    mlp.gate, layer_idx, recorded[layer_idx], patched_forward
                )
                for layer_idx, mlp in blocks
            },
        )
        ctx.assert_all_consumed()
    finally:
        ctx.end_pass()

    for layer_idx, _ in blocks:
        assert torch.allclose(
            layer_outputs_replay[layer_idx],
            layer_outputs_ref[layer_idx],
            atol=1e-5,
            rtol=1e-5,
        )
    assert torch.allclose(logits_replay, logits_ref, atol=1e-4, rtol=1e-5)


def test_unforced_reruns_are_deterministic(model, input_ids):
    # Guards the bitwise assertions above: two plain CPU forwards must agree.
    blocks = _sparse_blocks(model)
    logits_a, _ = _run(model, input_ids, blocks)
    logits_b, _ = _run(model, input_ids, blocks)
    assert torch.equal(logits_a, logits_b)
