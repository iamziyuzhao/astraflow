"""R3 acceptance property on a tiny HF Qwen3-MoE model (CPU only).

Record the top-k expert choices made by ``Qwen3MoeForCausalLM``'s own routers
on a random input, then re-run the model with the routers replaced by
forced-routing reimplementations that are handed those recorded indices.
Replaying the model's own choices must be a no-op:

- with the HF router math (gather the fp32 softmax at the recorded indices
  and renormalize), the outputs are bit-identical;
- with the Megatron replay-patch math (fp32 masked-fill softmax), the outputs
  match to numerical tolerance (the two formulations are mathematically
  identical, proven exactly in ``test_routing_replay_math.py``).

No megatron import — the HF model serves as the reference router stack.
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")

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


def _forced_router_replay_math(gate, forced_indices: torch.Tensor):
    """The Megatron replay-patch math: fp32 masked-fill softmax."""

    def forward(hidden_states):
        hidden_states = hidden_states.reshape(-1, gate.hidden_dim)
        router_logits = F.linear(hidden_states, gate.weight)
        routing_map = torch.zeros_like(router_logits, dtype=torch.bool).scatter_(
            1, forced_indices, True
        )
        probs = torch.softmax(
            router_logits.float().masked_fill(~routing_map, float("-inf")), dim=-1
        )
        router_top_value = probs.gather(-1, forced_indices).to(router_logits.dtype)
        return router_logits, router_top_value, forced_indices

    return forward


def test_replaying_own_choices_is_a_noop(model, input_ids):
    blocks = _sparse_blocks(model)
    num_experts = getattr(model.config, "num_experts", None)
    if num_experts is None:
        num_experts = model.config.num_local_experts
    top_k = model.config.num_experts_per_tok

    # Pass 1: record the model's own routing choices.
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
    for layer_idx, _ in blocks:
        assert recorded[layer_idx].shape == (num_tokens, top_k)
        assert recorded[layer_idx].min() >= 0
        assert recorded[layer_idx].max() < num_experts

    # Pass 2: force the recorded choices through the HF router math.
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
        assert torch.equal(
            layer_outputs_forced[layer_idx], layer_outputs_ref[layer_idx]
        )
    assert torch.equal(logits_forced, logits_ref)

    # Pass 3: force the recorded choices through the Megatron replay math.
    logits_replay, layer_outputs_replay = _run(
        model,
        input_ids,
        blocks,
        router_overrides={
            layer_idx: _forced_router_replay_math(mlp.gate, recorded[layer_idx])
            for layer_idx, mlp in blocks
        },
    )
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
