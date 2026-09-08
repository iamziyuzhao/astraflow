"""Pure-torch checks of the R3 forced-routing math (no megatron required).

The Megatron ``TopKRouter`` patch computes
``probs = softmax(logits.masked_fill(~routing_map, -inf))`` in fp32. These
tests prove the masked softmax is mathematically identical to gathering the
full softmax at the same top-k set and renormalizing (HF
``norm_topk_prob=True``), and that gradients flow back to the linear router
producing the logits identically for both formulations.
"""

import torch

NUM_TOKENS = 64
HIDDEN = 32
NUM_EXPERTS = 16
TOP_K = 4


def _topk_mask(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    top_indices = torch.topk(logits, top_k, dim=-1).indices
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, top_indices, True)
    return top_indices, mask


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float().masked_fill(~mask, float("-inf")), dim=-1)


def _gather_renorm(logits: torch.Tensor, top_indices: torch.Tensor) -> torch.Tensor:
    full = torch.softmax(logits.float(), dim=-1)
    top_values = full.gather(-1, top_indices)
    top_values = top_values / top_values.sum(dim=-1, keepdim=True)
    return torch.zeros_like(full).scatter(1, top_indices, top_values)


def test_masked_softmax_equals_topk_renorm():
    torch.manual_seed(0)
    logits = torch.randn(NUM_TOKENS, NUM_EXPERTS, dtype=torch.float32) * 4.0
    top_indices, mask = _topk_mask(logits, TOP_K)

    probs_masked = _masked_softmax(logits, mask)
    probs_renorm = _gather_renorm(logits, top_indices)

    assert torch.allclose(probs_masked, probs_renorm, atol=1e-6, rtol=1e-5)
    # Non-selected experts get exactly zero probability in both formulations.
    assert torch.all(probs_masked[~mask] == 0.0)
    assert torch.allclose(probs_masked.sum(dim=-1), torch.ones(NUM_TOKENS), atol=1e-6)


def test_masked_softmax_matches_renorm_for_arbitrary_forced_sets():
    # Replayed indices are generally NOT the trainer's own argmax top-k;
    # the identity must hold for any forced expert set.
    torch.manual_seed(1)
    logits = torch.randn(NUM_TOKENS, NUM_EXPERTS, dtype=torch.float32) * 4.0
    forced = torch.stack(
        [torch.randperm(NUM_EXPERTS)[:TOP_K] for _ in range(NUM_TOKENS)]
    )
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, forced, True)

    probs_masked = _masked_softmax(logits, mask)
    probs_renorm = _gather_renorm(logits, forced)

    assert torch.allclose(probs_masked, probs_renorm, atol=1e-6, rtol=1e-5)


def test_gradients_flow_to_router_weight_and_match():
    torch.manual_seed(2)
    hidden = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.float32)
    weight_init = torch.randn(NUM_EXPERTS, HIDDEN, dtype=torch.float32) * 0.1
    downstream = torch.randn(NUM_TOKENS, NUM_EXPERTS, dtype=torch.float32)

    weight_masked = torch.nn.Parameter(weight_init.clone())
    logits = torch.nn.functional.linear(hidden, weight_masked)
    top_indices, mask = _topk_mask(logits.detach(), TOP_K)
    loss_masked = (_masked_softmax(logits, mask) * downstream).sum()
    loss_masked.backward()

    weight_renorm = torch.nn.Parameter(weight_init.clone())
    logits = torch.nn.functional.linear(hidden, weight_renorm)
    loss_renorm = (_gather_renorm(logits, top_indices) * downstream).sum()
    loss_renorm.backward()

    assert weight_masked.grad is not None
    assert weight_masked.grad.abs().sum() > 0
    assert torch.allclose(weight_masked.grad, weight_renorm.grad, atol=1e-5, rtol=1e-4)
