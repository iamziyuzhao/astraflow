"""RoutingReplayContext invariants and the patched ``TopKRouter.forward`` (CPU).

The context is a per-layer record queue with two cursors (forward, recompute
re-forward). These tests pin the fail-fast contract around it:

- a record is served once per forward and once more per recompute; a third
  read, a forward-only second read, a row-count mismatch and an install that
  is not followed by its consuming forward (lockstep) all raise
  ``RoutingReplayError``;
- ``assert_all_consumed`` rejects unread records and mapped layers that never
  received one (a vacuous pass);
- the packed ``[tokens, num_hidden_layers * top_k]`` int32 tensor is viewed,
  not copied, and only the served chunk is widened to int64;
- a wrong column count fails at install (torch view error);
- ``assert_router_config_supported`` accepts exactly the softmax/top-k/no-aux
  router the masked-softmax replay is equivalent to;
- the shipped ``TopKRouter.forward`` patch (driven with a duck-typed ``self``,
  since a real router needs process groups) forces the served top-k, returns
  the fp32 masked softmax in the logits dtype, and lets torch's ``scatter_``
  reject an out-of-range expert id.
"""

from types import SimpleNamespace

import pytest
import torch

from astraflow.train_worker.utils.mcore.routing_replay import (
    _SUPPORTED_ROUTER,
    RoutingReplayContext,
    RoutingReplayError,
    assert_router_config_supported,
    get_replay_context,
    install_topk_router_patch,
)

NUM_LAYERS = 4
TOP_K = 2
NUM_EXPERTS = 8
ROWS = 13
ALL_LAYERS = {1: 0, 2: 1, 3: 2, 4: 3}


def _mb(seed: int) -> torch.Tensor:
    """A packed micro-batch ``[ROWS, NUM_LAYERS * TOP_K]`` with distinct cells."""
    return (
        torch.arange(ROWS * NUM_LAYERS * TOP_K, dtype=torch.int32).reshape(
            ROWS, NUM_LAYERS * TOP_K
        )
        + 100 * seed
    )


def _expected(mb: torch.Tensor, layer_number: int) -> torch.Tensor:
    return mb.view(ROWS, NUM_LAYERS, TOP_K)[:, layer_number - 1, :].long()


@pytest.fixture()
def ctx():
    context = RoutingReplayContext([ALL_LAYERS], NUM_LAYERS, TOP_K)
    yield context
    context.end_pass()
    assert get_replay_context() is None


# ---------------------------------------------------------------------------
# Serving order
# ---------------------------------------------------------------------------


def test_forward_then_recompute_are_served_the_same_rows(ctx):
    mbs = [_mb(i) for i in range(3)]
    ctx.begin_pass(forward_only=False)
    assert get_replay_context() is ctx
    for mb in mbs:
        ctx.install_packed(mb, 0)
        for layer_number in ALL_LAYERS:  # genuine forward
            served = ctx.fetch(layer_number, ROWS)
            assert served.dtype == torch.int64
            assert torch.equal(served, _expected(mb, layer_number))
        for layer_number in ALL_LAYERS:  # recompute re-forward during backward
            assert torch.equal(ctx.fetch(layer_number, ROWS), _expected(mb, layer_number))
    ctx.assert_all_consumed()


def test_interleaved_1f1b_order_serves_each_op_its_own_microbatch(ctx):
    mbs = [_mb(i) for i in range(3)]
    ctx.begin_pass(forward_only=False)
    for kind, i in [("f", 0), ("f", 1), ("b", 0), ("f", 2), ("b", 1), ("b", 2)]:
        if kind == "f":
            ctx.install_packed(mbs[i], 0)
        for layer_number in ALL_LAYERS:
            assert torch.equal(ctx.fetch(layer_number, ROWS), _expected(mbs[i], layer_number))
    ctx.assert_all_consumed()


def test_forward_only_pass_serves_once_and_rejects_a_second_read(ctx):
    mbs = [_mb(i) for i in range(2)]
    ctx.begin_pass(forward_only=True)
    for mb in mbs:
        ctx.install_packed(mb, 0)
        for layer_number in ALL_LAYERS:
            assert torch.equal(ctx.fetch(layer_number, ROWS), _expected(mb, layer_number))
    ctx.assert_all_consumed()
    with pytest.raises(RoutingReplayError, match="no unconsumed routing record"):
        ctx.fetch(1, ROWS)


def test_third_read_raises(ctx):
    ctx.begin_pass(forward_only=False)
    ctx.install_packed(_mb(0), 0)
    ctx.fetch(1, ROWS)
    ctx.fetch(1, ROWS)
    with pytest.raises(RoutingReplayError, match="no routing record left"):
        ctx.fetch(1, ROWS)


def test_row_count_mismatch_raises(ctx):
    ctx.begin_pass(forward_only=False)
    ctx.install_packed(_mb(0), 0)
    with pytest.raises(RoutingReplayError, match=f"recorded {ROWS} rows, router got {ROWS - 1}"):
        ctx.fetch(2, ROWS - 1)


def test_unmapped_layer_has_no_record(ctx):
    ctx.begin_pass(forward_only=True)
    ctx.install_packed(_mb(0), 0)
    with pytest.raises(RoutingReplayError, match="layer 99"):
        ctx.fetch(99, ROWS)


def test_two_installs_before_a_fetch_violate_the_lockstep(ctx):
    ctx.begin_pass(forward_only=False)
    ctx.install_packed(_mb(0), 0)
    ctx.install_packed(_mb(1), 0)
    with pytest.raises(RoutingReplayError, match="2 records installed but forward consumed only 0"):
        ctx.fetch(1, ROWS)


# ---------------------------------------------------------------------------
# assert_all_consumed
# ---------------------------------------------------------------------------


def test_unread_record_fails_assert_all_consumed(ctx):
    ctx.begin_pass(forward_only=True)
    ctx.install_packed(_mb(0), 0)
    with pytest.raises(RoutingReplayError, match="1 records, forward consumed 0"):
        ctx.assert_all_consumed()


def test_mapped_layer_without_any_record_is_a_vacuous_pass(ctx):
    ctx.begin_pass(forward_only=True)
    with pytest.raises(RoutingReplayError, match="0 records"):
        ctx.assert_all_consumed()


def test_partial_recompute_fails_assert_all_consumed(ctx):
    """Backward must read every record or none: a half-recomputed pass is a bug."""
    ctx.begin_pass(forward_only=False)
    for i in range(2):
        ctx.install_packed(_mb(i), 0)
        for layer_number in ALL_LAYERS:
            ctx.fetch(layer_number, ROWS)
    ctx.fetch(1, ROWS)  # one recompute read of two
    with pytest.raises(RoutingReplayError, match="forward consumed 2, backward 1"):
        ctx.assert_all_consumed()


def test_begin_pass_resets_records_and_cursors(ctx):
    ctx.begin_pass(forward_only=True)
    ctx.install_packed(_mb(0), 0)
    ctx.begin_pass(forward_only=True)  # a fresh forward_backward_func call
    with pytest.raises(RoutingReplayError, match="0 records"):
        ctx.assert_all_consumed()


# ---------------------------------------------------------------------------
# Layout, dtype, chunks
# ---------------------------------------------------------------------------


def test_records_are_int32_views_and_fetch_widens_only_the_served_chunk(ctx):
    mb = _mb(0)
    ctx.begin_pass(forward_only=True)
    ctx.install_packed(mb, 0)
    record = ctx._records[1][0]
    assert record.dtype == torch.int32
    assert record.data_ptr() == mb.data_ptr()  # a view of the resident tensor
    served = ctx.fetch(1, ROWS)
    assert served.dtype == torch.int64
    assert torch.equal(served, mb.view(ROWS, NUM_LAYERS, TOP_K)[:, 0, :].long())


@pytest.mark.parametrize("columns", [NUM_LAYERS * TOP_K - 1, NUM_LAYERS * TOP_K + 1, 0])
def test_wrong_column_count_fails_at_install(ctx, columns):
    """C must be exactly num_hidden_layers * top_k: the view refuses anything else."""
    ctx.begin_pass(forward_only=True)
    with pytest.raises(RuntimeError):
        ctx.install_packed(torch.zeros(ROWS, columns, dtype=torch.int32), 0)


def test_chunk_layer_maps_install_only_that_chunks_layers():
    ctx = RoutingReplayContext([{1: 0, 2: 1}, {3: 2, 4: 3}], NUM_LAYERS, TOP_K)
    mbs = [_mb(i) for i in range(2)]
    ctx.begin_pass(forward_only=False)
    for mb in mbs:
        ctx.install_packed(mb, 0)
        assert sorted(ctx._records) == [1, 2] or all(
            len(ctx._records.get(ln, ())) < len(ctx._records[1]) for ln in (3, 4)
        )
        for layer_number in (1, 2):
            assert torch.equal(ctx.fetch(layer_number, ROWS), _expected(mb, layer_number))
        ctx.install_packed(mb, 1)
        for layer_number in (3, 4):
            assert torch.equal(ctx.fetch(layer_number, ROWS), _expected(mb, layer_number))
        for layer_number in ALL_LAYERS:  # recompute
            assert torch.equal(ctx.fetch(layer_number, ROWS), _expected(mb, layer_number))
    ctx.assert_all_consumed()
    ctx.end_pass()


def test_hf_index_selects_the_decoder_layers_columns():
    """Columns are ordered by HF decoder-layer id, K per layer; dense layers are skipped."""
    ctx = RoutingReplayContext([{7: 3, 8: 1}], NUM_LAYERS, TOP_K)  # sparse layers 1 and 3 only
    mb = _mb(0)
    ctx.begin_pass(forward_only=True)
    ctx.install_packed(mb, 0)
    view = mb.view(ROWS, NUM_LAYERS, TOP_K)
    assert torch.equal(ctx.fetch(7, ROWS), view[:, 3, :].long())
    assert torch.equal(ctx.fetch(8, ROWS), view[:, 1, :].long())
    ctx.assert_all_consumed()
    ctx.end_pass()


# ---------------------------------------------------------------------------
# Router config gate
# ---------------------------------------------------------------------------


def test_router_config_validation_accepts_the_supported_router():
    assert_router_config_supported(SimpleNamespace(**_SUPPORTED_ROUTER))


@pytest.mark.parametrize(
    "overrides",
    [
        {"moe_router_score_function": "sigmoid"},
        {"moe_router_pre_softmax": True},
        {"moe_router_topk_scaling_factor": 2.0},
        {"moe_expert_capacity_factor": 1.0},
        {"moe_router_group_topk": 4},
        {"moe_router_enable_expert_bias": True},
        {"moe_router_load_balancing_type": "aux_loss"},
    ],
    ids=lambda o: next(iter(o)),
)
def test_router_config_validation_rejects_unsupported(overrides):
    config = SimpleNamespace(**{**_SUPPORTED_ROUTER, **overrides})
    key = next(iter(overrides))
    with pytest.raises(RoutingReplayError, match=key):
        assert_router_config_supported(config)


def test_router_config_validation_reads_real_transformer_config_attributes():
    pytest.importorskip("megatron.core")
    from megatron.core.transformer import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=TOP_K,
        add_bias_linear=False,
    )
    # megatron's default load balancing is aux_loss: refused (the patch skips it).
    with pytest.raises(RoutingReplayError, match="moe_router_load_balancing_type"):
        assert_router_config_supported(config)
    config.moe_router_load_balancing_type = "none"
    assert_router_config_supported(config)


# ---------------------------------------------------------------------------
# The shipped TopKRouter.forward patch
# ---------------------------------------------------------------------------


class _PatchTarget:
    """Duck-typed ``TopKRouter`` self: everything the replay branch touches.

    ``Router.gating`` moves its weight to the current CUDA device, so the
    gating is a plain linear here; the rest is the real patched code.
    """

    def __init__(self, weight: torch.Tensor, layer_number: int):
        self.weight = weight
        self.layer_number = layer_number
        self.config = SimpleNamespace(num_moe_experts=weight.shape[0])

    def _maintain_float32_expert_bias(self):
        pass

    def apply_input_jitter(self, x):
        return x

    def gating(self, x):
        return torch.nn.functional.linear(x, self.weight)


def _patched_forward():
    pytest.importorskip("megatron.core")
    from megatron.core.transformer.moe.router import TopKRouter

    install_topk_router_patch()
    assert TopKRouter._astraflow_routing_replay_patched is True
    forward = TopKRouter.forward
    install_topk_router_patch()  # idempotent
    assert TopKRouter.forward is forward
    return forward


def _forced_ids(generator) -> torch.Tensor:
    # not the gate's own argmax set: the identity must hold for any forced set
    return torch.stack(
        [torch.randperm(NUM_EXPERTS, generator=generator)[:TOP_K] for _ in range(ROWS)]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_patched_forward_forces_the_served_topk_and_masks_the_softmax(dtype):
    forward = _patched_forward()
    generator = torch.Generator().manual_seed(3)
    weight = torch.nn.Parameter(torch.randn(NUM_EXPERTS, 16, generator=generator).to(dtype))
    hidden = torch.randn(ROWS, 16, generator=generator).to(dtype)
    ids = _forced_ids(generator)
    packed = ids.to(torch.int32)  # [ROWS, 1 * TOP_K]

    ctx = RoutingReplayContext([{1: 0}], num_layers=1, top_k=TOP_K)
    ctx.begin_pass(forward_only=True)
    ctx.install_packed(packed, 0)
    probs, routing_map = forward(_PatchTarget(weight, layer_number=1), hidden)
    ctx.assert_all_consumed()
    ctx.end_pass()

    expected_map = torch.zeros(ROWS, NUM_EXPERTS, dtype=torch.bool).scatter_(1, ids, True)
    assert torch.equal(routing_map, expected_map)
    logits = torch.nn.functional.linear(hidden, weight)
    expected_probs = torch.softmax(
        logits.float().masked_fill(~expected_map, float("-inf")), dim=-1
    ).to(dtype)
    assert probs.dtype == dtype
    assert torch.equal(probs, expected_probs)
    assert torch.all(probs[~expected_map] == 0)
    assert torch.allclose(probs.float().sum(-1), torch.ones(ROWS), atol=1e-2 if dtype is torch.bfloat16 else 1e-6)
    # differentiable through the gate
    probs.float().sum().backward()
    assert weight.grad is not None


def test_patched_forward_rejects_an_out_of_range_expert_id():
    forward = _patched_forward()
    weight = torch.randn(NUM_EXPERTS, 16)
    hidden = torch.randn(ROWS, 16)
    ids = torch.zeros(ROWS, TOP_K, dtype=torch.int32)
    ids[0, 1] = NUM_EXPERTS  # one past the last logical expert

    ctx = RoutingReplayContext([{1: 0}], num_layers=1, top_k=TOP_K)
    ctx.begin_pass(forward_only=True)
    ctx.install_packed(ids, 0)
    try:
        with pytest.raises(RuntimeError):
            forward(_PatchTarget(weight, layer_number=1), hidden)
    finally:
        ctx.end_pass()


def test_patched_forward_reads_the_layer_it_belongs_to():
    forward = _patched_forward()
    generator = torch.Generator().manual_seed(5)
    weight = torch.randn(NUM_EXPERTS, 16, generator=generator)
    hidden = torch.randn(ROWS, 16, generator=generator)
    mb = torch.randint(0, NUM_EXPERTS, (ROWS, NUM_LAYERS * TOP_K), generator=generator, dtype=torch.int32)

    ctx = RoutingReplayContext([ALL_LAYERS], NUM_LAYERS, TOP_K)
    ctx.begin_pass(forward_only=True)
    ctx.install_packed(mb, 0)
    for layer_number in ALL_LAYERS:
        _, routing_map = forward(_PatchTarget(weight, layer_number), hidden)
        expected = torch.zeros(ROWS, NUM_EXPERTS, dtype=torch.bool).scatter_(
            1, _expected(mb, layer_number), True
        )
        assert torch.equal(routing_map, expected)
    ctx.assert_all_consumed()
    ctx.end_pass()
