"""R3 routing-replay record lifetime and dtype discipline (CPU, no GPU).

The replay records are the biggest per-pass allocation R3 adds, and both of the
properties pinned here are memory properties with a correctness edge:

1. Records are stored exactly as the rollout captured them — int16, and as the
   strided view :meth:`install_packed` slices out of the packed
   ``[tokens, num_moe_layers, top_k]`` micro-batch tensor. int64 is needed only
   as a ``scatter_`` index for the single layer being served, so ``fetch()``
   materializes it as a per-call transient. Widening on ``record()`` instead
   made every layer of every micro-batch carry a resident 4x contiguous copy
   (48 layers x top-8 x 8 B = 3072 B/token, against 768 B/token at the source).
2. A record is released as soon as no cursor can legally read it again. The
   dangerous half of that is releasing too early: a chunk consumed by the
   forward cursor is NOT dead while an activation-recompute replay may still
   re-read it, so the release condition must wait for both cursors unless the
   pass is provably forward-only.

The regression guard that matters most is
``test_replay_math_is_bitwise_unchanged_by_the_dtype_refactor``: the routing
map and the masked-softmax probabilities the patched router produces must be
bit-for-bit what the pre-refactor (widen-on-record) code produced.
"""

import types
import weakref

import pytest
import torch

from astraflow.train_worker.utils.mcore.routing_replay import (
    ReplayStage,
    RoutingReplayContext,
    RoutingReplayError,
    release_replay_context,
    set_replay_context,
)

TOP_K = 2
NUM_EXPERTS = 8
ROWS = 6


def _routed(num_moe_layers: int, seed: int, rows: int = ROWS) -> torch.Tensor:
    """One micro-batch of rollout-captured ids: ``[rows, num_moe_layers, top_k]``.

    int16 is what the whole data path carries (``data_acquisition`` rejects any
    other dtype for ``routed_experts``), and each row selects ``top_k``
    *distinct* experts, as a real top-k capture does.
    """
    generator = torch.Generator().manual_seed(seed)
    ids = torch.stack(
        [
            torch.randperm(NUM_EXPERTS, generator=generator)[:TOP_K]
            for _ in range(rows * num_moe_layers)
        ]
    )
    return ids.view(rows, num_moe_layers, TOP_K).to(torch.int16)


def _slots(ctx: RoutingReplayContext, layer_number: int):
    """The raw record slots for a layer (None once released)."""
    return ctx._records[layer_number]


# ---------------------------------------------------------------------------
# 1. Records keep the rollout dtype; the int64 widening happens at fetch time
# ---------------------------------------------------------------------------


def test_install_packed_records_stay_int16_views_and_fetch_widens():
    routed = _routed(num_moe_layers=3, seed=0)
    ctx = RoutingReplayContext(layer_map={1: 0, 2: 1, 3: 2}, num_moe_layers=3)
    ctx.begin_pass(forward_only=True)
    ctx.install_packed(routed)

    for layer_number, moe_index in ((1, 0), (2, 1), (3, 2)):
        stored = _slots(ctx, layer_number)[0]
        assert stored.dtype == torch.int16, (
            f"layer {layer_number} record was widened on the way in: {stored.dtype}"
        )
        # A strided view of the caller's packed tensor, not a per-layer copy.
        assert not stored.is_contiguous()
        assert (
            stored.untyped_storage().data_ptr() == routed.untyped_storage().data_ptr()
        )

    for layer_number, moe_index in ((1, 0), (2, 1), (3, 2)):
        served = ctx.fetch(layer_number, ROWS)
        assert served.dtype == torch.int64
        assert torch.equal(served, routed[:, moe_index, :].long())
    ctx.assert_all_consumed(require_records=True)
    ctx.end_pass()


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
def test_record_stores_the_tensor_as_it_arrives(dtype):
    ids = torch.randint(0, NUM_EXPERTS, (ROWS, TOP_K), dtype=dtype)
    ctx = RoutingReplayContext()
    ctx.set_stage(ReplayStage.RECORD)
    ctx.record(1, ids)
    assert _slots(ctx, 1)[0].dtype == dtype

    ctx.set_stage(ReplayStage.REPLAY_FORWARD)
    served = ctx.fetch(1, ROWS)
    # torch requires an int64 scatter_ index, so fetch always hands back int64.
    assert served.dtype == torch.int64
    assert torch.equal(served, ids.long())


def test_int64_is_only_needed_as_a_scatter_index():
    """Pin the reason the widening cannot simply be dropped."""
    logits = torch.randn(ROWS, NUM_EXPERTS)
    ids = _routed(1, seed=42)[:, 0, :]
    assert ids.dtype == torch.int16
    with pytest.raises(RuntimeError, match="[Ii]ndex"):
        torch.zeros_like(logits, dtype=torch.bool).scatter_(1, ids, True)
    routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        1, ids.long(), True
    )
    assert int(routing_map.sum()) == ROWS * TOP_K


# ---------------------------------------------------------------------------
# 2. The replayed router math is bit-for-bit unchanged
# ---------------------------------------------------------------------------


class _LegacyContext:
    """The pre-refactor record/fetch behavior: widen on record, never release."""

    owner = "legacy"

    def __init__(self):
        self.stage = ReplayStage.REPLAY_FORWARD
        self.records: dict[int, list[torch.Tensor]] = {}
        self.cursors: dict[int, int] = {}

    def record(self, layer_number: int, topk_ids: torch.Tensor) -> None:
        self.records.setdefault(layer_number, []).append(topk_ids.detach().long())
        self.cursors.setdefault(layer_number, 0)

    def install_packed(self, routed_experts: torch.Tensor, layer_map) -> None:
        for layer_number, moe_index in layer_map.items():
            self.record(layer_number, routed_experts[:, moe_index, :])

    def fetch(self, layer_number: int, num_rows: int) -> torch.Tensor:
        cursor = self.cursors[layer_number]
        self.cursors[layer_number] = cursor + 1
        return self.records[layer_number][cursor]


class _StubRouter:
    """Duck-typed ``TopKRouter`` self for the patched forward (CPU, no CUDA).

    The real ``Router.gating`` moves its weight to ``cuda.current_device()``,
    so the gating is stubbed; everything the replay branch of the patch does
    *after* the gating is the real, shipped code.
    """

    def __init__(self, weight: torch.Tensor, layer_number: int = 1):
        self.weight = weight
        self.topk = TOP_K
        self.layer_number = layer_number
        self.config = types.SimpleNamespace(num_moe_experts=NUM_EXPERTS)

    def _maintain_float32_expert_bias(self):
        pass

    def apply_input_jitter(self, x):
        return x

    def gating(self, x):
        return x @ self.weight.t()


def test_replay_math_is_bitwise_unchanged_by_the_dtype_refactor():
    """Same probs and routing_map as the widen-on-record implementation.

    Drives the *real* patched ``TopKRouter.forward`` (the shipped masked-softmax
    replay branch) twice over the same inputs: once against the current context
    (int16 strided records, widened per fetch) and once against a stand-in that
    reproduces the pre-refactor behavior. Every output must be identical bit for
    bit, not merely close.
    """
    pytest.importorskip("megatron.core")
    from megatron.core.transformer.moe.router import TopKRouter

    from astraflow.train_worker.utils.mcore.routing_replay import (
        install_topk_router_patch,
    )

    install_topk_router_patch()
    patched_forward = TopKRouter.forward

    num_microbatches = 3
    layer_map = {1: 0, 2: 1}
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(NUM_EXPERTS, 4, generator=generator)
    hidden = [
        torch.randn(ROWS, 4, generator=generator) for _ in range(num_microbatches)
    ]
    routed = [_routed(2, seed=100 + i) for i in range(num_microbatches)]

    def _run(context, install):
        routers = {layer: _StubRouter(weight, layer) for layer in layer_map}
        outputs = []
        set_replay_context(context)
        try:
            for mb_index in range(num_microbatches):
                install(routed[mb_index])
                for layer_number in layer_map:
                    probs, routing_map = patched_forward(
                        routers[layer_number], hidden[mb_index]
                    )
                    outputs.append((probs, routing_map))
        finally:
            release_replay_context(context)
        return outputs

    ctx = RoutingReplayContext(layer_map=layer_map, num_moe_layers=2)
    ctx.begin_pass(forward_only=True)
    new = _run(ctx, ctx.install_packed)
    ctx.assert_all_consumed(require_records=True)
    ctx.end_pass()

    legacy_ctx = _LegacyContext()
    legacy = _run(
        legacy_ctx, lambda routed_mb: legacy_ctx.install_packed(routed_mb, layer_map)
    )

    assert len(new) == len(legacy) == num_microbatches * len(layer_map)
    for index, ((probs, routing_map), (ref_probs, ref_map)) in enumerate(
        zip(new, legacy, strict=True)
    ):
        assert torch.equal(routing_map, ref_map), (
            f"routing_map differs at fetch {index}"
        )
        assert torch.equal(probs, ref_probs), f"probs differ at fetch {index}"
        # Sanity: the replay really is forcing a top-k renormalized softmax.
        assert int(routing_map.sum()) == ROWS * TOP_K
        assert torch.allclose(probs.sum(dim=-1), torch.ones(ROWS), atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Lifetime: released once dead, never before
# ---------------------------------------------------------------------------


def test_forward_only_pass_releases_each_chunk_as_it_is_consumed():
    """A forward-only pass frees a chunk the moment its forward has read it.

    ``forward_only=True`` means megatron issues no backward and therefore no
    activation-recompute re-forward, so a chunk behind the forward cursor is
    provably dead. Once the caller lets go of the micro-batch tensor, the record
    views are the only thing keeping its storage alive (a non-autograd view
    holds the storage, not the base tensor object), so the records dying is the
    micro-batch's routing memory being freed.
    """
    num_microbatches = 4
    layer_map = {1: 0, 2: 1, 3: 2}
    ctx = RoutingReplayContext(layer_map=layer_map, num_moe_layers=3)
    ctx.begin_pass(forward_only=True)

    for mb_index in range(num_microbatches):
        routed = _routed(3, seed=200 + mb_index)
        ctx.install_packed(routed)
        stored = [weakref.ref(_slots(ctx, layer)[mb_index]) for layer in layer_map]
        del routed  # only the context's records hold this storage now
        assert all(ref() is not None for ref in stored)

        for layer_number in layer_map:
            ctx.fetch(layer_number, ROWS)

        assert all(ref() is None for ref in stored), (
            f"micro-batch {mb_index} routing is still resident after its "
            "forward-only pass consumed it"
        )
        for layer_number in layer_map:
            assert all(slot is None for slot in _slots(ctx, layer_number))

    # Releasing slots must not disturb the bookkeeping that reasons about them.
    for layer_number in layer_map:
        assert len(_slots(ctx, layer_number)) == num_microbatches
    ctx.assert_all_consumed(require_records=True)
    ctx.end_pass()


def test_chunk_is_not_released_while_recompute_replay_is_pending():
    """With backward replay armed, the forward cursor alone must not free."""
    ctx = RoutingReplayContext(layer_map={1: 0})
    ctx.begin_pass(forward_only=False)  # auto_backward: recompute will re-read

    mb0 = _routed(1, seed=300)
    mb1 = _routed(1, seed=301)
    # An independent int64 copy (int16 -> int64 always reallocates), so holding
    # it for the comparison below does not pin mb0's storage.
    mb0_rows = mb0[:, 0, :].long()

    ctx.install_packed(mb0)
    mb0_ref = weakref.ref(_slots(ctx, 1)[0])
    ctx.fetch(1, ROWS)  # fwd(mb0)
    del mb0
    assert _slots(ctx, 1)[0] is not None, (
        "chunk 0 was released after its forward, but an activation-recompute "
        "re-forward still has to be served the same rows"
    )
    assert mb0_ref() is not None

    ctx.install_packed(mb1)
    ctx.fetch(1, ROWS)  # fwd(mb1)
    assert _slots(ctx, 1)[0] is not None

    # Recompute re-forward of mb0 (auto_backward fall-through): it must still
    # be served mb0's rows, and only now does chunk 0 become dead.
    assert torch.equal(ctx.fetch(1, ROWS), mb0_rows)
    assert _slots(ctx, 1)[0] is None
    assert mb0_ref() is None
    assert _slots(ctx, 1)[1] is not None  # mb1 recompute still pending

    ctx.fetch(1, ROWS)  # recompute(mb1)
    assert _slots(ctx, 1)[1] is None
    ctx.assert_all_consumed(require_backward=True, require_records=True)
    ctx.end_pass()


def test_recompute_replay_still_gets_its_own_microbatch_rows():
    """The 1F1B interleave, with the exact rows checked at every fetch."""
    ctx = RoutingReplayContext(layer_map={1: 0})
    ctx.begin_pass(forward_only=False)
    mbs = [_routed(1, seed=400 + i) for i in range(3)]

    ctx.install_packed(mbs[0])
    assert torch.equal(ctx.fetch(1, ROWS), mbs[0][:, 0, :].long())  # fwd(mb0)
    ctx.install_packed(mbs[1])
    assert torch.equal(ctx.fetch(1, ROWS), mbs[1][:, 0, :].long())  # fwd(mb1)
    assert torch.equal(ctx.fetch(1, ROWS), mbs[0][:, 0, :].long())  # recompute(mb0)
    ctx.install_packed(mbs[2])
    assert torch.equal(ctx.fetch(1, ROWS), mbs[2][:, 0, :].long())  # fwd(mb2)
    assert torch.equal(ctx.fetch(1, ROWS), mbs[1][:, 0, :].long())  # recompute(mb1)
    assert torch.equal(ctx.fetch(1, ROWS), mbs[2][:, 0, :].long())  # recompute(mb2)
    ctx.assert_all_consumed(require_backward=True)
    ctx.end_pass()


def test_manual_staging_never_releases_on_the_forward_pass():
    """set_stage() callers may flip to REPLAY_BACKWARD, so nothing is dead yet.

    A manually staged context cannot know that no backward replay is coming
    (``begin_pass`` is what proves it), so the conservative branch keeps every
    chunk until the backward cursor has passed it too.
    """
    ctx = RoutingReplayContext()
    ctx.set_stage(ReplayStage.RECORD)
    chunks = [_routed(1, seed=500 + i)[:, 0, :] for i in range(2)]
    for chunk in chunks:
        ctx.record(1, chunk)

    ctx.set_stage(ReplayStage.REPLAY_FORWARD)  # auto_backward=False, no lockstep
    for chunk in chunks:
        assert torch.equal(ctx.fetch(1, ROWS), chunk.long())
    assert all(slot is not None for slot in _slots(ctx, 1))

    ctx.set_stage(ReplayStage.REPLAY_BACKWARD)
    for index, chunk in enumerate(chunks):
        assert torch.equal(ctx.fetch(1, ROWS), chunk.long())
        assert _slots(ctx, 1)[index] is None  # dead only now
    ctx.assert_all_consumed(require_backward=True)


def test_release_preserves_lockstep_and_double_consume_detection():
    """The checks that reason about len(records) still see the full history."""
    ctx = RoutingReplayContext(layer_map={1: 0})
    ctx.begin_pass(forward_only=True)
    for mb_index in range(3):
        ctx.install_packed(_routed(1, seed=600 + mb_index))
        ctx.fetch(1, ROWS)
    assert _slots(ctx, 1) == [None, None, None]

    # Double-consume is still caught (and reports the true chunk count).
    with pytest.raises(RoutingReplayError, match="all 3 chunks"):
        ctx.fetch(1, ROWS)

    # ...and so is a queue that runs ahead of its own forward.
    ctx.install_packed(_routed(1, seed=700))
    ctx.install_packed(_routed(1, seed=701))
    with pytest.raises(RoutingReplayError, match="lockstep violated"):
        ctx.fetch(1, ROWS)
    ctx.end_pass()


def test_reset_clears_the_release_watermark():
    ctx = RoutingReplayContext(layer_map={1: 0})
    ctx.begin_pass(forward_only=True)
    ctx.install_packed(_routed(1, seed=800))
    ctx.fetch(1, ROWS)
    assert _slots(ctx, 1) == [None]

    ctx.begin_pass(forward_only=True)  # calls reset()
    assert 1 not in ctx._records
    routed = _routed(1, seed=801)
    ctx.install_packed(routed)
    assert _slots(ctx, 1)[0] is not None
    assert torch.equal(ctx.fetch(1, ROWS), routed[:, 0, :].long())
    ctx.assert_all_consumed(require_records=True)
    ctx.end_pass()


def test_forward_only_pass_over_many_microbatches_pins_one_at_a_time():
    """The whole point: resident records stay O(1) in the micro-batch count."""
    num_microbatches = 32
    layer_map = {layer: layer - 1 for layer in range(1, 5)}
    ctx = RoutingReplayContext(layer_map=layer_map, num_moe_layers=4)
    ctx.begin_pass(forward_only=True)

    peak_live = 0
    for mb_index in range(num_microbatches):
        ctx.install_packed(_routed(4, seed=900 + mb_index))
        for layer_number in layer_map:
            ctx.fetch(layer_number, ROWS)
            live = sum(
                1
                for layer in layer_map
                for slot in _slots(ctx, layer)
                if slot is not None
            )
            peak_live = max(peak_live, live)
    # At most one micro-batch's worth of layer records is ever pinned.
    assert peak_live <= len(layer_map)
    assert all(slot is None for layer in layer_map for slot in _slots(ctx, layer))
    ctx.assert_all_consumed(require_records=True)
    ctx.end_pass()
