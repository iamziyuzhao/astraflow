"""R3 install/consume lockstep under megatron's real pipeline schedules (CPU).

The routing-replay FIFO queues are only correct if

    records for megatron layer L are appended only by forwards that also
    consume L, in the same ``forward_step``.

``megatron.core.pipeline_parallel.schedules`` drives the order in which
forwards, backwards and (through activation recompute) re-forwards happen, so
these tests replay the *real* schedule tables against a
:class:`RoutingReplayContext` and check which micro-batch's rows each router
would be served.

Two regressions are pinned:

1. Under virtual pipeline parallelism megatron hands the *same* micro-batch
   dict to every model chunk's forward. Installing every local layer once per
   micro-batch (on chunk 0's forward) makes the queues of layers hosted on
   chunks >= 1 run ahead of their own forwards. Cursors still balance, so
   ``assert_all_consumed`` passed and nothing raised — while recompute
   re-forwards were served a *later* micro-batch's rows and later genuine
   forwards an *earlier* one's. That must now raise.
2. Installing per chunk restores the lockstep, and every fetch — genuine
   forward and recompute re-forward alike — is served its own micro-batch.

No GPU, no distributed init: the parallel-state lookups inside
``get_pp_rank_microbatches`` are stubbed out, everything else is the real
megatron code.
"""

import types

import pytest
import torch

pytest.importorskip("megatron.core")

from megatron.core.pipeline_parallel import schedules  # noqa: E402

from astraflow.train_worker.utils.mcore.routing_replay import (  # noqa: E402
    RoutingReplayContext,
    RoutingReplayError,
)

TOP_K = 2
NUM_EXPERTS = 8
ROWS = 4


def _routed_experts(num_microbatches: int, num_moe_layers: int) -> list[torch.Tensor]:
    """One ``[rows, num_moe_layers, top_k]`` tensor per micro-batch."""
    generator = torch.Generator().manual_seed(1234)
    return [
        torch.randint(
            0,
            NUM_EXPERTS,
            (ROWS, num_moe_layers, TOP_K),
            generator=generator,
            dtype=torch.int32,
        )
        for _ in range(num_microbatches)
    ]


def _schedule(
    monkeypatch,
    *,
    num_microbatches: int,
    num_model_chunks: int,
    pipeline_parallel_size: int,
    pipeline_parallel_rank: int,
    microbatch_group_size_per_vp_stage: int,
    forward_only: bool = False,
):
    """Real megatron schedule table plus the warmup/1F1B/cooldown op order.

    Returns ``(schedule_table, ops)`` where ``ops`` is a list of
    ``("F" | "B", virtual_microbatch_id)`` in execution order — exactly the
    order ``forward_backward_pipelining_with_interleaving`` (and, for a single
    chunk, ``..._without_interleaving``) issues them.
    """
    virtual_pipeline_size = num_model_chunks if num_model_chunks > 1 else None
    monkeypatch.setattr(
        schedules.parallel_state,
        "get_pipeline_model_parallel_world_size",
        lambda: pipeline_parallel_size,
    )
    monkeypatch.setattr(
        schedules.parallel_state,
        "get_pipeline_model_parallel_rank",
        lambda: pipeline_parallel_rank,
    )
    monkeypatch.setattr(
        schedules.parallel_state,
        "get_virtual_pipeline_model_parallel_world_size",
        lambda: virtual_pipeline_size,
    )
    schedule_table = schedules.get_schedule_table(
        num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage
    )
    (
        total_num_microbatches,
        _,
        num_warmup_microbatches,
        num_microbatches_remaining,
    ) = schedules.get_pp_rank_microbatches(
        num_microbatches,
        num_model_chunks,
        microbatch_group_size_per_vp_stage,
        forward_only,
    )
    ops = [("F", v) for v in range(num_warmup_microbatches)]
    for i in range(num_microbatches_remaining):
        ops.append(("F", num_warmup_microbatches + i))
        if not forward_only:
            ops.append(("B", i))
    if not forward_only:
        ops.extend(
            ("B", i) for i in range(num_microbatches_remaining, total_num_microbatches)
        )
    return schedule_table, ops


def _run_schedule(ctx, schedule_table, ops, chunk_layer_maps, routed, *, per_chunk):
    """Drive ``ctx`` through ``ops``, returning the micro-batch served per fetch.

    ``per_chunk=True`` is the fixed behavior (each chunk's forward installs only
    its own layers). ``per_chunk=False`` reproduces the old behavior: the
    micro-batch dict is popped on chunk 0's forward, installing *every* local
    layer exactly once per micro-batch.
    """
    microbatch_id_table, model_chunk_id_table = zip(*schedule_table)
    num_model_chunks = len(chunk_layer_maps)
    forward_order: list[list[int]] = [[] for _ in range(num_model_chunks)]
    backward_count = [0] * num_model_chunks
    served: list[tuple[str, int, int, torch.Tensor]] = []

    def _consume(kind, chunk_id, microbatch_id):
        for layer_number, moe_index in chunk_layer_maps[chunk_id].items():
            chunk = ctx.fetch(layer_number, ROWS)
            served.append((kind, chunk_id, layer_number, chunk))
            yield layer_number, moe_index, microbatch_id, chunk

    for kind, virtual_microbatch_id in ops:
        if kind == "F":
            chunk_id = model_chunk_id_table[virtual_microbatch_id]
            microbatch_id = microbatch_id_table[virtual_microbatch_id]
            if per_chunk:
                ctx.install_packed(routed[microbatch_id], chunk_index=chunk_id)
            elif chunk_id == 0:
                # Old behavior: one install per micro-batch, covering *every*
                # local layer, on chunk 0's forward only.
                ctx.install_packed(routed[microbatch_id])
            forward_order[chunk_id].append(microbatch_id)
        else:
            chunk_id = (
                num_model_chunks - 1 - model_chunk_id_table[virtual_microbatch_id]
            )
            microbatch_id = forward_order[chunk_id][backward_count[chunk_id]]
            backward_count[chunk_id] += 1
        # A genuine forward and an activation-recompute re-forward both run
        # every router hosted on that chunk exactly once.
        for layer_number, moe_index, mb_id, chunk in _consume(
            kind, chunk_id, microbatch_id
        ):
            expected = routed[mb_id][:, moe_index, :].long()
            assert torch.equal(chunk, expected), (
                f"{kind} op on chunk {chunk_id} layer {layer_number} was served "
                f"the wrong micro-batch's routing (wanted micro-batch {mb_id})"
            )
    return served


# ---------------------------------------------------------------------------
# Virtual pipeline parallelism
# ---------------------------------------------------------------------------

VPP_CONFIGS = [
    # (pp_size, pp_rank, num_model_chunks, num_microbatches, group_size)
    (2, 0, 2, 4, 2),
    (2, 1, 2, 4, 2),
    (2, 0, 2, 8, 4),
    (4, 0, 2, 8, 4),
    (4, 2, 2, 8, 4),
    (4, 0, 4, 8, 4),
    (4, 3, 4, 8, 4),
    (2, 0, 3, 6, 2),
]


def _vpp_layer_maps(num_model_chunks, layers_per_chunk=2):
    """Interleaved layer assignment, as megatron lays chunks out (1-based)."""
    chunk_layer_maps = []
    moe_index = 0
    layer_number = 1
    for _ in range(num_model_chunks):
        chunk_map = {}
        for _ in range(layers_per_chunk):
            chunk_map[layer_number] = moe_index
            layer_number += 1
            moe_index += 1
        chunk_layer_maps.append(chunk_map)
    return chunk_layer_maps


@pytest.mark.parametrize(
    ("pp_size", "pp_rank", "num_model_chunks", "num_microbatches", "group_size"),
    VPP_CONFIGS,
)
def test_vpp_per_chunk_install_serves_the_right_microbatch(
    monkeypatch, pp_size, pp_rank, num_model_chunks, num_microbatches, group_size
):
    chunk_layer_maps = _vpp_layer_maps(num_model_chunks)
    num_moe_layers = sum(len(m) for m in chunk_layer_maps)
    routed = _routed_experts(num_microbatches, num_moe_layers)
    schedule_table, ops = _schedule(
        monkeypatch,
        num_microbatches=num_microbatches,
        num_model_chunks=num_model_chunks,
        pipeline_parallel_size=pp_size,
        pipeline_parallel_rank=pp_rank,
        microbatch_group_size_per_vp_stage=group_size,
    )
    ctx = RoutingReplayContext(
        chunk_layer_maps=chunk_layer_maps, num_moe_layers=num_moe_layers
    )
    ctx.begin_pass(forward_only=False)
    served = _run_schedule(
        ctx, schedule_table, ops, chunk_layer_maps, routed, per_chunk=True
    )
    # Every layer was forwarded once and re-forwarded (recompute) once per
    # micro-batch, and every fetch got its own micro-batch's rows.
    assert len(served) == 2 * num_microbatches * num_moe_layers
    ctx.assert_all_consumed(require_backward=True, require_records=True)
    ctx.end_pass()


@pytest.mark.parametrize(
    ("pp_size", "pp_rank", "num_model_chunks", "num_microbatches", "group_size"),
    VPP_CONFIGS,
)
def test_vpp_single_install_per_microbatch_now_raises(
    monkeypatch, pp_size, pp_rank, num_model_chunks, num_microbatches, group_size
):
    """The old (silently corrupting) behavior must now fail loudly."""
    chunk_layer_maps = _vpp_layer_maps(num_model_chunks)
    num_moe_layers = sum(len(m) for m in chunk_layer_maps)
    routed = _routed_experts(num_microbatches, num_moe_layers)
    schedule_table, ops = _schedule(
        monkeypatch,
        num_microbatches=num_microbatches,
        num_model_chunks=num_model_chunks,
        pipeline_parallel_size=pp_size,
        pipeline_parallel_rank=pp_rank,
        microbatch_group_size_per_vp_stage=group_size,
    )
    merged = {k: v for chunk_map in chunk_layer_maps for k, v in chunk_map.items()}
    ctx = RoutingReplayContext(layer_map=merged, num_moe_layers=num_moe_layers)
    ctx.begin_pass(forward_only=False)
    with pytest.raises(RoutingReplayError, match="lockstep violated"):
        _run_schedule(
            ctx, schedule_table, ops, chunk_layer_maps, routed, per_chunk=False
        )
    ctx.end_pass()


# ---------------------------------------------------------------------------
# Plain (non-virtual) pipeline parallelism — behavior must be unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("num_microbatches", [1, 4, 16])
def test_non_vpp_1f1b_is_unchanged(monkeypatch, pp_size, num_microbatches):
    chunk_layer_maps = [{1: 0, 2: 1, 3: 2}]
    num_moe_layers = 3
    routed = _routed_experts(num_microbatches, num_moe_layers)
    for pp_rank in range(pp_size):
        schedule_table, ops = _schedule(
            monkeypatch,
            num_microbatches=num_microbatches,
            num_model_chunks=1,
            pipeline_parallel_size=pp_size,
            pipeline_parallel_rank=pp_rank,
            microbatch_group_size_per_vp_stage=1,
        )
        # Both the chunk-aware and the legacy single-install call sites are
        # identical when there is only one model chunk.
        for per_chunk in (True, False):
            ctx = RoutingReplayContext(
                chunk_layer_maps=chunk_layer_maps, num_moe_layers=num_moe_layers
            )
            ctx.begin_pass(forward_only=False)
            served = _run_schedule(
                ctx, schedule_table, ops, chunk_layer_maps, routed, per_chunk=per_chunk
            )
            assert len(served) == 2 * num_microbatches * num_moe_layers
            ctx.assert_all_consumed(require_backward=True, require_records=True)
            ctx.end_pass()


@pytest.mark.parametrize("pp_size", [1, 2, 4])
def test_forward_only_pass_has_no_recompute(monkeypatch, pp_size):
    chunk_layer_maps = [{1: 0, 2: 1}]
    num_microbatches = 4
    routed = _routed_experts(num_microbatches, 2)
    schedule_table, ops = _schedule(
        monkeypatch,
        num_microbatches=num_microbatches,
        num_model_chunks=1,
        pipeline_parallel_size=pp_size,
        pipeline_parallel_rank=0,
        microbatch_group_size_per_vp_stage=1,
        forward_only=True,
    )
    assert all(kind == "F" for kind, _ in ops)
    ctx = RoutingReplayContext(chunk_layer_maps=chunk_layer_maps, num_moe_layers=2)
    ctx.begin_pass(forward_only=True)
    _run_schedule(ctx, schedule_table, ops, chunk_layer_maps, routed, per_chunk=True)
    ctx.assert_all_consumed(require_records=True)
    ctx.end_pass()


# ---------------------------------------------------------------------------
# packed_context_parallel_forward: install per chunk, never mutate the batch
# ---------------------------------------------------------------------------


class _FakeModelChunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(sequence_parallel=False)

    def forward(self, input_ids, **kwargs):
        return torch.zeros(*input_ids.shape, 4)


@pytest.fixture()
def _single_rank_mpu(monkeypatch):
    import astraflow.train_worker.utils.mcore.packed_context_parallel as pcp

    monkeypatch.setattr(pcp.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pcp.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(pcp.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pcp.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(pcp.mpu, "is_pipeline_last_stage", lambda **kwargs: False)
    return pcp


def test_packed_forward_installs_per_chunk_and_does_not_pop(
    _single_rank_mpu, monkeypatch
):
    pcp = _single_rank_mpu
    from astraflow.train_worker.utils.mcore.routing_replay import (
        release_replay_context,
        set_replay_chunk_index,
        set_replay_context,
    )

    routed = _routed_experts(1, 2)[0]
    micro_batch = {
        "input_ids": torch.zeros(ROWS, dtype=torch.long),
        "cu_seqlens": torch.tensor([0, ROWS], dtype=torch.int32),
        "position_ids": torch.arange(ROWS).unsqueeze(0),
        "routed_experts": routed,
    }
    chunk_layer_maps = [{1: 0}, {2: 1}]
    ctx = RoutingReplayContext(chunk_layer_maps=chunk_layer_maps, num_moe_layers=2)
    chunks = [_FakeModelChunk(), _FakeModelChunk()]
    for index, chunk in enumerate(chunks):
        set_replay_chunk_index(chunk, index)

    set_replay_context(ctx)
    try:
        # Two passes over the *same* micro-batch dict: the first must not
        # consume 'routed_experts' out of it.
        for _ in range(2):
            ctx.begin_pass(forward_only=True)
            for index, chunk in enumerate(chunks):
                pcp.packed_context_parallel_forward(chunk, micro_batch)
                assert "routed_experts" in micro_batch
                # Only that chunk's own layer got a record installed.
                layer_number = next(iter(chunk_layer_maps[index]))
                assert torch.equal(
                    ctx.fetch(layer_number, ROWS),
                    routed[:, chunk_layer_maps[index][layer_number], :].long(),
                )
            ctx.assert_all_consumed(require_records=True)
            ctx.end_pass()
    finally:
        release_replay_context(ctx)
