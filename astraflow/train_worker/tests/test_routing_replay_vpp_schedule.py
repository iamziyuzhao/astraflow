"""R3 install/consume lockstep under megatron's real pipeline schedules (CPU).

The routing-replay records are only correct if

    records for megatron layer L are appended only by forwards that also
    consume L, in the same ``forward_step``.

``megatron.core.pipeline_parallel.schedules`` drives the order in which
forwards, backwards and (through activation recompute) re-forwards happen, so
these tests replay the *real* schedule tables against a
:class:`RoutingReplayContext` and check which micro-batch's rows each router
would be served.

Two properties are pinned:

1. Under virtual pipeline parallelism megatron hands the *same* micro-batch
   dict to every model chunk's forward. Installing every local layer once per
   micro-batch (on chunk 0's forward) makes the records of layers hosted on
   chunks >= 1 run ahead of their own forwards. Cursors would still balance,
   so recompute re-forwards would be served a *later* micro-batch's rows and
   later genuine forwards an *earlier* one's. The lockstep check must raise.
2. Installing per chunk (``install_packed(..., chunk_index)`` from
   ``forward_step``, keyed by ``self.model.index(model)``) keeps the lockstep,
   and every fetch -- genuine forward and recompute re-forward alike -- is
   served its own micro-batch.

The last tests drive the real ``MegatronEngine.forward_backward_batch``
closure with a fake schedule and fake model chunks: install per chunk, read
(never pop) the micro-batch dict, ``begin_pass``/``assert_all_consumed``/
``end_pass`` around the schedule, and a loud ``KeyError`` when the batch
carries no ``routed_experts`` at all (a server that does not capture).

No GPU, no distributed init: the parallel-state lookups inside
``get_pp_rank_microbatches`` are stubbed out, everything else is the real
megatron code.
"""

import pytest
import torch

pytest.importorskip("megatron.core")

from megatron.core.pipeline_parallel import schedules  # noqa: E402

from astraflow.train_worker.utils.mcore.routing_replay import (  # noqa: E402
    RoutingReplayContext,
    RoutingReplayError,
    get_replay_context,
)

TOP_K = 2
NUM_EXPERTS = 8
ROWS = 4


def _routed_experts(num_microbatches: int, num_layers: int) -> list[torch.Tensor]:
    """One packed ``[rows, num_layers * top_k]`` int32 tensor per micro-batch."""
    generator = torch.Generator().manual_seed(1234)
    return [
        torch.randint(
            0,
            NUM_EXPERTS,
            (ROWS, num_layers * TOP_K),
            generator=generator,
            dtype=torch.int32,
        )
        for _ in range(num_microbatches)
    ]


def _expected(routed_mb: torch.Tensor, hf_index: int) -> torch.Tensor:
    """The rows ``fetch`` must serve for HF decoder layer ``hf_index``."""
    return routed_mb.view(ROWS, -1, TOP_K)[:, hf_index, :].long()


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
    ``("F" | "B", virtual_microbatch_id)`` in execution order -- exactly the
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
    """Drive ``ctx`` through ``ops``, returning one entry per router fetch.

    ``per_chunk=True`` is the shipped behavior (each chunk's forward installs
    only its own layers, ``install_packed(mb, chunk_index)``).
    ``per_chunk=False`` reproduces the broken variant: one install per
    micro-batch on chunk 0's forward, covering *every* local layer (``ctx``
    must then be built with a single merged chunk map).
    """
    microbatch_id_table, model_chunk_id_table = zip(*schedule_table)
    num_model_chunks = len(chunk_layer_maps)
    forward_order: list[list[int]] = [[] for _ in range(num_model_chunks)]
    backward_count = [0] * num_model_chunks
    served: list[tuple[str, int, int]] = []

    for kind, virtual_microbatch_id in ops:
        if kind == "F":
            chunk_id = model_chunk_id_table[virtual_microbatch_id]
            microbatch_id = microbatch_id_table[virtual_microbatch_id]
            if per_chunk:
                ctx.install_packed(routed[microbatch_id], chunk_id)
            elif chunk_id == 0:
                ctx.install_packed(routed[microbatch_id], 0)
            forward_order[chunk_id].append(microbatch_id)
        else:
            chunk_id = (
                num_model_chunks - 1 - model_chunk_id_table[virtual_microbatch_id]
            )
            microbatch_id = forward_order[chunk_id][backward_count[chunk_id]]
            backward_count[chunk_id] += 1
        # A genuine forward and an activation-recompute re-forward both run
        # every router hosted on that chunk exactly once.
        for layer_number, hf_index in chunk_layer_maps[chunk_id].items():
            chunk = ctx.fetch(layer_number, ROWS)
            served.append((kind, chunk_id, layer_number))
            assert torch.equal(chunk, _expected(routed[microbatch_id], hf_index)), (
                f"{kind} op on chunk {chunk_id} layer {layer_number} was served "
                f"the wrong micro-batch's routing (wanted micro-batch {microbatch_id})"
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
    """Interleaved layer assignment, as megatron lays chunks out.

    ``{megatron layer_number (1-based): HF decoder-layer index (0-based)}``
    per chunk, which is exactly what ``_init_routing_replay`` builds.
    """
    chunk_layer_maps = []
    layer_number = 1
    for _ in range(num_model_chunks):
        chunk_map = {}
        for _ in range(layers_per_chunk):
            chunk_map[layer_number] = layer_number - 1
            layer_number += 1
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
    num_layers = sum(len(m) for m in chunk_layer_maps)
    routed = _routed_experts(num_microbatches, num_layers)
    schedule_table, ops = _schedule(
        monkeypatch,
        num_microbatches=num_microbatches,
        num_model_chunks=num_model_chunks,
        pipeline_parallel_size=pp_size,
        pipeline_parallel_rank=pp_rank,
        microbatch_group_size_per_vp_stage=group_size,
    )
    ctx = RoutingReplayContext(chunk_layer_maps, num_layers, TOP_K)
    ctx.begin_pass(forward_only=False)
    served = _run_schedule(
        ctx, schedule_table, ops, chunk_layer_maps, routed, per_chunk=True
    )
    # Every layer was forwarded once and re-forwarded (recompute) once per
    # micro-batch, and every fetch got its own micro-batch's rows.
    assert len(served) == 2 * num_microbatches * num_layers
    ctx.assert_all_consumed()
    ctx.end_pass()


@pytest.mark.parametrize(
    ("pp_size", "pp_rank", "num_model_chunks", "num_microbatches", "group_size"),
    VPP_CONFIGS,
)
def test_vpp_single_install_per_microbatch_raises(
    monkeypatch, pp_size, pp_rank, num_model_chunks, num_microbatches, group_size
):
    """The silently corrupting variant must fail loudly at the lockstep check."""
    chunk_layer_maps = _vpp_layer_maps(num_model_chunks)
    num_layers = sum(len(m) for m in chunk_layer_maps)
    routed = _routed_experts(num_microbatches, num_layers)
    schedule_table, ops = _schedule(
        monkeypatch,
        num_microbatches=num_microbatches,
        num_model_chunks=num_model_chunks,
        pipeline_parallel_size=pp_size,
        pipeline_parallel_rank=pp_rank,
        microbatch_group_size_per_vp_stage=group_size,
    )
    merged = {k: v for chunk_map in chunk_layer_maps for k, v in chunk_map.items()}
    ctx = RoutingReplayContext([merged], num_layers, TOP_K)
    ctx.begin_pass(forward_only=False)
    with pytest.raises(RoutingReplayError, match="records installed"):
        _run_schedule(
            ctx, schedule_table, ops, chunk_layer_maps, routed, per_chunk=False
        )
    ctx.end_pass()


# ---------------------------------------------------------------------------
# Plain (non-virtual) pipeline parallelism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("num_microbatches", [1, 4, 16])
def test_non_vpp_1f1b_serves_forward_and_recompute_their_own_microbatch(
    monkeypatch, pp_size, num_microbatches
):
    chunk_layer_maps = [{1: 0, 2: 1, 3: 2}]
    num_layers = 3
    routed = _routed_experts(num_microbatches, num_layers)
    for pp_rank in range(pp_size):
        schedule_table, ops = _schedule(
            monkeypatch,
            num_microbatches=num_microbatches,
            num_model_chunks=1,
            pipeline_parallel_size=pp_size,
            pipeline_parallel_rank=pp_rank,
            microbatch_group_size_per_vp_stage=1,
        )
        # With one model chunk the per-chunk and the single-install call
        # sites are the same call.
        for per_chunk in (True, False):
            ctx = RoutingReplayContext(chunk_layer_maps, num_layers, TOP_K)
            ctx.begin_pass(forward_only=False)
            served = _run_schedule(
                ctx, schedule_table, ops, chunk_layer_maps, routed, per_chunk=per_chunk
            )
            assert len(served) == 2 * num_microbatches * num_layers
            ctx.assert_all_consumed()
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
    ctx = RoutingReplayContext(chunk_layer_maps, 2, TOP_K)
    ctx.begin_pass(forward_only=True)
    _run_schedule(ctx, schedule_table, ops, chunk_layer_maps, routed, per_chunk=True)
    ctx.assert_all_consumed()
    ctx.end_pass()


# ---------------------------------------------------------------------------
# MegatronEngine.forward_backward_batch: install per chunk, never pop
# ---------------------------------------------------------------------------


class _FakeChunk:
    """A local model chunk; ``forward_step`` locates it with ``self.model.index``."""

    def __init__(self, name: str, layer_map: dict[int, int]):
        self.name = name
        self.layer_map = layer_map


def _padded_mb(routed: torch.Tensor | None) -> dict[str, torch.Tensor]:
    mb = {
        "input_ids": torch.zeros(ROWS, dtype=torch.long),
        "cu_seqlens": torch.tensor([0, ROWS], dtype=torch.int32),
        "position_ids": torch.arange(ROWS).unsqueeze(0),
    }
    if routed is not None:
        mb["routed_experts"] = routed
    return mb


def _mb_list(padded_mbs):
    from astraflow.train_worker.api.cli_args import MicroBatchSpec
    from astraflow.train_worker.utils.data import MicroBatchList

    n = len(padded_mbs)
    return MicroBatchList(
        data={},
        mb_spec=MicroBatchSpec(n_mbs=n),
        mbs=[{} for _ in range(n)],
        forward_indices=list(range(n)),
        backward_indices=list(range(n)),
        group_lens=[1] * n,
        padded_mbs=list(padded_mbs),
        padding_lengths=[0] * n,
        padded_to_lengths=[ROWS] * n,
    )


def _bare_engine(ctx, chunks):
    """A MegatronEngine with only what ``forward_backward_batch`` touches."""
    from astraflow.train_worker.engine.megatron_engine import MegatronEngine

    engine = object.__new__(MegatronEngine)
    engine.model = chunks
    engine.routing_replay_context = ctx
    engine.is_offload = False
    return engine


@pytest.fixture()
def fake_pipeline(monkeypatch):
    """Swap megatron's schedule and the model forward under ``forward_step``.

    The fake schedule mimics what matters here: every chunk gets its own
    iterator over the *same* micro-batch dicts, ``forward_step`` runs once per
    (micro-batch, chunk), and in a training pass every chunk forward is
    re-run once more (activation recompute) after the genuine forwards.
    The fake model forward plays the routers: one ``fetch`` per mapped layer.
    """
    import astraflow.train_worker.engine.megatron_engine as engine_mod

    calls: dict[str, list] = {"forwards": [], "seen": []}

    def fake_model_forward(model, padded_mb):
        for layer_number, hf_index in model.layer_map.items():
            chunk = get_replay_context().fetch(layer_number, ROWS)
            assert torch.equal(chunk, _expected(padded_mb["routed_experts"], hf_index))
            calls["forwards"].append((model.name, layer_number))
        calls["seen"].append((model, padded_mb))
        return torch.zeros(ROWS, 4)

    def fake_schedule(
        *, forward_step_func, data_iterator, model, num_microbatches, forward_only, **_
    ):
        chunks = model if isinstance(model, list) else [model]
        iterators = data_iterator if isinstance(data_iterator, list) else [data_iterator]
        first = len(calls["seen"])
        for _ in range(num_microbatches):
            for iterator, chunk in zip(iterators, chunks, strict=True):
                forward_step_func(iterator, chunk)
        if not forward_only:
            # backward runs micro-batches FIFO and the chunks of one micro-batch
            # last-to-first; each backward re-forwards its chunk (recompute)
            seen = calls["seen"][first:]
            for start in range(0, len(seen), len(chunks)):
                for chunk, padded_mb in reversed(seen[start : start + len(chunks)]):
                    fake_model_forward(chunk, padded_mb)
        return []

    monkeypatch.setattr(engine_mod, "get_forward_backward_func", lambda: fake_schedule)
    monkeypatch.setattr(engine_mod, "packed_context_parallel_forward", fake_model_forward)
    monkeypatch.setattr(engine_mod.mpu, "is_pipeline_last_stage", lambda **kwargs: False)
    return calls


@pytest.mark.parametrize("chunk_layer_maps", [[{1: 0, 2: 1}], [{1: 0}, {2: 1}]])
@pytest.mark.parametrize("forward_only", [True, False])
def test_forward_step_installs_per_chunk_and_never_pops(
    fake_pipeline, chunk_layer_maps, forward_only
):
    chunks = [_FakeChunk(f"c{i}", m) for i, m in enumerate(chunk_layer_maps)]
    routed = _routed_experts(2, 2)
    mb_list = _mb_list([_padded_mb(r) for r in routed])
    ctx = RoutingReplayContext(chunk_layer_maps, 2, TOP_K)
    engine = _bare_engine(ctx, chunks)

    # Two passes over the *same* micro-batch dicts: the first must not consume
    # 'routed_experts' out of them, and every pass ends with the context cleared.
    for _ in range(2):
        engine.forward_backward_batch(
            mb_list, process_output_fn=lambda *a: None, forward_only=forward_only
        )
        assert get_replay_context() is None
        assert all("routed_experts" in mb for mb in mb_list.padded_mbs)

    per_mb = [(c.name, ln) for c in chunks for ln in c.layer_map]
    genuine = per_mb * 2  # two micro-batches
    # backward visits a micro-batch's chunks last-to-first; a chunk's own
    # layers still re-forward in order
    recompute_mb = [(c.name, ln) for c in reversed(chunks) for ln in c.layer_map]
    recompute = recompute_mb * 2 if not forward_only else []
    assert fake_pipeline["forwards"] == (genuine + recompute) * 2


def test_forward_step_without_routed_experts_fails_loudly(fake_pipeline):
    """A rollout that captured nothing is caught before any router runs."""
    chunk_layer_maps = [{1: 0}, {2: 1}]
    chunks = [_FakeChunk(f"c{i}", m) for i, m in enumerate(chunk_layer_maps)]
    mb_list = _mb_list([_padded_mb(None)])
    ctx = RoutingReplayContext(chunk_layer_maps, 2, TOP_K)
    engine = _bare_engine(ctx, chunks)

    with pytest.raises(KeyError, match="routed_experts"):
        engine.forward_backward_batch(mb_list, process_output_fn=lambda *a: None)
    assert fake_pipeline["forwards"] == []
    # no cleanup on the error path by design: a failed step kills the trainer, and the
    # next begin_pass resets the context anyway
    ctx.end_pass()
