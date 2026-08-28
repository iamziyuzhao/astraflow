"""Unit tests for the R3 routing-replay stage machine (CPU, no megatron)."""

import dataclasses
import types

import pytest
import torch

from astraflow.train_worker.utils.mcore.routing_replay import (
    ReplayStage,
    RoutingReplayContext,
    RoutingReplayError,
    assert_router_config_supported,
    get_replay_context,
    hf_moe_layer_indices,
    release_replay_context,
    routing_map_to_topk_indices,
    set_replay_context,
)

TOP_K = 2


def _ids(rows: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 8, (rows, TOP_K), generator=generator, dtype=torch.int16)


def test_record_then_replay_forward_then_replay_backward():
    ctx = RoutingReplayContext()
    ctx.set_stage(ReplayStage.RECORD)
    chunk_a = _ids(6, seed=0)
    chunk_b = _ids(6, seed=1)
    ctx.record(1, chunk_a)
    ctx.record(1, chunk_b)
    ctx.record(2, chunk_a)

    ctx.set_stage(ReplayStage.REPLAY_FORWARD)
    assert torch.equal(ctx.fetch(1, 6), chunk_a.long())
    assert torch.equal(ctx.fetch(1, 6), chunk_b.long())
    assert torch.equal(ctx.fetch(2, 6), chunk_a.long())
    # Forward queue exhausted; without auto_backward a re-fetch is an error.
    with pytest.raises(RoutingReplayError, match="double-consume"):
        ctx.fetch(1, 6)

    # Simulated activation recompute: the same per-layer sequence is served
    # again, from the separate backward cursors.
    ctx.set_stage(ReplayStage.REPLAY_BACKWARD)
    assert torch.equal(ctx.fetch(1, 6), chunk_a.long())
    assert torch.equal(ctx.fetch(1, 6), chunk_b.long())
    assert torch.equal(ctx.fetch(2, 6), chunk_a.long())
    with pytest.raises(RoutingReplayError, match="double-consume"):
        ctx.fetch(2, 6)

    ctx.assert_all_consumed(require_backward=True)


def test_assert_all_consumed_catches_underconsumption():
    ctx = RoutingReplayContext()
    ctx.set_stage(ReplayStage.RECORD)
    ctx.record(1, _ids(4, seed=0))
    ctx.record(1, _ids(4, seed=1))
    ctx.record(2, _ids(4, seed=2))

    ctx.set_stage(ReplayStage.REPLAY_FORWARD)
    ctx.fetch(1, 4)
    with pytest.raises(RoutingReplayError, match="layer 1: forward consumed 1/2"):
        ctx.assert_all_consumed()

    ctx.fetch(1, 4)
    ctx.fetch(2, 4)
    ctx.assert_all_consumed()

    # A partially-replayed backward pass is an error...
    ctx.set_stage(ReplayStage.REPLAY_BACKWARD)
    ctx.fetch(1, 4)
    with pytest.raises(RoutingReplayError, match="backward partially consumed"):
        ctx.assert_all_consumed()
    # ...but consuming the rest makes every layer all-or-nothing again.
    ctx.fetch(1, 4)
    ctx.fetch(2, 4)
    ctx.assert_all_consumed(require_backward=True)


def test_no_recompute_backward_is_allowed_unless_required():
    ctx = RoutingReplayContext()
    ctx.set_stage(ReplayStage.RECORD)
    ctx.record(1, _ids(4, seed=0))
    ctx.set_stage(ReplayStage.REPLAY_FORWARD)
    ctx.fetch(1, 4)
    # No activation recompute ran: backward cursors at 0 is fine by default.
    ctx.assert_all_consumed()
    with pytest.raises(RoutingReplayError, match="backward consumed 0/1"):
        ctx.assert_all_consumed(require_backward=True)


def test_auto_backward_serves_single_chunk_1f1b_schedule():
    # NOTE: this models the SINGLE model chunk (no virtual pipeline
    # parallelism) 1F1B case: install_packed calls are 1:1 with the genuine
    # forwards. Interleaved/VPP schedules, where megatron forwards the same
    # micro-batch once per model chunk, are covered by
    # test_routing_replay_vpp_schedule.py.
    #
    # Micro-batch records are installed lazily right before each forward, so
    # during a recompute the forward queue is always exhausted — auto_backward
    # then serves the backward cursor without an explicit stage flip.
    ctx = RoutingReplayContext(layer_map={1: 0})
    ctx.begin_pass(forward_only=False)
    assert ctx.stage is ReplayStage.REPLAY_FORWARD

    mb0 = _ids(4, seed=0).unsqueeze(1)  # [tokens, 1 moe layer, top_k]
    mb1 = _ids(4, seed=1).unsqueeze(1)
    mb2 = _ids(4, seed=2).unsqueeze(1)

    ctx.install_packed(mb0)
    assert torch.equal(ctx.fetch(1, 4), mb0[:, 0].long())  # fwd(mb0)
    ctx.install_packed(mb1)
    assert torch.equal(ctx.fetch(1, 4), mb1[:, 0].long())  # fwd(mb1)
    assert torch.equal(ctx.fetch(1, 4), mb0[:, 0].long())  # recompute(mb0)
    ctx.install_packed(mb2)
    assert torch.equal(ctx.fetch(1, 4), mb2[:, 0].long())  # fwd(mb2)
    assert torch.equal(ctx.fetch(1, 4), mb1[:, 0].long())  # recompute(mb1)
    assert torch.equal(ctx.fetch(1, 4), mb2[:, 0].long())  # recompute(mb2)

    ctx.assert_all_consumed(require_backward=True)
    with pytest.raises(RoutingReplayError, match="double-consume"):
        ctx.fetch(1, 4)

    ctx.end_pass()
    assert ctx.stage is ReplayStage.OFF


def test_fetch_validates_rows_stage_and_missing_records():
    ctx = RoutingReplayContext()
    ctx.set_stage(ReplayStage.RECORD)
    ctx.record(1, _ids(4, seed=0))

    ctx.set_stage(ReplayStage.REPLAY_FORWARD)
    with pytest.raises(RoutingReplayError, match="do not match router"):
        ctx.fetch(1, 5)
    with pytest.raises(RoutingReplayError, match="No recorded routing"):
        ctx.fetch(99, 4)

    ctx.set_stage(ReplayStage.OFF)
    with pytest.raises(RoutingReplayError, match="stage is off"):
        ctx.fetch(1, 4)

    with pytest.raises(RoutingReplayError, match=r"Expected \[tokens, top_k\]"):
        ctx.record(1, torch.zeros(4, 1, TOP_K, dtype=torch.int16))


def test_install_packed_uses_layer_map_slices():
    routed = torch.stack([_ids(4, seed=0), _ids(4, seed=1)], dim=1)  # [4, 2, K]
    ctx = RoutingReplayContext(layer_map={3: 0, 4: 1})
    ctx.begin_pass(forward_only=True)
    ctx.install_packed(routed)
    assert torch.equal(ctx.fetch(3, 4), routed[:, 0].long())
    assert torch.equal(ctx.fetch(4, 4), routed[:, 1].long())
    ctx.assert_all_consumed()

    with pytest.raises(RoutingReplayError, match="only has 2 MoE layers"):
        RoutingReplayContext(layer_map={1: 2}).install_packed(routed)
    with pytest.raises(RoutingReplayError, match="requires a layer_map"):
        RoutingReplayContext().install_packed(routed)


def test_routing_map_to_topk_indices_roundtrip():
    generator = torch.Generator().manual_seed(0)
    indices = torch.stack(
        [torch.randperm(8, generator=generator)[:TOP_K] for _ in range(16)]
    )
    routing_map = torch.zeros(16, 8, dtype=torch.bool).scatter_(1, indices, True)
    recovered = routing_map_to_topk_indices(routing_map, TOP_K)
    assert torch.equal(recovered.sort(dim=-1).values, indices.sort(dim=-1).values)
    with pytest.raises(RoutingReplayError, match="exactly top_k"):
        routing_map_to_topk_indices(routing_map, TOP_K + 1)


def test_hf_moe_layer_indices():
    def config(**kwargs):
        return types.SimpleNamespace(**kwargs)

    # Qwen3-MoE style: every decoder layer is MoE.
    assert hf_moe_layer_indices(
        config(
            num_hidden_layers=4,
            mlp_only_layers=[],
            decoder_sparse_step=1,
            num_experts=8,
        )
    ) == [0, 1, 2, 3]
    # Sparse step and mlp_only_layers exclusions.
    assert hf_moe_layer_indices(
        config(
            num_hidden_layers=6,
            mlp_only_layers=[3],
            decoder_sparse_step=2,
            num_experts=8,
        )
    ) == [1, 5]
    # No experts -> no MoE layers.
    assert (
        hf_moe_layer_indices(
            config(
                num_hidden_layers=4,
                mlp_only_layers=[],
                decoder_sparse_step=1,
                num_experts=0,
            )
        )
        == []
    )
    # num_local_experts fallback (transformers 5.x config serialization).
    assert hf_moe_layer_indices(
        config(
            num_hidden_layers=2,
            mlp_only_layers=[],
            decoder_sparse_step=1,
            num_local_experts=8,
        )
    ) == [0, 1]


def test_lockstep_violation_is_loud():
    """A queue that runs ahead of its own forward must raise, not serve rows.

    This is the virtual-pipeline-parallelism failure mode in miniature: one
    forward appends records for a layer it does not itself consume, so the
    layer's queue runs ahead and a later fetch would silently return a
    different micro-batch's routing.
    """
    ctx = RoutingReplayContext(layer_map={1: 0})
    ctx.begin_pass(forward_only=False)
    mb0 = _ids(4, seed=0).unsqueeze(1)
    mb1 = _ids(4, seed=1).unsqueeze(1)
    ctx.install_packed(mb0)
    ctx.install_packed(mb1)  # second install without an intervening fetch
    with pytest.raises(RoutingReplayError, match="lockstep violated"):
        ctx.fetch(1, 4)


def test_manual_staging_keeps_multi_chunk_record_semantics():
    # set_stage() (debug/unit-test entry) must not apply the lockstep check:
    # records may legitimately be appended in bulk via record().
    ctx = RoutingReplayContext()
    ctx.set_stage(ReplayStage.RECORD)
    ctx.record(1, _ids(4, seed=0))
    ctx.record(1, _ids(4, seed=1))
    ctx.set_stage(ReplayStage.REPLAY_FORWARD)
    ctx.fetch(1, 4)
    ctx.fetch(1, 4)
    ctx.assert_all_consumed()


def test_install_packed_requires_a_chunk_index_when_multi_chunk():
    ctx = RoutingReplayContext(chunk_layer_maps=[{1: 0}, {2: 1}])
    ctx.begin_pass(forward_only=True)
    routed = torch.stack([_ids(4, seed=0), _ids(4, seed=1)], dim=1)
    with pytest.raises(RoutingReplayError, match="must name the megatron model chunk"):
        ctx.install_packed(routed)
    with pytest.raises(RoutingReplayError, match="out of range"):
        ctx.install_packed(routed, chunk_index=2)


def test_chunk_layer_maps_must_be_disjoint():
    with pytest.raises(RoutingReplayError, match="more than one model chunk"):
        RoutingReplayContext(chunk_layer_maps=[{1: 0}, {1: 0}])
    with pytest.raises(RoutingReplayError, match="not both"):
        RoutingReplayContext(layer_map={1: 0}, chunk_layer_maps=[{1: 0}])


def test_install_packed_requires_exact_moe_layer_count():
    routed = torch.stack([_ids(4, seed=0), _ids(4, seed=1)], dim=1)  # 2 MoE layers
    ctx = RoutingReplayContext(layer_map={1: 0}, num_moe_layers=3)
    ctx.begin_pass(forward_only=True)
    with pytest.raises(RoutingReplayError, match="refusing to use a prefix"):
        ctx.install_packed(routed)
    # The exact count is accepted.
    ok = RoutingReplayContext(layer_map={1: 0}, num_moe_layers=2)
    ok.begin_pass(forward_only=True)
    ok.install_packed(routed)
    assert torch.equal(ok.fetch(1, 4), routed[:, 0].long())


def test_assert_all_consumed_require_records_catches_a_vacuous_pass():
    # A context that was never consulted by any router (e.g. because another
    # engine's context was the process-wide active one) has an empty record
    # dict, which used to pass assert_all_consumed vacuously.
    ctx = RoutingReplayContext(layer_map={1: 0, 2: 1})
    ctx.begin_pass(forward_only=True)
    ctx.assert_all_consumed()  # vacuously fine without require_records
    with pytest.raises(RoutingReplayError, match=r"layers \[1, 2\] recorded nothing"):
        ctx.assert_all_consumed(require_records=True)


def test_set_replay_context_refuses_to_displace_another_owner():
    actor = RoutingReplayContext(layer_map={1: 0}, owner="actor-engine")
    ref = RoutingReplayContext(layer_map={1: 0}, owner="ref-engine")
    try:
        set_replay_context(actor)
        assert get_replay_context() is actor
        with pytest.raises(RoutingReplayError, match="actor-engine"):
            set_replay_context(ref)
        # The incumbent is untouched: the actor's routers still see its records.
        assert get_replay_context() is actor
        # Re-installing the same context is a no-op, not a conflict.
        set_replay_context(actor)
        assert get_replay_context() is actor
    finally:
        release_replay_context(actor)
    assert get_replay_context() is None

    # Scoped install/release lets several engines share one process.
    try:
        set_replay_context(actor)
        release_replay_context(actor)
        set_replay_context(ref)
        assert get_replay_context() is ref
    finally:
        release_replay_context(ref)
    assert get_replay_context() is None


def test_release_replay_context_only_clears_its_own():
    actor = RoutingReplayContext(layer_map={1: 0}, owner="actor-engine")
    other = RoutingReplayContext(layer_map={1: 0}, owner="other-engine")
    try:
        set_replay_context(actor)
        release_replay_context(other)
        assert get_replay_context() is actor
    finally:
        release_replay_context(actor)


def _router_config(**overrides):
    """A TransformerConfig-shaped stand-in with Qwen3-MoE's replay-safe values."""
    defaults = dict(
        moe_router_score_function="softmax",
        moe_router_pre_softmax=False,
        moe_router_topk_scaling_factor=None,
        moe_expert_capacity_factor=None,
        moe_router_group_topk=None,
        moe_router_enable_expert_bias=False,
        moe_z_loss_coeff=None,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.001,
        moe_router_force_load_balancing=False,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def test_router_config_validation_accepts_qwen3_moe():
    # mbridge.models.qwen3moe sets score_function='softmax', pre_softmax=False,
    # load_balancing_type='none' and leaves scaling/capacity factors unset.
    assert_router_config_supported(_router_config()) is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        # mbridge qwen2moe / mixtral
        ({"moe_router_pre_softmax": True}, "moe_router_pre_softmax=True"),
        # mbridge deepseek_v3
        ({"moe_router_score_function": "sigmoid"}, "moe_router_score_function"),
        ({"moe_router_topk_scaling_factor": 2.5}, "moe_router_topk_scaling_factor"),
        ({"moe_expert_capacity_factor": 1.0}, "moe_expert_capacity_factor"),
        ({"moe_router_group_topk": 4}, "moe_router_group_topk"),
        ({"moe_router_enable_expert_bias": True}, "moe_router_enable_expert_bias"),
        ({"moe_z_loss_coeff": 1e-3}, "moe_z_loss_coeff"),
        ({"moe_router_force_load_balancing": True}, "moe_router_force_load_balancing"),
        (
            {"moe_router_load_balancing_type": "aux_loss"},
            "moe_router_load_balancing_type",
        ),
    ],
)
def test_router_config_validation_rejects_unsupported(overrides, message):
    with pytest.raises(RoutingReplayError, match=message):
        assert_router_config_supported(_router_config(**overrides), layer_number=7)


def test_router_config_validation_reads_real_transformer_config_attributes():
    """Guard against megatron renaming the attributes we probe with getattr().

    Every name checked by assert_router_config_supported must exist on the real
    TransformerConfig, otherwise the getattr() defaults would make the
    validation silently pass for every model.
    """
    pytest.importorskip("megatron.core")
    from megatron.core.transformer import TransformerConfig

    fields = {field.name for field in dataclasses.fields(TransformerConfig)}
    probed = {
        "moe_router_score_function",
        "moe_router_pre_softmax",
        "moe_router_topk_scaling_factor",
        "moe_expert_capacity_factor",
        "moe_router_group_topk",
        "moe_router_enable_expert_bias",
        "moe_z_loss_coeff",
        "moe_router_load_balancing_type",
        "moe_aux_loss_coeff",
        "moe_router_force_load_balancing",
    }
    assert probed <= fields, (
        f"missing from TransformerConfig: {sorted(probed - fields)}"
    )

    # A default TransformerConfig (which is what mbridge's qwen3moe bridge
    # starts from) is replay-safe apart from the fields the bridge overrides.
    config = TransformerConfig(
        num_layers=1, hidden_size=8, num_attention_heads=1, moe_router_topk=2
    )
    config.moe_router_load_balancing_type = "none"
    assert_router_config_supported(config)
