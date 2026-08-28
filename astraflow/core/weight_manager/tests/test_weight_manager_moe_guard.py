"""CPU tests for the MoE guard on the legacy megatron_metadata path.

The legacy shard-direct sender path only understands dense TP sharding;
``WeightManager.initialize`` must refuse it for MoE models and point the
caller at the mbridge HF-export path. The guard fires before any buffer
layout / sender-agent / shm work, so these tests are pure CPU and spawn
nothing."""

from types import SimpleNamespace

import pytest

from astraflow.core.weight_manager.config import WeightManagerConfig
from astraflow.core.weight_manager.transfer.config import (
    SenderAgentConfig,
    TransferEngineConfig,
)
from astraflow.core.weight_manager.weight_manager import (
    WeightManager,
    _detect_num_moe_experts,
)


def _make_weight_manager() -> WeightManager:
    sender_config = SenderAgentConfig(
        trainer_global_rank=0,
        trainer_world_size=1,
        engine_configs=[
            TransferEngineConfig(local_hostname="localhost", handshake_port=21000)
        ],
    )
    return WeightManager(WeightManagerConfig(sender_config=sender_config))


class TestDetectNumMoeExperts:
    def test_dense_metadata_returns_zero(self):
        meta = {
            "tp_size": 4,
            "shard_specs": [],
            "conversion_config": {"model_type": "qwen3"},
        }
        assert _detect_num_moe_experts(meta) == 0

    def test_conversion_config_dict_hf_spelling(self):
        meta = {"conversion_config": {"model_type": "qwen3_moe", "num_experts": 128}}
        assert _detect_num_moe_experts(meta) == 128

    def test_conversion_config_object_megatron_spelling(self):
        meta = {"conversion_config": SimpleNamespace(num_moe_experts=64)}
        assert _detect_num_moe_experts(meta) == 64

    def test_top_level_key(self):
        meta = {"num_moe_experts": 8, "conversion_config": {}}
        assert _detect_num_moe_experts(meta) == 8

    def test_missing_conversion_config(self):
        assert _detect_num_moe_experts({"shard_specs": []}) == 0


class TestLegacyPathMoeGuard:
    def test_raises_for_moe_config_on_legacy_path(self):
        wm = _make_weight_manager()
        megatron_metadata = {
            "tp_size": 2,
            "shard_specs": [],
            "conversion_config": {"model_type": "qwen3_moe", "num_experts": 128},
        }
        # local_rank != 0 so nothing beyond the guard could start a sender
        # agent even if the guard regressed.
        with pytest.raises(ValueError, match="megatron_hf_meta"):
            wm.initialize(
                iter(()),
                local_rank=1,
                global_rank=1,
                megatron_metadata=megatron_metadata,
            )

    def test_raises_for_attr_style_moe_config(self):
        wm = _make_weight_manager()
        megatron_metadata = {
            "tp_size": 1,
            "shard_specs": [],
            "conversion_config": SimpleNamespace(num_moe_experts=128),
        }
        with pytest.raises(ValueError, match="MoE"):
            wm.initialize(
                iter(()),
                local_rank=1,
                global_rank=1,
                megatron_metadata=megatron_metadata,
            )

    def test_dense_legacy_path_passes_guard(self):
        wm = _make_weight_manager()
        megatron_metadata = {
            "tp_size": 1,
            "shard_specs": [],
            "conversion_config": {"model_type": "qwen3"},
        }
        # Dense metadata passes the guard; with local_rank=1 and torch
        # dist uninitialized the rest of initialize is a no-op.
        wm.initialize(
            iter(()),
            local_rank=1,
            global_rank=1,
            megatron_metadata=megatron_metadata,
        )

    def test_hf_export_path_unaffected_by_moe(self):
        wm = _make_weight_manager()
        # megatron_hf_meta (the mbridge export path) is the supported MoE
        # route: the legacy guard must not fire there.
        megatron_hf_meta = [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", ([8, 4], "bfloat16")),
        ]
        wm.initialize(
            iter(()),
            local_rank=1,
            global_rank=1,
            megatron_hf_meta=megatron_hf_meta,
        )
