"""CPU tests for the mbridge transformers-5.x rope_theta compat patch.

Uses duck-typed fake ``hf_config`` objects (old-style flat ``rope_theta``
vs new-style ``rope_parameters`` dict) so no model download, GPU, or
transformer_engine is needed. mbridge itself is required (it imports
megatron-core, which works CPU-only)."""

from types import SimpleNamespace

import pytest

from astraflow.train_worker.models.mcore.mbridge_compat import (
    _resolve_rope_theta,
    apply_mbridge_compat_patches,
)

llm_bridge = pytest.importorskip(
    "mbridge.core.llm_bridge", reason="mbridge (with megatron-core) not installed"
)


def _fake_bridge_self(hf_config) -> SimpleNamespace:
    """Duck-typed stand-in for an LLMBridge: _get_gptmodel_args only reads
    ``self.hf_config``."""
    return SimpleNamespace(hf_config=hf_config)


def _old_style_config(theta: float = 1000000.0) -> SimpleNamespace:
    return SimpleNamespace(
        vocab_size=1024,
        max_position_embeddings=256,
        rope_theta=theta,
    )


class _NewStyleConfig:
    """transformers>=5 shape: no ``rope_theta`` attribute at all, only a
    ``rope_parameters`` dict (SimpleNamespace would not raise
    AttributeError semantics any differently, but a plain class keeps the
    absence explicit)."""

    def __init__(self, theta: float = 1000000.0):
        self.vocab_size = 1024
        self.max_position_embeddings = 256
        self.rope_parameters = {"rope_theta": theta, "rope_type": "default"}


class TestResolveRopeTheta:
    def test_flat_attribute_wins(self):
        cfg = _old_style_config(theta=5000.0)
        assert _resolve_rope_theta(cfg) == 5000.0

    def test_rope_parameters_dict_fallback(self):
        cfg = _NewStyleConfig(theta=7777.0)
        assert _resolve_rope_theta(cfg) == 7777.0

    def test_rope_parameters_object_fallback(self):
        cfg = SimpleNamespace(rope_parameters=SimpleNamespace(rope_theta=42.0))
        assert _resolve_rope_theta(cfg) == 42.0

    def test_neither_form_returns_none(self):
        assert _resolve_rope_theta(SimpleNamespace()) is None
        assert _resolve_rope_theta(SimpleNamespace(rope_parameters={})) is None


class TestApplyPatch:
    def test_new_style_config_backfilled(self):
        apply_mbridge_compat_patches()
        cfg = _NewStyleConfig(theta=1234567.0)
        args = llm_bridge.LLMBridge._get_gptmodel_args(_fake_bridge_self(cfg))
        assert args["rotary_base"] == 1234567.0
        # The attribute is backfilled onto the config so any other
        # mbridge-side rope_theta read on this object also succeeds.
        assert cfg.rope_theta == 1234567.0
        # And the rest of the args are untouched.
        assert args["vocab_size"] == 1024
        assert args["max_sequence_length"] == 256

    def test_old_style_config_passthrough(self):
        apply_mbridge_compat_patches()
        cfg = _old_style_config(theta=9999.0)
        args = llm_bridge.LLMBridge._get_gptmodel_args(_fake_bridge_self(cfg))
        assert args["rotary_base"] == 9999.0

    def test_idempotent(self):
        apply_mbridge_compat_patches()
        patched_once = llm_bridge.LLMBridge._get_gptmodel_args
        assert getattr(patched_once, "_astraflow_rope_theta_compat", False)
        apply_mbridge_compat_patches()
        # Second application must be a no-op: same function object, not a
        # wrapper-of-a-wrapper.
        assert llm_bridge.LLMBridge._get_gptmodel_args is patched_once

    def test_real_qwen3_moe_config(self):
        transformers = pytest.importorskip("transformers")
        from transformers.models.qwen3_moe import Qwen3MoeConfig

        major = int(transformers.__version__.split(".")[0])
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        if major >= 5:
            # The bug this patch fixes: transformers 5.x dropped the flat
            # attribute.
            assert not hasattr(cfg, "rope_theta")
        apply_mbridge_compat_patches()
        args = llm_bridge.LLMBridge._get_gptmodel_args(_fake_bridge_self(cfg))
        expected = _resolve_rope_theta(cfg)
        assert expected is not None
        assert args["rotary_base"] == expected
