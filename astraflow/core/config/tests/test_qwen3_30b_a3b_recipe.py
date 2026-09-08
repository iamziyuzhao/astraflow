"""Round-trip tests for the qwen3-30b-a3b-m2po one-node B200 recipe YAMLs.

Feeds the recipe's YAMLs through the exact production load path -- the
dict-level ``astraflow.core.config.loader`` functions plus the trainer's
``to_structured_cfg`` OmegaConf merge -- and asserts every MoE/R3-critical
key survives instead of being silently dropped, that the sglang block passes
the R3 launch guards in ``SGLangConfig.build_args``, and that the trainer
copies ``moe_router_replay`` onto the reference policy whenever one is
built. All CPU, no services.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from astraflow.core.config.loader import (
    load_and_merge_configs,
    load_dataflow_config,
    load_raas_config,
    load_trainer_config,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
RECIPE_DIR = REPO_ROOT / "examples" / "math" / "qwen3-30b-a3b-m2po"
EXPERIMENT_YAML = RECIPE_DIR / "yaml" / "experiment_b200_1node.yaml"
RAAS_YAML = RECIPE_DIR / "yaml" / "raas_b200_1node.yaml"

MODEL = "Qwen/Qwen3-30B-A3B"
EXPECTED_ENGINE = {
    "backend": "megatron",
    "data_parallel_size": 4,
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "expert_parallel_size": 4,
    "expert_tensor_parallel_size": 1,
}


def _trainer_obj():
    from astraflow.train_worker.api.cli_args import GRPOConfig, to_structured_cfg

    raw = load_and_merge_configs([str(EXPERIMENT_YAML)])
    trainer_dict = load_trainer_config(raw, trainer_key="trainer_model0")
    cfg = to_structured_cfg(OmegaConf.create(trainer_dict), GRPOConfig)
    return OmegaConf.to_object(cfg)


def test_recipe_files_exist():
    for p in (EXPERIMENT_YAML, RAAS_YAML):
        assert p.exists(), f"missing recipe file: {p}"


def test_trainer_structured_roundtrip():
    obj = _trainer_obj()

    # Parallelism reaches the trainer intact (engine -> allocation_mode).
    assert obj.allocation_mode == EXPECTED_ENGINE
    # MoE/R3-critical trainer flags survive the structured merge.
    assert obj.actor.megatron.use_deterministic_algorithms is True
    assert obj.actor.megatron.moe_router_replay is True
    # Full (not delta) transfer for MoE bring-up.
    assert obj.weight_transfer_strategies == "full"
    # Batch stays divisible by dp_world_size * group_size = 4 * 8.
    assert obj.train_batch_size == 256
    assert obj.train_batch_size % (4 * 8) == 0
    assert obj.actor.path == MODEL


def test_engine_block_is_valid_megatron_parallel_strategy():
    from astraflow.train_worker.api.alloc_mode import MegatronParallelStrategy

    raw = load_and_merge_configs([str(EXPERIMENT_YAML)])
    trainer_dict = load_trainer_config(raw, trainer_key="trainer_model0")
    engine = dict(trainer_dict["allocation_mode"])
    assert engine.pop("backend") == "megatron"

    # __post_init__ validates the EP nesting; world must be 4 GPUs.
    strategy = MegatronParallelStrategy(**engine)
    assert strategy.world_size == 4
    assert strategy.expert_model_parallel_size == 4
    assert strategy.world_size % strategy.expert_model_parallel_size == 0


def _raas_cfg():
    raw = load_and_merge_configs([str(EXPERIMENT_YAML), str(RAAS_YAML)])
    return load_raas_config(raw)


def test_raas_r3_keys_survive_merge():
    raas_cfg = _raas_cfg()

    model0 = raas_cfg["models"]["model0"]
    sglang = model0["sglang"]
    assert sglang["enable_return_routed_experts"] is True
    # sglang sizes the routed-experts capturer buffer from it; -1 is refused.
    assert sglang["chunked_prefill_size"] == 32768
    # "auto" picks a non-capturing MoE backend on sm_100.
    assert sglang["moe_runner_backend"] == "triton"
    assert sglang["context_length"] == 4096
    assert sglang["model_path"] == MODEL

    gconfig = model0["gconfig"]
    assert gconfig["return_routed_experts"] is True
    assert gconfig["n_samples"] == 8

    # Rollout hardware: 2-way DP SGLang beside the 4-GPU trainer.
    assert raas_cfg["allocation_mode"]["model0"]["data_parallel_size"] == 2


def _sglang_obj():
    from astraflow.raas.api.cli_args import SGLangConfig

    sglang_dict = _raas_cfg()["models"]["model0"]["sglang"]
    merged = OmegaConf.merge(
        OmegaConf.structured(SGLangConfig), OmegaConf.create(sglang_dict)
    )
    return OmegaConf.to_object(merged)


def test_raas_sglang_block_matches_dataclass():
    """Struct-mode OmegaConf merge errors on unknown keys, so a passing
    merge proves every sglang key in the recipe exists on SGLangConfig
    (nothing is silently dropped) and values survive to_object."""
    obj = _sglang_obj()
    assert obj.enable_return_routed_experts is True
    assert obj.moe_runner_backend == "triton"
    assert obj.chunked_prefill_size == 32768
    assert obj.context_length == 4096
    assert obj.disable_radix_cache is True


def test_raas_sglang_block_passes_the_r3_launch_guards():
    """What the RaaS server actually launches with: build_args must accept it."""
    pytest.importorskip("sglang")
    from astraflow.raas.api.cli_args import SGLangConfig

    args = SGLangConfig.build_args(sglang_config=_sglang_obj(), tp_size=1, base_gpu_id=0)
    assert args["enable_return_routed_experts"] is True
    assert args["moe_runner_backend"] == "triton"
    assert args["chunked_prefill_size"] == 32768
    assert args["context_length"] == 4096


def test_gconfig_block_matches_dataclass():
    from astraflow.raas.api.cli_args import GenerationHyperparameters

    gconfig_dict = _raas_cfg()["models"]["model0"]["gconfig"]
    merged = OmegaConf.merge(
        OmegaConf.structured(GenerationHyperparameters),
        OmegaConf.create(gconfig_dict),
    )
    obj = OmegaConf.to_object(merged)
    assert obj.return_routed_experts is True
    assert obj.max_new_tokens == 3000


def test_dataflow_config_survives():
    raw = load_and_merge_configs([str(EXPERIMENT_YAML), str(RAAS_YAML)])
    dataflow_cfg = load_dataflow_config(raw)

    agent = dataflow_cfg["agent"]
    assert agent["expected_model_ids"] == ["model0"]
    assert agent["workflow_spec"]["workflow_cls"] == "rlvr"
    assert agent["tokenizer_path"] == MODEL
    # The closed-loop keys reach AgentConfig through the buffer block.
    assert agent["max_buffered_samples"] == 512
    assert agent["max_staleness"] == 12
    assert agent["filter_function"] == "filter_zero_adv"


# ---------------------------------------------------------------------------
# Reference policy: built only with a KL term, and then replaying like the actor
# ---------------------------------------------------------------------------


class _StopAfterModels(Exception):
    """Raised from ``_init_rollout``, the first thing after the model block."""


def _drive_model_creation(monkeypatch, config):
    """Run ``PPOTrainerBase.__init__`` up to and including the ref block.

    ``_create_actor`` records the configs it was handed instead of building
    engines; tokenizer loading and the perf tracer are stubbed.
    """
    from astraflow.train_worker.trainer import ppo_base as ppo_base_mod

    created = []

    class _Trainer(ppo_base_mod.PPOTrainerBase):
        def _create_actor(self, actor_config):
            created.append(actor_config)
            return object()

        def _init_rollout(self, *args, **kwargs):
            raise _StopAfterModels

        def prepare_batch_from_buffer(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

        def train(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    monkeypatch.setattr(
        ppo_base_mod, "load_hf_processor_and_tokenizer", lambda path: (None, None)
    )
    monkeypatch.setattr(ppo_base_mod.perf_tracer, "configure", lambda *a, **k: None)
    monkeypatch.setattr(ppo_base_mod.seeding, "set_random_seed", lambda *a, **k: None)
    with pytest.raises(_StopAfterModels):
        _Trainer(config, train_dataset=None)
    return created


def test_recipe_builds_no_reference_policy(monkeypatch):
    """Both KL terms are 0: only the actor is built. (The loader always
    materialises a ``ref`` block with path/dtype defaults; the trainer is
    what decides not to build it.)"""
    obj = _trainer_obj()
    assert obj.actor.kl_penalty_coef == 0.0 and obj.actor.kl_ctl == 0.0
    assert obj.ref is not None

    created = _drive_model_creation(monkeypatch, obj)
    assert created == [obj.actor]


def test_ref_inherits_moe_router_replay_from_actor_when_kl_is_on(monkeypatch):
    """The reference policy must score the same routing the actor replays.

    ``ref`` is a full PPOActorConfig built independently of ``actor``, so
    ``moe_router_replay`` defaults to False there. With a KL term on, that
    would silently compute ref_logp under the reference model's own expert
    routing while the actor is pinned to the rollout's, making the penalty
    bound routing divergence as well as parameter drift. The trainer copies
    the actor's flag onto the ref before building it (an explicit ``false``
    on the ref is overwritten, not rejected).
    """
    obj = _trainer_obj()
    obj.actor.kl_penalty_coef = 0.001
    # the loader-materialised ref: independent of actor, replay off by default
    assert obj.ref.megatron.moe_router_replay is False

    created = _drive_model_creation(monkeypatch, obj)

    assert created == [obj.actor, obj.ref]
    assert obj.ref.megatron.moe_router_replay is True
    assert obj.actor.megatron.moe_router_replay is True


def test_b200_1node_recipe_closes_the_rollout_loop():
    """The single-node B200 recipe pins the closed-loop, on-policy settings.

    Two runs of the open-loop version eroded exactly as sample staleness
    climbed to the max_staleness ceiling (see the YAML header). These are the
    numbers that bound staleness now, plus the GRPO objective the R3 paper
    and miles validated on this model -- all read through the production
    loaders so a silent drop or rename fails here, not 100 steps in.
    """
    from astraflow.train_worker.api.cli_args import GRPOConfig, to_structured_cfg

    raw = load_and_merge_configs([str(EXPERIMENT_YAML), str(RAAS_YAML)])

    agent = load_dataflow_config(raw)["agent"]
    train_batch_size = raw["trainer_base"]["train_batch_size"]
    # Buffered half of the loop: two training batches (one would leave no
    # pipelining: the in-flight tail lands after the gate closes).
    assert agent["max_buffered_samples"] == 512 == 2 * train_batch_size
    # Safety net behind the gate, not the control; loose enough never to bite.
    assert agent["max_staleness"] == 12

    # In-flight half of the loop: 96 prompts x 8 samples = 768 sequences.
    raas_cfg = load_raas_config(raw)
    assert raas_cfg["rollout"]["max_concurrent_rollouts"] == 96
    assert raas_cfg["models"]["model0"]["gconfig"]["n_samples"] == 8
    assert 96 * 8 >= train_batch_size
    # Worst-case outstanding data, in training steps: five, against the
    # open loop's forty (10,000 buffered + 2,048 in flight).
    outstanding = agent["max_buffered_samples"] + 96 * 8
    assert outstanding / train_batch_size <= 5
    assert agent["max_staleness"] > outstanding / train_batch_size

    trainer_dict = load_trainer_config(raw, trainer_key="trainer_model0")
    cfg = to_structured_cfg(OmegaConf.create(trainer_dict), GRPOConfig)
    obj = OmegaConf.to_object(cfg)
    # GRPO with decoupled clipping, M2PO off, no KL term (R3 paper / miles).
    assert obj.actor.m2_threshold is None
    assert obj.actor.eps_clip == pytest.approx(0.2)
    assert obj.actor.eps_clip_higher == pytest.approx(0.28)
    assert obj.actor.kl_penalty_coef == 0.0
    assert obj.actor.kl_ctl == 0.0
    assert obj.actor.optimizer.lr == pytest.approx(1e-6)
    assert obj.actor.optimizer.weight_decay == pytest.approx(0.1)
    assert obj.actor.optimizer.beta2 == pytest.approx(0.98)
    # One Adam step per 256-sample batch, as in both references.
    assert obj.actor.ppo_n_minibatches == 1
    # Group-normalised rewards only; no second batch-level advantage norm.
    assert obj.actor.adv_norm is None
    assert obj.actor.reward_norm.mean_level == "group"
    assert obj.actor.reward_norm.std_level == "group"
    # R3 stays on; the closed loop is in addition to replay, not instead.
    assert obj.actor.megatron.moe_router_replay is True
    assert obj.train_batch_size == train_batch_size
