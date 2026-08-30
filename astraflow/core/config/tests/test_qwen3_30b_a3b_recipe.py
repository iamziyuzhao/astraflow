"""Round-trip tests for the qwen3-30b-a3b-m2po recipe YAMLs (map T1.1).

Feeds the recipe's YAMLs through the exact production load path — the
dict-level ``astraflow.core.config.loader`` functions plus the trainer's
``to_structured_cfg`` OmegaConf merge — and asserts every MoE/R3-critical
key survives instead of being silently dropped. All CPU, no services.
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
EXPERIMENT_YAML = RECIPE_DIR / "yaml" / "experiment.yaml"
RAAS_YAML = RECIPE_DIR / "yaml" / "raas.yaml"
RAAS_B200_YAML = RECIPE_DIR / "yaml" / "raas_b200.yaml"
EXPERIMENT_B200_1NODE_YAML = RECIPE_DIR / "yaml" / "experiment_b200_1node.yaml"
RAAS_B200_1NODE_YAML = RECIPE_DIR / "yaml" / "raas_b200_1node.yaml"

EXPECTED_ENGINE = {
    "backend": "megatron",
    "data_parallel_size": 3,
    "tensor_parallel_size": 2,
    "pipeline_parallel_size": 1,
    "expert_parallel_size": 2,
    "expert_tensor_parallel_size": 1,
}


def test_recipe_files_exist():
    for p in (EXPERIMENT_YAML, RAAS_YAML, RAAS_B200_YAML):
        assert p.exists(), f"missing recipe file: {p}"


def test_trainer_structured_roundtrip():
    from astraflow.train_worker.api.cli_args import GRPOConfig, to_structured_cfg

    raw = load_and_merge_configs([str(EXPERIMENT_YAML)])
    trainer_dict = load_trainer_config(raw, trainer_key="trainer_model0")

    cfg = OmegaConf.create(trainer_dict)
    cfg = to_structured_cfg(cfg, GRPOConfig)
    obj = OmegaConf.to_object(cfg)

    # Parallelism reaches the trainer intact (engine -> allocation_mode).
    assert obj.allocation_mode == EXPECTED_ENGINE
    # MoE/R3-critical trainer flags survive the structured merge.
    assert obj.actor.megatron.use_deterministic_algorithms is True
    assert obj.actor.megatron.moe_router_replay is True
    # Full (not delta) transfer for MoE bring-up.
    assert obj.weight_transfer_strategies == "full"
    # Batch stays divisible by dp_world_size * group_size = 3 * 8.
    assert obj.train_batch_size == 240
    assert obj.train_batch_size % (3 * 8) == 0
    assert obj.actor.path == "Qwen/Qwen3-30B-A3B"


def test_engine_block_is_valid_megatron_parallel_strategy():
    from astraflow.train_worker.api.alloc_mode import MegatronParallelStrategy

    raw = load_and_merge_configs([str(EXPERIMENT_YAML)])
    trainer_dict = load_trainer_config(raw, trainer_key="trainer_model0")
    engine = dict(trainer_dict["allocation_mode"])
    assert engine.pop("backend") == "megatron"

    # __post_init__ validates the EP nesting; world must be 6 GPUs.
    strategy = MegatronParallelStrategy(**engine)
    assert strategy.world_size == 6
    assert strategy.expert_model_parallel_size == 2
    assert strategy.world_size % strategy.expert_model_parallel_size == 0


@pytest.mark.parametrize("raas_yaml", [RAAS_YAML, RAAS_B200_YAML])
def test_raas_r3_keys_survive_merge(raas_yaml):
    raw = load_and_merge_configs([str(EXPERIMENT_YAML), str(raas_yaml)])
    raas_cfg = load_raas_config(raw)

    model0 = raas_cfg["models"]["model0"]
    sglang = model0["sglang"]
    assert sglang["enable_return_routed_experts"] is True
    # Required pairing: sizes the routed-experts capturer device buffer.
    assert sglang["chunked_prefill_size"] == 32768
    assert sglang["context_length"] == 4096
    assert sglang["model_path"] == "Qwen/Qwen3-30B-A3B"

    gconfig = model0["gconfig"]
    assert gconfig["return_routed_experts"] is True
    assert gconfig["n_samples"] == 8

    # Rollout hardware: 4-way DP SGLang.
    assert raas_cfg["allocation_mode"]["model0"]["data_parallel_size"] == 4


@pytest.mark.parametrize("raas_yaml", [RAAS_YAML, RAAS_B200_YAML])
def test_raas_sglang_block_matches_dataclass(raas_yaml):
    """Struct-mode OmegaConf merge errors on unknown keys, so a passing
    merge proves every sglang key in the recipe exists on SGLangConfig
    (nothing is silently dropped) and values survive to_object."""
    from astraflow.raas.api.cli_args import SGLangConfig

    raw = load_and_merge_configs([str(EXPERIMENT_YAML), str(raas_yaml)])
    raas_cfg = load_raas_config(raw)
    sglang_dict = raas_cfg["models"]["model0"]["sglang"]

    merged = OmegaConf.merge(
        OmegaConf.structured(SGLangConfig), OmegaConf.create(sglang_dict)
    )
    obj = OmegaConf.to_object(merged)
    assert obj.enable_return_routed_experts is True
    assert obj.chunked_prefill_size == 32768
    assert obj.context_length == 4096
    # AstraFlow default the R3 capture path relies on (fresh prompt rows).
    assert obj.disable_radix_cache is True


def test_gconfig_block_matches_dataclass():
    from astraflow.raas.api.cli_args import GenerationHyperparameters

    raw = load_and_merge_configs([str(EXPERIMENT_YAML), str(RAAS_YAML)])
    raas_cfg = load_raas_config(raw)
    gconfig_dict = raas_cfg["models"]["model0"]["gconfig"]

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
    assert agent["tokenizer_path"] == "Qwen/Qwen3-30B-A3B"


def _bare_trainer(alloc):
    """A PPOTrainerBase with __init__ bypassed, for guards that only read config.

    PPOTrainerBase is abstract, so the stubs exist purely to make it
    instantiable; none of them is called here.
    """
    from astraflow.train_worker.trainer.ppo_base import PPOTrainerBase

    class _Concrete(PPOTrainerBase):
        def _init_rollout(self, *a, **k):  # pragma: no cover - never called
            raise NotImplementedError

        def prepare_batch_from_buffer(self, *a, **k):  # pragma: no cover
            raise NotImplementedError

        def train(self, *a, **k):  # pragma: no cover - never called
            raise NotImplementedError

    trainer = object.__new__(_Concrete)
    trainer.allocation_mode = alloc
    return trainer


def test_ref_inherits_moe_router_replay_from_actor():
    """The reference policy must replay the same routing as the actor.

    ``ref`` is a full PPOActorConfig built independently of ``actor``, so
    ``moe_router_replay`` defaults to False there. With the KL penalty on,
    that silently computes ref_logp under the reference model's own expert
    routing while the actor is pinned to the rollout's, making the penalty
    bound routing divergence as well as parameter drift. The recipe turns the
    penalty on (kl_penalty_coef=0.001), so this is the configuration that
    ships.
    """
    from astraflow.train_worker.api.cli_args import GRPOConfig, to_structured_cfg

    raw = load_and_merge_configs([str(EXPERIMENT_YAML)])
    trainer_dict = load_trainer_config(raw, trainer_key="trainer_model0")

    cfg = OmegaConf.create(trainer_dict)
    cfg = to_structured_cfg(cfg, GRPOConfig)
    obj = OmegaConf.to_object(cfg)

    # The recipe builds a reference policy at all only because of this.
    assert obj.actor.kl_penalty_coef > 0 or obj.actor.kl_ctl > 0
    assert obj.actor.megatron.moe_router_replay is True
    assert obj.ref is not None
    assert obj.ref.megatron.moe_router_replay is True


def test_ref_yaml_value_overrides_the_inherited_one():
    """Inheritance fills a gap; it does not overwrite an explicit choice."""
    raw = load_and_merge_configs([str(EXPERIMENT_YAML)])
    trainer = raw["trainer_model0"]
    trainer.setdefault("ref", {}).setdefault("megatron", {})[
        "moe_router_replay"
    ] = False

    trainer_dict = load_trainer_config(raw, trainer_key="trainer_model0")

    assert trainer_dict["ref"]["megatron"]["moe_router_replay"] is False


def test_actor_replay_with_free_running_ref_is_refused():
    """An explicit mismatch fails loudly instead of skewing the KL term."""
    from astraflow.train_worker.api.cli_args import GRPOConfig, to_structured_cfg
    from astraflow.train_worker.api.alloc_mode import AllocationMode

    raw = load_and_merge_configs([str(EXPERIMENT_YAML)])
    trainer_dict = load_trainer_config(raw, trainer_key="trainer_model0")
    cfg = OmegaConf.create(trainer_dict)
    cfg = to_structured_cfg(cfg, GRPOConfig)
    obj = OmegaConf.to_object(cfg)
    obj.ref.megatron.moe_router_replay = False

    trainer = _bare_trainer(AllocationMode.resolve(obj.allocation_mode))

    with pytest.raises(ValueError, match="moe_router_replay"):
        trainer._assert_ref_replay_matches_actor(obj)


def test_ref_replay_guard_ignores_non_megatron_backends():
    """FSDP has no megatron block to disagree about."""
    class _Alloc:
        train_backend = "fsdp"

    trainer = _bare_trainer(_Alloc())
    # Would raise on attribute access if the backend check did not short-circuit.
    trainer._assert_ref_replay_matches_actor(object())


def test_b200_1node_recipe_closes_the_rollout_loop():
    """The single-node B200 recipe pins the closed-loop, on-policy settings.

    Two runs of the open-loop version eroded exactly as sample staleness
    climbed to the max_staleness ceiling (see the YAML header). These are the
    numbers that bound staleness now, plus the GRPO objective the R3 paper
    and miles validated on this model -- all read through the production
    loaders so a silent drop or rename fails here, not 100 steps in.
    """
    from astraflow.train_worker.api.cli_args import GRPOConfig, to_structured_cfg

    raw = load_and_merge_configs(
        [str(EXPERIMENT_B200_1NODE_YAML), str(RAAS_B200_1NODE_YAML)]
    )

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
