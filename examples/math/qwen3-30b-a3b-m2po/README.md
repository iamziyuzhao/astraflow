# Qwen3-30B-A3B Math RL — Megatron MoE backend + Rollout Routing Replay (R3)

Math RL recipe for **Qwen3-30B-A3B** (MoE: 48 layers, 128 experts, top-8),
cloned from [`qwen3-8b-megatron-delta`](../qwen3-8b-megatron-delta) (M2PO,
DeepScaleR data) with three deliberate differences:

1. **MoE parallelism** — Megatron backend with TP=2, EP=2, ETP=1, PP=1,
   DP=3 on 6 GPUs (H200 class).
1. **Full (not delta) TCP weight transfer** — MoE per-step delta density
   is unmeasured; a delta that overflows its pre-allocated buffer falls
   back to full permanently anyway. Measure before switching back.
1. **R3 (Rollout Routing Replay)** — the SGLang server records which
   experts each token was routed to during rollout, and the trainer
   replays exactly that routing in its forwards, closing the
   rollout/training MoE mismatch that destabilizes RL on MoE models.

This is a **bring-up config**: context 4096, one eval set. Scale
`sglang.context_length`, `gconfig.max_new_tokens`, dataset `max_length`,
and `mb_spec.max_tokens_per_mb` together once the pipeline is validated.

## The three R3 flags

R3 must be enabled on all three sides at once (all set in this recipe):

| Flag                                                     | File                                     | What it does                                                                                                                          |
| -------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `sglang.enable_return_routed_experts: true`              | `yaml/raas.yaml` / `yaml/raas_b200.yaml` | SGLang (>=0.5.13) allocates its routed-experts capturer; requests may ask for per-token expert indices.                               |
| `raas.models.model0.gconfig.return_routed_experts: true` | `yaml/experiment.yaml`                   | Every rollout request asks for the recorded expert indices; they ride the rollout record into the training batch as `routed_experts`. |
| `trainer_base.actor.megatron.moe_router_replay: true`    | `yaml/experiment.yaml`                   | The trainer's MoE router replays the recorded top-8 expert indices in every training forward instead of re-deciding routing.          |

Two supporting settings that are **not optional** with R3:

- `sglang.chunked_prefill_size: 32768` — sizes the capturer's device
  buffer; the AstraFlow default (`-1`, unchunked) undersizes it and
  crashes any large prefill batch mid-forward.
- `actor.megatron.use_deterministic_algorithms: true` — the replayed
  forward must be reproducible; the R3 equivalence gate depends on it.

Keep speculative decoding, hierarchical cache, and PD disaggregation off
(all default off and not exposed in `SGLangConfig`); leave
`disable_radix_cache` at its AstraFlow default (`true`).

## GPU layout

**Default (two nodes):** trainer node has 6 GPUs, rollout node has 4.

| Component                 | Node    | GPUs       | Parallelism                   |
| ------------------------- | ------- | ---------- | ----------------------------- |
| AstraFlow HTTP service    | trainer | none (CPU) | —                             |
| Trainer model0 (Megatron) | trainer | 0-5        | TP=2, PP=1, DP=3, EP=2, ETP=1 |
| RaaS (SGLang, model0)     | rollout | 0-3        | DP=4, TP=1                    |

World size = tp x pp x dp = 6; expert layers nest EP=2 inside the DP
domain (expert_model_parallel_size = pp x etp x ep = 2, which divides 6).
Weights are ~61 GB bf16 — every full sync crosses the node link, which is
why the weight-update grace windows are sized in minutes.

## Environment prerequisites (trainer node)

- **Transformer Engine** must be importable (the Megatron backend needs
  it); build from source into the conda env if absent — see
  `docs/en/get-started/installation.md`.
- **mbridge / transformers 5.x**: transformers 5 removed
  `hf_config.rope_theta`, which mbridge 0.1.0 still reads. AstraFlow
  applies a runtime compat patch automatically
  (`astraflow/train_worker/models/mcore/mbridge_compat.py`) — no manual
  site-packages edit needed.
- **/dev/shm**: the sender double-buffers the full HF byte layout —
  budget >=2x61 GB on the trainer node.
- Rollout node needs **sglang >= 0.5.13** (native
  `--enable-return-routed-experts` support).

## Run (bring-up order)

This is a multi-node recipe, so there is no all-in-one launcher; start
the three components in order:

```bash
# Terminal 1 — trainer node (CPU): data service
bash examples/math/qwen3-30b-a3b-m2po/scripts/1_astraflow.sh

# Terminal 2 — rollout node: SGLang + R3 capture + TCP receiver
#   generic 4-GPU node:
bash examples/math/qwen3-30b-a3b-m2po/scripts/2_raas.sh
#   or the remote 4xB200 node:
ASTRAFLOW_URL=http://<trainer-host>:8000 \
  bash examples/math/qwen3-30b-a3b-m2po/scripts/2b_raas_b200.sh

# Terminal 3 — trainer node (6 GPUs): Megatron trainer
ASTRAFLOW_RAAS_URL=http://<rollout-host>:19190 \
  bash examples/math/qwen3-30b-a3b-m2po/scripts/3_trainer_model0.sh
```

Wait for each component to report ready before starting the next: the
RaaS self-registers against the AstraFlow URL, and the trainer's first
weight sync needs the RaaS receiver up.

Cross-node ports (verify reachability both ways before launching):
rollout -> trainer 8000 (AstraFlow), 19861 (sender HTTP), 21000
(handshake); trainer -> rollout 19190 (RaaS). Hostnames must resolve in
both directions or you must export IPs explicitly.

## What to watch during bring-up

- First weight sync: expect minutes, not seconds (~61 GB full sync,
  ~18.9k HF tensors). The grace windows allow 300 s.
- SGLang logs: any `routed_experts row-count mismatch` warning means the
  capture path is broken — stop and investigate.
- Trainer logs: with R3 on, the router-replay path asserts that recorded
  rows match the local token count; a shape assert here means
  token/routing misalignment, not a shape bug to paper over.
- Router entropy and train-infer KL: R3 should visibly reduce the
  rollout/training KL gap versus a run with `moe_router_replay: false`.

## Scaling up

Raise context by scaling these together: `sglang.context_length`,
`gconfig.max_new_tokens`, `rollout_dataset.max_length`,
`mb_spec.max_tokens_per_mb` (and eval dataset lengths). R3 adds ~1.5 KB
per token of rollout payload (48 layers x top-8 int32), so longer
contexts also mean proportionally heavier `/pull` traffic — keep
`max_concurrent_rollouts` bounded.
