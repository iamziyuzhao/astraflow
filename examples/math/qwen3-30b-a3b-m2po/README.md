# Qwen3-30B-A3B Math RL — Megatron MoE backend + Rollout Routing Replay (R3)

Math RL recipe for **Qwen3-30B-A3B** (MoE: 48 layers, 128 experts, top-8)
on a single 8xB200 node, cloned from
[`qwen3-8b-megatron-delta`](../qwen3-8b-megatron-delta) with these
deliberate differences:

1. **MoE parallelism** — Megatron backend with TP=1, PP=1, DP=4, EP=4,
   ETP=1 on 4 trainer GPUs; SGLang DP=2 on 2 rollout GPUs.
1. **R3 (Rollout Routing Replay)** — the SGLang server records which
   experts each token was routed to during rollout, and the trainer
   replays exactly that routing in its forwards, closing the
   rollout/training MoE mismatch that destabilises RL on MoE models.
1. **Closed-loop rollout buffer** — `dataflow.buffer.max_buffered_samples`
   pauses prompt submission while two training batches are already
   buffered, so sample staleness stays at 2–5 model versions instead of
   climbing to ~22 (see the header of `yaml/experiment_b200_1node.yaml`).
1. **GRPO objective** (clip 0.2/0.28, lr 1e-6, 256 samples per step, no
   KL term) — the recipe both the R3 paper and miles validated on this
   model. The directory keeps its historical `m2po` name; M2PO is off.
1. **Full (not delta) TCP weight transfer** — MoE per-step delta density
   is unmeasured; a delta that overflows its pre-allocated buffer falls
   back to full permanently anyway. Measure before switching back.

This configuration trained for 800 steps: MATH-500 avg@4 rose from 86.95
(step 0) to 92.35 (step 800, pass@4 97.0), still rising at the final eval.

## The three R3 flags

R3 must be enabled on all three sides at once (all set in this recipe):

| Flag                                                     | File                             | What it does                                                                                                                          |
| -------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `sglang.enable_return_routed_experts: true`              | `yaml/raas_b200_1node.yaml`      | SGLang (>=0.5.13) allocates its routed-experts capturer; requests may ask for per-token expert indices.                               |
| `raas.models.model0.gconfig.return_routed_experts: true` | `yaml/experiment_b200_1node.yaml` | Every rollout request asks for the recorded expert indices; they ride the rollout record into the training batch as `routed_experts`. |
| `trainer_base.actor.megatron.moe_router_replay: true`    | `yaml/experiment_b200_1node.yaml` | The trainer's MoE router replays the recorded top-8 expert indices in every training forward instead of re-deciding routing.          |

Supporting settings that are **not optional** with R3:

- `sglang.chunked_prefill_size: 32768` — sizes the capturer's device
  buffer; the AstraFlow default (`-1`, unchunked) undersizes it and
  crashes any large prefill batch mid-forward. `SGLangConfig.build_args`
  pairs the two automatically when the value is unset and refuses an
  explicit value that is too small.
- `sglang.moe_runner_backend: triton` — required on sm_100 (B200). The
  capture hook lives in `select_experts`, which the `flashinfer_trtllm`
  backend that `auto` picks on Blackwell bypasses: the server starts,
  reports healthy, and records nothing. `build_args` refuses `auto` on
  sm_100 and the known bypassing backends.
- `actor.megatron.use_deterministic_algorithms: true` — the replayed
  forward must be reproducible.

Keep speculative decoding, hierarchical cache, and PD disaggregation off
(all default off; `build_args` refuses them with R3), and leave
`disable_radix_cache` at its AstraFlow default (`true`).

## GPU layout (one node, 6 of 8 B200s)

| Component                 | GPUs       | Parallelism                   |
| ------------------------- | ---------- | ----------------------------- |
| AstraFlow HTTP service    | none (CPU) | —                             |
| Trainer model0 (Megatron) | 0-3        | TP=1, PP=1, DP=4, EP=4, ETP=1 |
| RaaS (SGLang, model0)     | 4-5        | DP=2, TP=1                    |

World size = tp x pp x dp = 4; expert layers nest EP=4 inside the DP
domain (128 experts / 4 = 32 per rank). Megatron reports 8.79 B params
per rank: the 28.99 B of experts split four ways plus 1.54 B of
non-expert weights replicated at TP=1. The optimizer state only fits with
the precision-aware settings in the experiment YAML (bf16 DDP gradient
buffer and first moment); the header there has the measured breakdown.

Weights are 56.9 GiB bf16; a full sync is a loopback TCP transfer of
~50 s plus ~15 s of SGLang load, against a 55–110 s training step, so
queued weight updates are coalesced to the newest version.

## Environment prerequisites

- **Transformer Engine** must be importable (the Megatron MoE path
  references `TENorm` unconditionally); build from source into the env
  if absent — see `docs/en/get-started/installation.md`.
- **mbridge / transformers 5.x**: transformers 5 removed
  `hf_config.rope_theta`, which mbridge still reads. AstraFlow applies a
  runtime compat patch automatically
  (`astraflow/train_worker/models/mcore/mbridge_compat.py`).
- **/dev/shm**: the sender double-buffers the full HF byte layout —
  budget >=2x57 GiB.
- **sglang >= 0.5.13** (native `--enable-return-routed-experts`).
- **Disk**: an HF save is 56.9 GiB and the recipe saves every 100 steps;
  `experiment.fileroot` must point at a filesystem with room for them.

## Run

Start the three components in order and wait for each to report ready:

```bash
# Terminal 1 (CPU): data service
bash examples/math/qwen3-30b-a3b-m2po/scripts/1_astraflow.sh

# Terminal 2 (GPUs 4,5): SGLang + R3 capture + TCP receiver
bash examples/math/qwen3-30b-a3b-m2po/scripts/2_raas.sh

# Terminal 3 (GPUs 0-3): Megatron trainer
bash examples/math/qwen3-30b-a3b-m2po/scripts/3_trainer_model0.sh
```

`SERVICE_CUDA_VISIBLE_DEVICES` and `TRAINER_MODEL0_GPUS` override the GPU
sets; `ASTRAFLOW_PORT`, `RAAS_PORT` and `WEIGHT_TRANSFER_HTTP_PORT_MODEL0`
the ports.

## What to watch

- First weight sync: ~2 min (the first pull competes with model loading);
  steady state ~76 s. The grace windows allow 300 s.
- `buffer/evicted` and `buffer/skipped_stale` must stay 0 and
  `sample_staleness_*` at 2–5: the closed loop is working. If the trainer
  logs "Waiting for data" every step, raise `rollout.max_concurrent_rollouts`
  in the RaaS YAML (in-flight generation is the supply bound), not
  `max_staleness`.
- RaaS log lines `notify_version: ... superseded` mean weight-update
  coalescing is working.
- SGLang logs: any `routed_experts row-count mismatch` warning means the
  capture path is broken — stop and investigate.
- Trainer logs: with R3 on, the router-replay path asserts that recorded
  rows match the local token count; a shape assert here means
  token/routing misalignment, not a shape bug to paper over.

## Known limitations

- `recover.mode` is `disabled`: a complete distributed checkpoint fails
  to load (`'Metadata' object has no attribute 'mcore_data'`), so HF
  saves (`saver.freq_steps`) are the resume path. Resuming means
  relaunching the whole stack from the saved HF weights; restarting only
  the trainer resets the version counter to 0 and RaaS refuses backwards
  loads.
- Context is 4096 (prompt <=1024 + 3000 new tokens). Raise
  `sglang.context_length`, `gconfig.max_new_tokens`,
  `rollout_dataset.max_length` and `mb_spec.max_tokens_per_mb` together.
  R3 adds 768 B per token of rollout payload (48 layers x top-8 int16).
