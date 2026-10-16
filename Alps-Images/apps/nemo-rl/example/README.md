# Chonk GRPO on Alps

Three standalone Slurm/Ray examples use Megatron training:

| Directory | Generation backend | Configuration |
| --- | --- | --- |
| [`vllm/`](vllm/) | vLLM, standard all-gather/reduce-scatter | `grpo-chonk-vllm.yaml` |
| [`megatron/`](megatron/) | Megatron dynamic inference | `grpo-chonk-megatron.yaml` |
| [`nemo-gym-vllm/`](nemo-gym-vllm/) | vLLM, remote NeMo Gym math verifier | `grpo-chonk.yaml` |

The first two examples use GSM8K. The `nemo-gym-vllm/` example prepares the full
OpenMathReasoning RL training and validation splits. It also includes an optional
500-task smoke input.

Each GSM8K directory has its own Bash launcher, YAML and five-shot prompt.
The examples share this README and [`environment.toml`](environment.toml).
Neither GSM8K example needs files from the other backend directory.

## Submit

Choose a backend and submit from its directory:

```bash
cd vllm
sbatch train-gsm8k-chonk.sh
```

Or:

```bash
cd megatron
sbatch train-gsm8k-chonk.sh
```

The defaults are account `csstaff`, partition `normal` and 24 nodes.
GSM8K/vLLM defaults to 150 GRPO steps; the other examples default to two.
The launchers define the walltime defaults; override `--time` as needed for a
full run.
Use `sinfo -a` to check which partitions are available.
Set `--partition` and other Slurm options for your account:

```bash
sbatch --account=YOUR_PROJECT --partition=YOUR_PARTITION --time=02:00:00 train-gsm8k-chonk.sh
```

The driver runs in the Ray head's container. Worker tasks wait for the driver
to finish. They return the driver's exit status.

## Environment overrides

Set these variables before submission. They apply to all three examples.
For example, from a GSM8K example directory:

```bash
MAX_NUM_STEPS=10 \
sbatch --time=04:00:00 train-gsm8k-chonk.sh
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `MAX_NUM_STEPS` | `150` (GSM8K/vLLM), `2` (others) | Maximum GRPO iterations. The YAML epoch limit also applies. |
| `NUM_PROMPTS` | `64` (GSM8K/vLLM), `32` (others) | Prompts per GRPO iteration. Each produces 16 responses in GSM8K/vLLM, or 8 in the other examples. |
| `TRAIN_MICRO_BATCH` | `2` (GSM8K/vLLM), `1` (others) | Training microbatch size per GPU. |
| `INFERENCE_REQUESTS` | `32` | Generation batch size; also the native Megatron request limit. |
| `HF_MODEL` | Chonk path shown below | Read-only Hugging Face model directory. |
| `NEMO_RL_ENVIRONMENT` | `../environment.toml` | Container environment file, relative to the example directory by default. |
| `CHONK_RUN_ROOT` | Backend-specific scratch directory in the launcher | Directory for job artifacts and source checkouts. |
| `NRL_MEGATRON_CHECKPOINT_DIR` | `$RUN_DIR/hf-import` | HF-to-Megatron conversion directory. Reuse only a compatible conversion. |
| `MEGATRON_CONFIG_LOCK_DIR` | `/tmp/chonk-caches-JOBID/megatron-config-locks` | Directory for Bridge config locks. |
| `RAY_PORT` | `6379` | Ray head port. |
| `NEMO_RL_SOURCE`, `BRIDGE_SOURCE`, `MEGATRON_SOURCE` | Swiss AI repository URLs in the launcher | Git URLs or local checkouts that contain the required revisions. |
| `EXAMPLE_DIR` | Submission directory, or script directory outside Slurm | Example files to use. Set this if you submit from another directory. |

Batch sizes must satisfy both rules:

- The global batch must be divisible by `16 * TRAIN_MICRO_BATCH`.
- `NUM_PROMPTS * responses_per_prompt` must be divisible by the global batch:
  use 16 responses for GSM8K/vLLM and 8 for the other examples.

The launchers fix the node split at 16 training nodes and 8 generation nodes.
The global batch is 512 for GSM8K/vLLM and 128 for the other examples.
These are not environment overrides.
Use Slurm options such as `--account`, `--partition`, `--time` and `--output`
for scheduling and log settings.

## Optional W&B logging

W&B is disabled by default. All three launchers load an optional `.env` file
beside this README and `environment.toml`. From this README's directory, create
it without replacing an existing file:

```bash
cp -n .env.example .env
chmod 600 .env
```

Edit `.env` before submitting. Set `WANDB_ENABLED=true` and supply your settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `WANDB_ENABLED` | `false` | Enable W&B logging. |
| `WANDB_PROJECT` | `chonk-grpo` | Project that contains the runs. |
| `WANDB_ENTITY` | W&B account default | User or team namespace. |
| `WANDB_RUN_GROUP` | Unset | Group related runs within a project. |
| `WANDB_NAME` | `chonk-JOBID`, or `chonk-gsm8k-JOBID` for GSM8K/vLLM | Run display name. W&B creates a unique run ID separately. |
| `WANDB_API_KEY` | Unset | Credentials for online logging. Keep the real key private. |
| `WANDB_MODE` | `online` when enabled | Set `offline` to save locally without contacting W&B. |

Use shell-compatible `KEY=value` assignments. Quote values that contain spaces.
The launchers source this file, so use only a trusted file. Its values override
existing environment variables with the same names. The real `.env` is
Git-ignored; do not put credentials in YAML or commit them.

Run files go under the job's `nemo-rl-logs/` directory. The shared EDF defaults
`WANDB_CACHE_DIR` to `/ritom/scratch/cscs/${USER}/cache/wandb` so jobs can reuse
artifact-cache files. Shell startup settings can override this default.
Set a shared cache path in any custom EDF too.

Config, upload staging and downloaded-artifact directories stay in job scratch.
Override those paths with `WANDB_CONFIG_DIR`, `WANDB_DATA_DIR` and
`WANDB_ARTIFACT_DIR` if needed. Keep them outside `$HOME`.

## Shared configuration

- Training: 16 nodes / 64 GPUs, TP1/PP4/CP1/EP16, no virtual pipeline.
  The four pipeline stages contain 8/12/12/10 layers.
- Generation: 8 nodes / 32 GPUs, TP1/PP1/EP8. Generation uses separate nodes.
- Training microbatch/global batch: 2/512 for GSM8K/vLLM;
  1/128 for the other examples.
- Training-worker allocator: `PYTORCH_ALLOC_CONF=expandable_segments:True`.
  Generation workers do not inherit this setting.
- Rollout batch: 64 prompts × 16 generations = 1,024 responses for GSM8K/vLLM;
  32 prompts × 8 generations = 256 responses for the other examples.
- BF16 model parameters, FP32 training gradient buffers and synchronous GRPO.
  The KL penalty is 0.01. Checkpoint saves are disabled. The DeepGEMM FP8
  expert backward accumulation kernel needs FP32 gradient storage.
- Router weights and quantile-balancing thresholds (`qb_beta`) are frozen.
- Training expert offloading: coarse-grained, 8 chunks, 2 stages, in-place FP8
  parameters and extra parameter storage.
- Training dispatcher: `flex` with the image's UCCL-backed `deepep` interface.
  Token dispatch uses BF16. The examples disable expert-communication overlap
  and delayed weight-gradient computation. UCCL settings match
  [`pretrain/chonk.sh`](https://github.com/swiss-ai/pretrain/blob/6cb927e8b153e2fe9cd660fe0a6332c2ee7ce2a2/apertus-2/pretraining/chonk/chonk.sh):
  - `UCCL_EP_TRANSPORT=cxi`
  - `UCCL_EP_DISPATCH_CONFIG=24,40,512,32,512`
  - `UCCL_EP_COMBINE_CONFIG=24,7,512,32,512`
  - `UCCL_EP_CPU_TIMEOUT_SECS=6000`
  - `CUDA_DEVICE_MAX_CONNECTIONS=1`
  These settings apply to training workers only. Generation keeps its existing
  dispatcher.
- Weight refit: ordinary NeMo packed NCCL with Bridge conversion for all examples.

The default read-only HF model is:

```text
/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v2/hf/chonk/pretraining/long_context/long-context-iter-0003576
```

The launcher and EDF set the cache locations.

Bridge config locks are small lock files used when Megatron-Bridge loads a
Hugging Face model configuration, such as `config.json`. Processes that share
the lock directory take turns loading the same model's config. This prevents
races when they download or cache it. These files contain no model weights
and are not checkpoints.

The default directory is `/tmp/chonk-caches-JOBID/megatron-config-locks`.
It is node-local, so it does not coordinate processes on different nodes.
Set `MEGATRON_CONFIG_LOCK_DIR` to use another location.

## Generation backends

### vLLM

Generation is asynchronous. GRPO updates are synchronous.
The [config](vllm/grpo-chonk-vllm.yaml) and
[launcher](vllm/train-gsm8k-chonk.sh) define the GSM8K default recipe.

| Setting | GSM8K/vLLM default |
| --- | --- |
| Rollout | 64 prompts × 16 responses = 1,024 samples |
| Training batch | Global 512; microbatch 2/GPU; two updates per rollout |
| Run budget | 150 GRPO steps; at most three dataset epochs |
| Token limits | 1,024 generated; 2,048 total |
| Peak LR | Matrices `5e-6`; base and gains `2e-7` |
| LR schedule | Cosine; 300 updates total, first 30 warmup |
| Sampling | Temperature 1; top-p 1; model probabilities, not uniform sampling |
| Validation | Start, every 20 steps, end; one sampled response per question |
| Checkpoints | Disabled |

Set `GSM8K_VAL_PERIOD` to change the validation interval. Validation uses the
training sampler and the official GSM8K test split (1,319 questions).

#### Measured results

Reference run: [5066872](https://wandb.ai/apertus/smoke-apertus2-chonk-grpo/runs/qhoymofg).

| Metric | GRPO step | Result |
| --- | --- | --- |
| GSM8K validation accuracy | 0 | 46.63% |
| GSM8K validation accuracy | 120 | **71.87%** |
| Accuracy gain | 0 → 120 | **+25.2 percentage points** |
| Time per GRPO step | 135 | 121.77 s (≈2.03 min) |
| Policy training (two optimizer updates) | 135 | 45.01 s (≈22.5 s/update) |
| Rollout generation (1,024 responses) | 135 | 23.59 s |
| Policy/reference log-probability inference | 135 | 32.48 s |
| Weight refit | 135 | 9.76 s |
| End-to-end throughput | 135 | ≈5,850 tokens/s |
| End-to-end throughput per GPU | 135 | ≈60.9 tokens/s/GPU |

Throughput counts prompt and generated tokens across 96 GPUs.

### Megatron

The Megatron example uses native dynamic inference with local, layer-scope
CUDA graphs and MoE padding. With the current EP8/NCCL dispatcher, decode uses
CUDA graphs. Prefill runs without CUDA graphs.

## Validation

- Megatron: native generation and training completed two GRPO updates
  (job 5039868).
- vLLM: async generation and packed NCCL refit completed two GRPO updates
  (job 5045801).
- NeMo Gym vLLM: remote math verification and packed NCCL refit completed two
  GRPO updates (job 5046558); see its [validation notes](nemo-gym-vllm/README.md).

These are operational smoke tests, not model-quality or numerical-parity checks.

## Sources

The launchers define repository URLs, commit pins and PR refs. Use the launcher
as the source of truth for versions:

- [vLLM launcher](vllm/train-gsm8k-chonk.sh)
- [Megatron launcher](megatron/train-gsm8k-chonk.sh)
- [NeMo Gym launcher](nemo-gym-vllm/train-chonk.sh)

Each launcher downloads public repositories into the job's run directory.
It records the fetched revisions in `applied-prs.txt` and the source trees in
`source-provenance.txt`. Check these files to identify the code used by a run.
No developer checkout, bundled patch or Python launcher helper is needed.

The launchers install Lens and soundfile in a job-local package directory.
The Gym example also installs the packages in `gym-client-requirements.txt`.
Set `NEMO_RL_SOURCE`, `BRIDGE_SOURCE` or `MEGATRON_SOURCE` to use compatible Git
URLs or local checkouts. These checkouts must contain the required revisions.
