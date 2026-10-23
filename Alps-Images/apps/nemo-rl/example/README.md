# Chonk GRPO on Alps

## Choose a recipe

| Recipe | Profile | Configuration | Defaults |
| --- | --- | --- | --- |
| `vllm/gsm8k` | `adamw` | [YAML](vllm/gsm8k/config.yaml) | [adamw.env](vllm/gsm8k/adamw.env) |
| `vllm/gsm8k` | `muon` | [YAML](vllm/gsm8k/config.yaml) | [muon.env](vllm/gsm8k/muon.env) |
| `megatron/gsm8k` | `adamw` | [YAML](megatron/gsm8k/config.yaml) | [adamw.env](megatron/gsm8k/adamw.env) |
| `megatron/gsm8k` | `muon` | [YAML](megatron/gsm8k/config.yaml) | [muon.env](megatron/gsm8k/muon.env) |
| `vllm/nemo-gym` (k8s) | `recipe` (default) | [YAML — sync GRPO](vllm/nemo-gym/grpo-chonk.yaml) | [recipe.env](vllm/nemo-gym/recipe.env) |

The GSM8K [shared YAML](common/gsm8k.yaml) defines training and data settings. Each
backend YAML defines its generation engine. Each `.env` profile is self-contained.

## Launch

Set `NEMO_RL_IMAGE` to the image built with the consolidated training stack
(in your shell or `.env`). The older runtime-patched image cannot run this launcher.
From this directory, choose one:

```bash
bash submit.sh vllm/gsm8k --profile muon \
  --grpo-steps 58 --prompts 128 --generations-per-prompt 16 --updates-per-rollout 4 \
  --account=YOUR_PROJECT --partition=YOUR_PARTITION --time=07:00:00
# Or use native generation:
bash submit.sh megatron/gsm8k --profile adamw \
  --account=YOUR_PROJECT --partition=YOUR_PARTITION --time=07:00:00
```

`submit.sh` reads the profile and training options, then passes the other arguments
to `sbatch`. No `--` separator is required. The account and partition have no
launcher defaults; select them on the command line or use your Slurm defaults.
Training option names, including `--profile`, belong to `submit.sh`. An optional
`--` passes all remaining arguments directly to Slurm if a name conflicts.

Run `bash submit.sh --help` for all options. CLI training values override `.env`
and profile defaults. You can call the script by its full path from another directory.

The two scripts have separate roles:
- **`submit.sh`:** runs on the login node, selects resources, submits `train.sh`,
  prints the job ID and exits. You can close the login shell after submission.
- **`train.sh`:** runs inside the Slurm allocation. It starts containers, Ray and
  training with `srun`. It never calls `sbatch`.

For GSM8K, AdamW uses 8 training nodes and 4 generation nodes. Muon uses 16
training nodes and 4 generation nodes. `submit.sh` sets the total node count and
time from the profile. Fixed per-node defaults are `#SBATCH` directives in
`train.sh`; Slurm command-line options can override them.
Check partition availability with `sinfo -a` before submitting.

### Direct `sbatch` (baked sources only)

For a run that uses the image's installed sources, you can skip `submit.sh`.
Run from this example directory:

```bash
sbatch --account=YOUR_PROJECT --partition=YOUR_PARTITION \
  --nodes=12 --time=07:00:00 \
  train.sh vllm/gsm8k --profile adamw
```

Logs default to `/iopsstor/scratch/cscs/%u/tmp/%x-%j.out`, where `%u` is your
username, `%x` is the job name and `%j` is the job ID. Use `--output` to change it.
Slurm requires the parent directory to exist; `submit.sh` creates the default
directory automatically.

Put Slurm options before `train.sh`, and the recipe and profile after it.
Read the experiment's `.env` profile for its node counts and time limit.
Use those values to set the Slurm allocation.

Direct runs reuse the settings function in `submit.sh`; sourcing it does not
submit a job. Training options go after the recipe, for example `--grpo-steps 3`.
The job reads `.env` and its profile when it starts. Set `CHONK_EXAMPLES_DIR` to
this directory if you submit from another working directory.

Use `submit.sh` for development source overrides and automatic resource selection.

## Development overrides

Use `submit.sh` for custom sources, PRs and patches. Replace `COMPONENT` with
`nemo`, `core`, `bridge` or `gym`:

| Option | Purpose |
| --- | --- |
| `--dev-COMPONENT-checkout PATH` | Use a local Git checkout, including working-tree changes. |
| `--dev-COMPONENT-ref REF` | Use a commit, branch or tag; use a SHA for repeatable runs. |
| `--dev-COMPONENT-repo URL` | Select another repository with a ref; defaults are the Swiss AI forks. |
| `--dev-COMPONENT-pr NUMBER_OR_URL` | Merge a PR from that repository, or use a full GitHub PR URL. |
| `--dev-COMPONENT-merge REF` | Merge another branch or commit. |
| `--dev-COMPONENT-patch FILE` | Apply a patch file. |

```bash
bash submit.sh megatron/gsm8k --profile muon \
  --dev-nemo-checkout /path/to/nemo-rl \
  --dev-nemo-pr 5 --dev-nemo-pr 1 \
  --dev-nemo-patch /path/to/fix.patch \
  --account=YOUR_PROJECT --partition=YOUR_PARTITION
```

Each library you patch or merge must have its own `--dev-COMPONENT-checkout`
or `--dev-COMPONENT-ref`.
Unselected libraries still use the image's installed versions. Repeat merge and
patch options as needed; PR merges run first, then patches, each in the listed order.
A conflict stops the job before training.

Keep local checkouts and patch files on shared storage. Repositories and PRs
can use GitHub URLs. At job start, sources are cloned or copied under
`$RUN_DIR/dev/` without changing your checkouts. Select submodules separately.
See `development-provenance.txt` for the resolved versions and patch checksums.
Source versions must work with the image's Python/CUDA stack; overrides do not
install dependencies or rebuild extensions.

## Shared training configuration

- UCCL provides the DeepEP-compatible training dispatcher over Slingshot/CXI
  (`UCCL_EP_TRANSPORT=cxi`). Token dispatch uses BF16.
- Router weights are frozen (`freeze_moe_router`), as are quantile-balancing
  thresholds (`moe_router_quantile_balancing_freeze`).

## Settings

- Edit the shared or backend **YAML** for structured model and engine settings.
- Edit the selected **`.env` profile** for resource, batch, learning-rate and run-budget defaults.
- Use `--micro-batch`, `--lr`, `--matrix-lr`, `--gains-lr`, `--train-nodes` or
  `--generation-nodes` to override those defaults for one run.
- `NEMO_RL_IMAGE` selects the container image. [train.sh](train.sh) embeds the
  mounts and cache settings and writes a job-local `environment.toml`.
- `HF_MODEL` overrides the read-only starting checkpoint.
- `CHONK_RUN_ROOT` selects where job directories, checkpoints and logs are written.
  The initial HF-to-Megatron conversion is stored under each job's `hf-import/`.
- Global batch is derived as `prompts × generations per prompt ÷ updates per rollout`.
  The launcher rejects fractional batches. NeMo-RL/Megatron checks parallelism
  and microbatch compatibility.
- The GSM8K default budget is **58 GRPO steps × 4 updates = 232 optimizer updates**.
  The requested GRPO step count controls the run; there is no separate one-epoch cap.

The default HF checkpoint is:

```text
/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v2/hf/chonk/pretraining/long_context/long-context-iter-0003576
```

### Local environment and W&B

Create a local settings file without replacing an existing one:

```bash
cp -n .env.example .env
chmod 600 .env
```

Set `WANDB_ENABLED=true`, `WANDB_PROJECT`, and `WANDB_API_KEY` in `.env`.
Set `WANDB_ENTITY` if you need a specific user or team namespace.
You can also set a run root there, for example:

```bash
CHONK_RUN_ROOT=/ritom/scratch/cscs/$USER/tmp/chonk-grpo/runs
```

The launcher sources `.env`, so use only a trusted file. Its values override
matching environment variables. It is Git-ignored; never commit API keys.
Keep run directories and caches outside `$HOME`.

## GSM8K

The active recipes validate on all 1,319 GSM8K test questions.
`GSM8K_VAL_PERIOD` overrides the validation interval. Weight-checkpoint settings
include a final save, no optimizer state, a save period of
1,000 steps, and a checkpoint deadline of 4h40m. All outputs stay under the run root.

By default, each GSM8K rollout batch has **128 prompts × 16 responses = 2,048 samples**.
It is split into **four groups of 512 samples**, with one optimizer update per
group: **four optimizer updates per GRPO step**. These are distinct training
batches, not four passes over the same samples.

GSM8K has 7,473 training prompts. With `drop_last=True`:

| Prompts per GRPO step | GRPO steps per epoch | Epochs at 150 steps |
| --- | --- | --- |
| 64 | 116 | 1.29 |
| 128 | 58 | 2.59 |

Epochs count prompt batches, regardless of responses per prompt or optimizer updates.

### Reference results

These supplied reference results are not measurements of the local examples.

| Metric | vLLM AdamW | Megatron AdamW | vLLM MD | Megatron MD |
| --- | --- | --- | --- | --- |
| Job ID | 3686420 | 3686429 | 3686113 | 3689066 |
| Training + generation nodes | 8 + 4 | 8 + 4 | 16 + 4 | 16 + 4 |
| Run time (min) | 78 | 132 | 74 | 110 |
| Evaluation before | 49.1% | 48.1% | 48.2% | 47.2% |
| Evaluation after | **90.7%** | **91.1%** | **78.5%** | **78.7%** |
| `gen_kl` max | 0.0021 | 0.0018 | 0.0007 | 0.0006 |
| Trainer time (s) | 60.8 | 59.4 | 55.7 | 54.5 |
| Generation time (s) | 33.9 | 174 | 33.2 | 143 |
| Throughput (tokens/s/GPU) | 432 | 322 | 263 | 206 |
| Time per GRPO step (s) | 69.8 | 125.9 | 65.9 | 106.7 |
| Average refit time (s) | 7.6 | 3.8 | 8.7 | 2.8 |
| Trajectory age (steps) | 1 | 1 | 1 | 1 |

## NeMo Gym (k8s) results (synchronous GRPO)

Run [5066314 / cfgsgu5r](https://forge.coreweave.com/wandb/apertus/smoke-apertus2-chonk-grpo/runs/cfgsgu5r).
Results below are from GRPO step 245 unless another step is stated.
Training reward is not validation accuracy.

| Metric | Result |
| --- | --- |
| Optimizer | Muon / MD-decoupling |
| Logged GRPO steps | 245 |
| Optimizer updates | 490 |
| Prompts × responses per rollout | 128 × 8 |
| Global batch | 512 |
| Beginning training reward (step 1) | **1.37%** |
| Last training reward (step 245) | **5.57%** |
| Loss | 0.0286 |
| Generation KL error | 0.0293 |
| Time per GRPO step (s) | 95.90 |
| Trainer time, two optimizer updates (s) | 28.89 |
| Generation time (s) | 18.15 |
| Policy/reference log-probability inference (s) | 26.09 |
| Refit time (s) | 10.36 |

## Source revisions

The image installs the fixed revisions in
[`install-training-stack.sh`](../sources/install-training-stack.sh), including NeMo PRs
5, 1, 6, 7 and 9, the PR 8 cache fix, and the tested upstream PR 2517 revision.
The [NeMo patches](../patches/nemo-rl/) and [Core patches](../patches/megatron/)
are applied when building the image, not when launching a job. NeMo-RL, Core,
Bridge and the Gym client are already installed; normal runs need no runtime
Git access or package installation.

The launcher copies the image's `source-provenance.txt` and `applied-patches.txt`
to the run directory. Recipe YAML and profiles remain editable outside the image.
The default HF checkpoint is read-only.

## Run more experiments

For running and comparing RL experiments, try [rl-infra](https://github.com/swiss-ai/rl-infra),
created by [@davmels](https://github.com/davmels).
