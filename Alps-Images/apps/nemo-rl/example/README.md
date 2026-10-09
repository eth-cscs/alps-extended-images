# Chonk GSM8K GRPO on Alps

Megatron trains and vLLM generates; Ray coordinates the 24-node run. The driver runs
inside the Ray head's container so both share its local socket. Worker tasks
stay alive until the driver finishes and return its exit status. Bash prepares
job-local source copies. The GRPO YAML and prompt stay separate.

From this directory:

```bash
sbatch train-gsm8k-chonk.sh
```

The Slurm defaults use account `csstaff` and partition `normal`; override the
account with `sbatch --account=YOUR_PROJECT train-gsm8k-chonk.sh` if needed.

The script uses the included `environment.toml`, copied from your EDF. Set
`NEMO_RL_ENVIRONMENT=/path/to/other.toml` to use another EDF. The read-only default model is
`/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v2/hf/chonk/pretraining/long_context/long-context-iter-0003576`.
Sources default to the online Swiss-AI NeMo-RL, Megatron-Bridge, and Megatron-Core
repositories. No pre-existing checkout or developer scratch folder is required.
Each source also accepts a local Git checkout containing the pinned revision:

```bash
export NEMO_RL_SOURCE=https://github.com/swiss-ai/Nemo-RL.git
export BRIDGE_SOURCE=https://github.com/swiss-ai/Megatron-Bridge.git
export MEGATRON_SOURCE=https://github.com/swiss-ai/Megatron-LM-MoE.git
# Or use local paths, for example:
# export BRIDGE_SOURCE=$HOME/open_source/SwissAI-Megatron-Bridge
# export MEGATRON_SOURCE=$HOME/open_source/Megatron-LM-MoE
```

The selected repositories must contain the revisions listed below.
Override `HF_MODEL`, `MAX_NUM_STEPS`, `TRAIN_MICRO_BATCH`, `INFERENCE_REQUESTS`,
`NUM_PROMPTS`, or `NRL_MEGATRON_CHECKPOINT_DIR`. The default is two updates with
a two-hour time limit. Run data and logs go under
`/iopsstor/scratch/cscs/$USER/tmp/nemo-rl-gsm8k-chonk/`; checkouts and model files
are not changed. Slurm's startup log is
`/iopsstor/scratch/cscs/$USER/tmp/gsm8k-chonk-JOBID.out`; that scratch `tmp`
directory must exist before submission.

Topology: 8 training nodes/32 GPUs (EP16), 16 inference nodes/64 GPUs (EP8).
Defaults: microbatch 4/GPU, global batch 128, generation batch size 32,
32 prompts × 8 generations, 2 steps. Both sides use PP1. Training uses the
vLLM recipe's coarse-grained expert offloading: 8 chunks, 2 stages, in-place
FP8 parameters and extra BF16 storage. vLLM uses BF16, TP1/PP1/EP8 with DP8,
80% GPU memory utilization, dummy initial weights, and standard
`allgather_reducescatter` communication (no DeepEP/UCCL).
Generation is synchronous; CUDA graphs are allowed (`enforce_eager: false`),
and multiprocessing uses `spawn`. Weights come from the policy via NeMo's
NCCL collective and Bridge conversion; native MCore refit and M-to-N are not used.

The vLLM configuration is based on job 5003401, which reached generation but
timed out in Gloo DP communication. The communication backend is now standard
all-gather/reduce-scatter instead of DeepEP/UCCL (for now as we need to fix a bug idx ranking in NemoRL). No complete GRPO update was
confirmed; this backend change is not a validated fix for the Gloo timeout. Training microbatch remains 4
(the historical run used 1). Saves are disabled.

A small job-local overlay installs Lens and soundfile; compiler caches use
node-local `/tmp`.

Sources:
- [NeMo-RL](https://github.com/swiss-ai/Nemo-RL), [`f196ef0a`](https://github.com/swiss-ai/Nemo-RL/commit/f196ef0a1ccb54d6ce71cd63277fb6a0e01168db)
- [Megatron-Core](https://github.com/swiss-ai/Megatron-LM-MoE), [PR #88](https://github.com/swiss-ai/Megatron-LM-MoE/pull/88), `a9e0c497`
- [Megatron-Bridge](https://github.com/swiss-ai/Megatron-Bridge), [`f3bc8e7e`](https://github.com/swiss-ai/Megatron-Bridge/commit/f3bc8e7e6a722adde1176b2ba8980109e09493ef)

The launcher applies NeMo's [no-grad PR #5](https://github.com/swiss-ai/Nemo-RL/pull/5),
[tokenizer PR #1](https://github.com/swiss-ai/Nemo-RL/pull/1), and the full upstream
[vLLM DP PR #2517](https://github.com/NVIDIA-NeMo/RL/pull/2517) in its job-local
checkout. Conflicting PR2517 hunks use the upstream multiprocessing executor;
the fork's larger nightly-test budget is retained. Bridge applies its
[offloading-expert conversion PR #4](https://github.com/swiss-ai/Megatron-Bridge/pull/4)
in a separate job-local checkout. Source revisions and result trees are recorded;
there are no bundled patches or Python launcher helpers.
