# Chonk GRPO with NeMo Gym

This example uses Megatron training, async vLLM generation and packed NCCL
weight updates. A remote NeMo Gym server verifies the math answers.
The job starts its own Gym head, agent and policy adapter. It does not start
a local math verifier.

## Run

From the main example directory:

```bash
bash submit.sh vllm/nemo-gym
```

Set `NEMO_RL_IMAGE` as described in the [parent README](../../README.md#launch).

- Training: 16 nodes. Generation: 8 nodes. Each node has 4 GPUs.
- Two GRPO iterations, each with 32 questions and 8 responses per question.
- Container mounts and caches: embedded in `../../train.sh`.
- Math verifier: `https://gym-dev.swissai.svc.cscs.ch/math_with_judge`.

Check that the partition is available before submission. See the
[parent README](../../README.md) for training settings. The
[submission script](../../submit.sh) selects the allocation. The
[batch script](../../train.sh) uses the image's installed sources and Gym client.
The [recipe profile](recipe.env) selects training and resource defaults.

## Dataset and prompt

The launcher automatically prepares the full training and validation splits of
NVIDIA's [Nemotron-RL-math-OpenMathReasoning dataset](https://huggingface.co/datasets/nvidia/Nemotron-RL-math-OpenMathReasoning/tree/219bf0170cd87cd5db2bbf33715ce9ea43a2c59c),
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The pinned splits contain 112,867 training tasks and 30 validation tasks.
These are RL tasks with prompts and reference answers, not the original
OpenMathReasoning reasoning-trace corpus.

`prepare-data.sh` downloads the pinned files through the HF cache configured by
`HF_HUB_CACHE` in the container environment. Later jobs reuse the cached files.
It verifies each checksum and adds `agent_ref` for the
[Swiss AI Gym client](https://github.com/swiss-ai/nemo-gym/tree/swissai-k8s).
It keeps all original task fields and does not modify the cached originals.
The prepared JSONL files go in the job's `data/` directory on scratch.
No manual download or preparation command is required.

The bundled `math-smoke.jsonl` contains the first 500 training tasks.
Use it as a smaller training input with
`GYM_TRAIN_DATA="$PWD/vllm/nemo-gym/math-smoke.jsonl"` when submitting from the
parent example directory. Two GRPO iterations do not use the full split.

The launcher loads `math-chat-template.jinja` for policy tokenization and HTTP
serving. The template asks for a short solution and a boxed final answer.
The verifier uses math verification, not an LLM judge.

## Environment overrides

Set `MAX_NUM_STEPS` to change the maximum number of GRPO iterations
(default: `2`). For example:

```bash
MAX_NUM_STEPS=10 \
bash submit.sh vllm/nemo-gym --time=04:00:00
```

See the [shared settings](../../README.md#settings) for model,
batch, container and output settings. For optional W&B logging, put project,
entity, group and credentials in the shared `../../.env`; see
[W&B setup](../../README.md#local-environment-and-wb). This example also accepts:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GYM_MATH_URL` | Math verifier URL shown above | Remote math resources server. |
| `GYM_TRAIN_DATA` | `$RUN_DIR/data/openmathreasoning-train.jsonl` | Training input. A custom path skips automatic training-data preparation. |
| `GYM_VAL_DATA` | `$RUN_DIR/data/openmathreasoning-validation.jsonl` | Validation input. A custom path skips automatic validation-data preparation. |
| `GYM_VAL_PERIOD` | `0` | Run validation every N GRPO iterations. `0` disables scheduled validation. |

Use absolute paths for custom datasets. These files must already include a valid
`agent_ref` for the configured agent. To validate every five iterations:

```bash
GYM_TRAIN_DATA=/path/to/train.jsonl \
GYM_VAL_DATA=/path/to/validation.jsonl \
GYM_VAL_PERIOD=5 \
MAX_NUM_STEPS=10 \
bash submit.sh vllm/nemo-gym
```

Validation at the start and end remains disabled. Set `GYM_VAL_PERIOD` to a
positive value to use the validation split during training.

Set `CHONK_RUN_ROOT` to change the default artifact directory:
`/iopsstor/scratch/cscs/$USER/tmp/nemo-rl-nemo-gym-chonk/`.
Slurm's `--output` option selects the log file.

## Check verifier access

Run this command on a compute node. The gateway restricts access by source.

```bash
curl --fail --max-time 15 https://gym-dev.swissai.svc.cscs.ch/math_with_judge/
```

## Test result

Job `5046558` completed both GRPO iterations with this recipe.
Remote rewards, weight updates and policy training completed. The driver and
Slurm exit codes were zero.

Losses were `0.0134` and `0.0057`. Average rewards were `0.0078` and `0.0039`.
Generation KL errors were `0.0179` and `0.2499`.
This is an integration test, not a model-quality or numerical-parity test.
It does not test stateful environments or concurrent training jobs.
