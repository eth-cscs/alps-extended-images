# Chonk GRPO with NeMo Gym

This example uses Megatron training, async vLLM generation and packed NCCL
weight updates. A remote NeMo Gym resources server, running on the CSCS TDS
Kubernetes cluster (Argo CD project `rl-gym`), provides the environment:
`math` (stateless answer verification, default profile) or `blackjack`
(stateful multi-turn Gymnasium-style game, profile `blackjack`). The job
starts its own Gym head, agent and policy adapter. It does not start a local
resources server.

## Run

From the main example directory:

```bash
bash train.sh vllm/nemo-gym
# Or the blackjack environment:
bash train.sh vllm/nemo-gym --profile blackjack
```

No environment variables are required for either profile.

- Account: `csstaff`; partition: `normal`; time limit: two hours.
- Training: 16 nodes. Generation: 8 nodes. Each node has 4 GPUs.
- Two GRPO iterations, each with 32 questions and 8 responses per question.
- Shared container environment: `../../environment.toml`.
- Math verifier: `https://gym-math.shared-int-tds.tds.cscs.ch`.
- Blackjack environment: `https://gym-blackjack.shared-int-tds.tds.cscs.ch`.

Check that the partition is available before submission. See the
[parent README](../../README.md) for training settings. The
[shared launcher](../../train.sh) defines source pins and job-local package installs.
The [recipe profile](recipe.env) selects defaults and the NeMo PR stack;
[blackjack.env](blackjack.env) is the same stack with `GYM_ENV=blackjack`.
It keeps the image's Torch, Ray and OpenAI versions unchanged.

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

The `blackjack` profile skips dataset preparation and uses the bundled
`blackjack-smoke.jsonl` (128 identical rows: system prompt and "Deal me in.")
for training and validation; every reset deals a fresh random hand on the
server. The `gymnasium_agent` plays up to four turns per episode, threading
the session cookies so each step reaches the pod that holds the hand. Reward
is +1, 0 or -1 per hand. `blackjack-chat-template.jinja` renders the
multi-turn transcript for the base model.

## Environment overrides

Set `MAX_NUM_STEPS` to change the maximum number of GRPO iterations
(default: `2`). For example:

```bash
MAX_NUM_STEPS=10 \
bash train.sh vllm/nemo-gym -- --time=04:00:00
```

See the [shared settings](../../README.md#settings) for model,
batch, container and output settings. For optional W&B logging, put project,
entity, group and credentials in the shared `../../.env`; see
[W&B setup](../../README.md#local-environment-and-wb). This example also accepts:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GYM_ENV` | `math` (`recipe.env`) or `blackjack` (`blackjack.env`) | Environment. Selects `gym-$GYM_ENV.yaml`, `$GYM_ENV-chat-template.jinja` and the dataset defaults. |
| `GYM_URL` | URL for `GYM_ENV` shown above | Remote resources server (no path, no trailing slash). |
| `GYM_CONFIG` | `$EXAMPLE_DIR/gym-$GYM_ENV.yaml` | Gym config with the remote server and agent. |
| `GYM_CHAT_TEMPLATE_FILE` | `$EXAMPLE_DIR/$GYM_ENV-chat-template.jinja` | Chat template file. |
| `GYM_TRAIN_DATA` | math: `$RUN_DIR/data/openmathreasoning-train.jsonl`; blackjack: `$EXAMPLE_DIR/blackjack-smoke.jsonl` | Training input. A custom path skips automatic training-data preparation. |
| `GYM_VAL_DATA` | math: `$RUN_DIR/data/openmathreasoning-validation.jsonl`; blackjack: `$EXAMPLE_DIR/blackjack-smoke.jsonl` | Validation input. A custom path skips automatic validation-data preparation. |
| `GYM_VAL_PERIOD` | `0` | Run validation every N GRPO iterations. `0` disables scheduled validation. |
| `NEMO_GYM_SOURCE` | `https://github.com/swiss-ai/nemo-gym.git` | Gym Git URL or local checkout. |
| `NEMO_GYM_REF` | Pinned revision in the launcher | Gym revision. Use matching client dependencies. |

Use absolute paths for custom datasets. These files must already include a valid
`agent_ref` for the configured agent. To validate every five iterations:

```bash
GYM_TRAIN_DATA=/path/to/train.jsonl \
GYM_VAL_DATA=/path/to/validation.jsonl \
GYM_VAL_PERIOD=5 \
MAX_NUM_STEPS=10 \
bash train.sh vllm/nemo-gym
```

Validation at the start and end remains disabled. Set `GYM_VAL_PERIOD` to a
positive value to use the validation split during training.

Set `CHONK_RUN_ROOT` to change the default artifact directory:
`/iopsstor/scratch/cscs/$USER/tmp/nemo-rl-nemo-gym-chonk/`.
Slurm's `--output` option selects the log file.

## Check verifier access

Run this command on a compute node. The gateway restricts access by source.

```bash
curl --fail --max-time 15 https://gym-math.shared-int-tds.tds.cscs.ch/
curl --fail --max-time 15 https://gym-blackjack.shared-int-tds.tds.cscs.ch/
```

## Test result

Job `5046558` completed both GRPO iterations with this recipe.
Remote rewards, weight updates and policy training completed. The driver and
Slurm exit codes were zero.

Losses were `0.0134` and `0.0057`. Average rewards were `0.0078` and `0.0039`.
Generation KL errors were `0.0179` and `0.2499`.
This is an integration test, not a model-quality or numerical-parity test.
It does not test stateful environments or concurrent training jobs.
