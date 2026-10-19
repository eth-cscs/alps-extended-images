# Chonk GRPO with NeMo Gym

This example uses Megatron training, async vLLM generation and packed NCCL
weight updates. A remote NeMo Gym resources server, running on the CSCS TDS
Kubernetes cluster (Argo CD project `rl-gym`), provides the environment:
`math` (stateless answer verification, default) or `blackjack` (stateful
multi-turn Gymnasium-style game). The job starts its own Gym head, agent and
policy adapter. It does not start a local resources server.

## Run

From the parent example directory:

```bash
cd nemo-gym-vllm
sbatch train-chonk.sh
```

No environment variables are required for the default (math) setup. For the
blackjack environment:

```bash
GYM_ENV=blackjack sbatch train-chonk.sh
```

- Account: `csstaff`; partition: `normal`; time limit: two hours.
- Training: 16 nodes. Generation: 8 nodes. Each node has 4 GPUs.
- Two GRPO iterations, each with 32 questions and 8 responses per question.
- Shared container environment: `../environment.toml`.
- Math verifier: `https://gym-math.shared-int-tds.tds.cscs.ch`.
- Blackjack environment: `https://gym-blackjack.shared-int-tds.tds.cscs.ch`.

Check that the partition is available before submission. See the
[parent README](../README.md) for training settings. The
[launcher](train-chonk.sh) defines source pins and job-local package installs.
It keeps the image's Torch, Ray and OpenAI versions unchanged.

## Dataset and prompt

`math-smoke.jsonl` contains the first 500 training tasks from NVIDIA's
[Nemotron-RL-math-OpenMathReasoning dataset](https://huggingface.co/datasets/nvidia/Nemotron-RL-math-OpenMathReasoning/tree/219bf0170cd87cd5db2bbf33715ce9ea43a2c59c),
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
It keeps the original questions, answers and prompts. Each row adds `agent_ref`
for the [Swiss AI Gym client](https://github.com/swiss-ai/nemo-gym/tree/swissai-k8s).
The subset is included in this directory. No dataset download is needed.
Two iterations do not use all 500 questions.

The launcher loads `math-chat-template.jinja` for policy tokenization and HTTP
serving. The template asks for a short solution and a boxed final answer.
The verifier uses math verification, not an LLM judge.

`blackjack-smoke.jsonl` holds 128 identical rows (system prompt and "Deal me
in."); every reset deals a fresh random hand on the server. The
`gymnasium_agent` plays up to four turns per episode, threading the session
cookies so each step reaches the pod that holds the hand. Reward is +1, 0 or
-1 per hand. `blackjack-chat-template.jinja` renders the multi-turn
transcript for the base model.

## Environment overrides

Set `MAX_NUM_STEPS` to change the maximum number of GRPO iterations
(default: `2`). For example:

```bash
MAX_NUM_STEPS=10 \
sbatch --time=04:00:00 train-chonk.sh
```

See the [shared override table](../README.md#environment-overrides) for model,
batch, container and output settings. This example also accepts:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GYM_ENV` | `math` | Environment: `math` or `blackjack`. Selects `gym-$GYM_ENV.yaml`, `$GYM_ENV-smoke.jsonl` and `$GYM_ENV-chat-template.jinja`. |
| `GYM_URL` | URL for `GYM_ENV` shown above | Remote resources server (no path, no trailing slash). |
| `GYM_CONFIG` | `$EXAMPLE_DIR/gym-$GYM_ENV.yaml` | Gym config with the remote server and agent. |
| `GYM_CHAT_TEMPLATE_FILE` | `$EXAMPLE_DIR/$GYM_ENV-chat-template.jinja` | Chat template file. |
| `GYM_TRAIN_DATA` | `$EXAMPLE_DIR/$GYM_ENV-smoke.jsonl` | Gym-format training dataset. |
| `NEMO_GYM_SOURCE` | `https://github.com/swiss-ai/nemo-gym.git` | Gym Git URL or local checkout. |
| `NEMO_GYM_REF` | Pinned revision in the launcher | Gym revision. Use matching client dependencies. |

To use another dataset:

```bash
GYM_TRAIN_DATA=/path/to/train.jsonl sbatch train-chonk.sh
```

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

Job `5046558` ran this repository launcher and completed both GRPO iterations.
Remote rewards, weight updates and policy training completed. The driver and
Slurm exit codes were zero.

Losses were `0.0134` and `0.0057`. Average rewards were `0.0078` and `0.0039`.
Generation KL errors were `0.0179` and `0.2499`.
This is an integration test, not a model-quality or numerical-parity test.
It does not test stateful environments or concurrent training jobs.
