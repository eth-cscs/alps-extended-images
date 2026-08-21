---
name: submit-hpc-job
description: Submit a Slurm/sbatch script to a CSCS Alps HPC system via FirecREST and monitor it. Use when the user asks to submit, run, test, or re-submit an sbatch/Slurm script on Alps via FirecREST, or to check on / resume watching a job already submitted that way.
user-invocable: true
allowed-tools:
  - Agent
  - AskUserQuestion
  - Read
  - Bash(ls *)
  - Bash(grep *)
---

# /submit-hpc-job — Submit and monitor a Slurm job on CSCS Alps via FirecREST

Arguments passed: `$ARGUMENTS` (typically a script path, optionally with notes like an expected failure window, e.g. `Alps-Images/apps/verl/apertus-benchmarks/rl-bench-apertus-v1.5-70B-full-async.sh prior attempts failed within ~20 min`).

The actual submission and monitoring is done by the `firecrest-job` subagent (`~/.claude/agents/firecrest-job.md`) — this skill's job is to resolve which script, confirm before spending real cluster allocation, and dispatch with a well-scoped prompt. Do not reimplement the FirecREST calls here.

## Steps

1. **Resolve the script path.** If `$ARGUMENTS` names one, use it — resolve it relative to the current repo if it's a relative path, and confirm with `Read` or `ls` that it exists before going further. If no script was named, ask which one.

2. **Confirm before submitting — always, every time.** Submitting is not reversible and spends real shared-cluster allocation (this mirrors an explicit, standing instruction from prior sessions: "before submitting a new job ALWAYS ask permission"). Show the user:
   - The script path
   - Its `#SBATCH` header (nodes, account, time limit — `grep '^#SBATCH' <script>`)
   - Anything you know about why it's being run (a fix being verified, context from the conversation)

   Use `AskUserQuestion` (or plain confirmation if the user already clearly authorized this exact script earlier in the conversation — don't ask twice for the same submission). Do not proceed without a yes.

3. **Dispatch to the `firecrest-job` agent** (`Agent` tool, `subagent_type: "firecrest-job"`) with a self-contained prompt that includes:
   - The resolved script path (absolute)
   - Any context worth knowing: what past attempts of this same script have looked like (did they fail fast? at what point? what were prior errors?) — this shapes how long the agent should watch before giving up
   - Any non-default poll-window the user wants

4. **Relay the result.** The agent's report is already meant to be concise (job id, system, final/observed state, log path, first error if any) — pass it through, don't re-summarize into something longer. If the user will likely want to dig into the log themselves, mention the saved path (`~/Downloads/slurm-<job_id>.out`) explicitly.

## Resuming an existing monitor

If the user asks to keep watching a job you (or a prior turn) already dispatched, resume that same agent via `SendMessage` to its name/id rather than launching a fresh `firecrest-job` dispatch — a fresh dispatch has no memory of the job id or prior context, and dispatching two monitors for the same job risks confusing, duplicate polling (this happened once — see the "Background polling" section of the `firecrest-job` agent definition for how that was fixed).
