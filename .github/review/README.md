# PR reviews

## Purpose

OpenCode inspects PR changes and publishes code-review findings.
Explicit requests control when inference runs and which changes receive feedback.
Inspection gathers evidence from the supplied context and checkout.
A separate final-response phase writes the review from that evidence with provider tool selection disabled.
PR content is untrusted input, so model execution is separated from GitHub authorization and publication.
The reviewer inspects code but must not execute PR code or tests, install packages, change the checkout, or publish anything itself.
The review reports supported static findings and material gaps in inspected coverage, not a generic list of unperformed checks.
The grumpy reviewer checks correctness, security, performance, duplication, readability, and complexity, requiring concrete impact rather than personal preferences.
Claims about external API or platform behavior need authoritative evidence; unverified claims are dropped.
Web research is reserved for specific external-behavior questions that repository evidence cannot answer.
These instructions guide the review but do not guarantee coverage or finding quality.

## Review behavior

- The [review workflow](../workflows/opencode-review.yml) accepts `/oc-review` comments in a PR's Conversation tab, for both same-repository and fork PRs.
  Only newly created comments are processed.
- The requester must have effective repository write access.
  Maintainers and administrators qualify, including access inherited through teams, organizations, enterprises, or custom roles with a write base permission.
  GitHub's repository-permission API determines whether the requester can write to the repository.
- The workflow first checks whether the comment starts with `/oc-review`.
  [context.py](host/context.py) then validates the complete command and checks the requester's access.
  Comments without a valid command and requests from users without write access are skipped.
  A missing collaborator is treated as having no access, while other permission API errors fail authorization.
- Requests enter a per-PR concurrency group after authorization.
  Entering that group cancels any active review for the same PR.
  Preparation checks access again before fetching review metadata and history.
- The `acknowledge` job adds a 👀 reaction to an accepted request before review preparation starts.
  The reaction confirms acceptance, not successful completion.
  A reaction API error stops the review before inference.
- Each request selects the current head of an open PR through the GitHub API.
  Context includes PR metadata, the full merge-base diff, prior bot and human reviews, conversation comments, and inline threads with all replies.
  The diff starts at the merge base of the PR head and its target branch.
  Checkout and diff operations use commit SHAs, while branch names are display metadata.
  REST and GraphQL pagination preserve authors, timestamps, commit links, review states, and resolved or outdated thread status.
  Preparation fails if an API request fails or history is incomplete.
- The reviewer treats history and instructions found in PR files as material to inspect.
  It checks previous concerns against the current source, avoids repeating fixed findings, and does not treat approval or thread resolution as proof of correctness.
  Earlier reviews can be wrong and do not replace inspection of other changed logical areas.
  Repeat reviews use brief linked status bullets for earlier findings, separate from the current findings and without duplicating them.
  Previous reviews are not templates for the new review.

## Repository layout

| Location | Role |
| --- | --- |
| [`.github/workflows/`](../workflows/) | GitHub orchestration, job permissions, and checkout selection |
| [`host/`](host/) | Python helpers executed on the GitHub runner, with regression tests in `host/tests/` |
| [`image/`](image/) | Image definition, phase runner, shared reviewer policy, configuration, and local runtime tests |
| [`diagnostics/`](diagnostics/) | Opt-in local response recorder and its HTTPS fixture tests; not packaged into the image or run by the review workflow |

The Containerfile copies the runtime files explicitly; regression tests are not packaged into the image.
The runner executes `host/` helpers from the repository's default branch, not from the PR.
The container uses the software and configuration from the image pinned in the worker workflow.
Changes to these files in a PR are reviewed but are not used to run that review.

## Reviewer instructions

The reviewer combines three trusted instruction sources:

| Source | Purpose | Delivery |
| --- | --- | --- |
| [grumpy-reviewer.md](image/grumpy-reviewer.md) | Shared reviewer voice, checklist, evidence standards, and read-only rules | Baked into the image and referenced by both agents in `opencode.json` |
| Default-branch `AGENTS.md` | Repository-specific engineering and review guidance | Read-only runtime mount loaded through `opencode.json` |
| [run-review.mjs](image/run-review.mjs) | Inspection and final-response tasks | Baked into the image and passed as separate `opencode run` prompts in the same session |

[opencode.json](image/opencode.json) defines the two agents and their separate step caps.
The inspection agent cannot launch subagents.
The finalizer denies all tool permissions and selects the `review-final` model variant, which sets `tool_choice: none`.
The variant is needed because deny-all permissions alone still leave skill discovery advertised by the CLI.
Phase-specific evidence gathering and final-response policy live in [grumpy-reviewer.md](image/grumpy-reviewer.md#review-phases); the runner supplies the phase tasks.
The runner also includes that same policy file in the final task so earlier reviews and inspection summaries do not displace it in a long session.
[entrypoint.sh](image/entrypoint.sh) sets private runtime paths and starts the phase runner.

## Review limits

The agents' `steps` settings in [opencode.json](image/opencode.json) cap model iterations at 24 for inspection and 2 for finalization, not individual tool calls.
Each phase can finish before reaching its cap.
The finalizer reserves more than one iteration so its first response does not receive OpenCode's generic last-step summary prompt.
A tools-disabled response normally ends finalization on that first iteration.
New user input resets OpenCode's allowance, so the two configured caps must be considered together.
Recovery from empty, token-truncated responses can add provider requests beyond this step count.
The setting is therefore not a hard HTTP request-count quota.
The model and selected variant's `body.max_tokens` in [opencode.json](image/opencode.json) limit generation per request without changing the model's advertised capabilities.
The per-request output allowance is 16,384 tokens for inspection and 32,768 for finalization.
This is not a review-wide quota for input and output tokens.
Inspection has a 10-minute wall deadline, and finalization has a separate 5-minute wall deadline.
Each provider request has a 180-second wall deadline, including response-body consumption rather than only waiting for headers.
OpenCode may retry timed-out requests, but retries do not extend the phase deadlines.
A stopped phase has a 5-second termination grace before forced termination.
The host enforces a 17-minute container deadline as a backstop.
The workflow's job timeout remains a separate hard stop and does not guarantee a publishable final response.
An inspection deadline, exhausted iteration allowance, or token-truncated last inspection response can lead to finalization only when the session exists and a qualifying inspection tool has succeeded.
That path marks inspection incomplete; other inspection failures and any finalization failure do not produce a publishable fallback.

## Security and runtime layout

- The [review workflow](../workflows/opencode-review.yml) runs an `authorize` job to validate the command and check the requester's repository write access.
  Conversation-comment events load this workflow from the repository's default branch, currently `main`.
- Authorized requests call the [reusable worker](../workflows/opencode-review-worker.yml) from `main`.
  Its `acknowledge` job adds the reaction, `review` prepares context and runs the reviewer, and `publish` posts the findings.
  The reaction job has `issues: write` permission, and the publisher has `pull-requests: write` permission.
  Neither job runs the model or receives the inference key.
- The `authorize` and `review` jobs each check out the default branch into `default-branch/` and execute helpers from its `.github/review/host/` directory.
  The validated PR-head checkout is separate, in `pr-source/`.
- The [image definition](image/Containerfile) pins the reviewer software and installs it with its configuration, agent definition, and entrypoint under `/opt/reviewer/`.
  [opencode.json](image/opencode.json) defines the selected model and its provider connection settings.
  `OPENCODE_DISABLE_PROJECT_CONFIG=1` disables PR-supplied configuration, plugins, agents, and automatic instruction discovery.
- The worker passes `default-branch/AGENTS.md` explicitly to [run.py](host/run.py) with `--instructions`.
  The runner rejects instruction files inside the PR checkout, including paths that resolve there through symlinks.
  The runner mounts a read-only snapshot at `/instructions/AGENTS.md`, the path configured in `opencode.json`.
- The host stores prepared metadata and context under `$RUNNER_TEMP/opencode-review/input/` and captured events and review text under `$RUNNER_TEMP/opencode-review/model-output/`.
  Prepared context contains untrusted PR text, and model output is untrusted text even when produced by a successful run.

| Container path | Origin | Access |
| --- | --- | --- |
| `/opt/reviewer/` | Published image software and configuration | Read-only |
| `/instructions/AGENTS.md` | Default-branch repository instructions | Read-only runtime mount |
| `/input/pr-source/` | Validated PR-head checkout | Read-only runtime mount |
| `/input/context.md` | Prepared review context | Read-only runtime mount |
| `/work/`, `/tmp/` | Private runtime state and scratch space | Writable temporary filesystems |

The container runs as UID/GID 1000 with a read-only root filesystem, dropped capabilities, and `no-new-privileges`.
Only the inference key is forwarded from the host environment, with no GitHub credentials, SSH agent, host home directory, or container socket.
The runner compares `HEAD` and Git status before and after execution.
On failure or interruption, it attempts to remove the container.
Cleanup failures are logged without hiding the original failure, and checkout verification still runs.

## Review output

The `Run isolated CSCS reviewer` step logs fixed launch and exit-status labels and elapsed time, not the container command or captured stderr.
While waiting, it reports the current phase, captured-output byte counts, counts of known JSON event types, and the last known step-finish reason every 30 seconds.
The report distinguishes bookkeeping events such as `step_start` from `tool_use`, `text`, or `error` events.
Tool-result counts use fixed tool names and status labels, with unknown values reported as `other`.
Code Mode errors and nonzero shell exits count as errors even when OpenCode marks the call `completed`.
A final event summary is printed when the process exits, including failed or interrupted runs.
Reports continue when no output arrives, but do not prove that an inference request is making progress.
CI diagnostics use fixed labels, counts, and statuses; raw source, model content, credentials, tool arguments, tool errors, and tool output are not printed.
Raw events and stderr stay in private runner files (`events.jsonl` and `events.stderr.log`), not CI logs or artifacts.
The host writes sanitized counters, phase labels, elapsed time, exit status, and cleanup outcomes to `events.diagnostics.json`.
The workflow adds review and extraction outcomes and uploads a separate `opencode-review-diagnostics` artifact, including when extraction fails.
The CLI uses `--thinking` to include provider-exposed reasoning in the private event stream.
Reasoning is excluded from the published review.
Missing reasoning events do not prove that the model generated no reasoning.

The [final-response policy](image/grumpy-reviewer.md#final-response) uses short finding headings with separate location/evidence, consequence, and correction bullets.
Grumpy commentary stays brief, and each root cause is reported once.
Material uninspected areas get a short coverage notice; uncertainty about a finding stays beside it.
The review avoids setup diagnostics, generic verification checklists, and claims that uninspected code is clean.
Intentional temporary testing arrangements are not standalone defects without an unintended current consequence.

- [run-review.mjs](image/run-review.mjs) marks each phase's start and accepted end in the event stream.
  An inspection deadline, exhausted iteration allowance, or token-truncated last inspection response marks the accepted inspection end with `incomplete: true` only when there is a session and a successful qualifying tool call.
  Finalization reuses that session and remains tool-free.
  Other failed CLI exits, error events, unexpected sessions, and finalization tool attempts stop the review.
  The final-response task requires a fresh assessment of inspection notes under the review policy, not a copy of their proposed review.
  Inspection must complete a `read`, `grep`, `shell`, or Code Mode `execute` call successfully before finalization starts; glob-only discovery, web search, failed calls, and copied context do not satisfy this requirement.
  A successful tool call does not prove review coverage or correctness.
- [extract.py](host/extract.py) selects completed text only from the explicit final-response phase.
  Extraction requires ordered phase markers in the same session and rejects errors, missing phases, malformed events, or incomplete final output.
  An accepted incomplete-inspection marker prepends this fixed notice:

  > Inspection was interrupted.
  > Findings cover only the code inspected; this is not a clean bill of health for the remaining changes.

  This exception still requires a completed final response; it never publishes inspection notes.
  Inspection summaries and notes are never used as a publishable fallback.
  Only `review.md` is uploaded to the `opencode-review` artifact, with seven-day retention.
  Raw events and prepared context are not uploaded as artifacts.
- A separate publisher job reads `review.md` and posts it through the GitHub API.
  It has no source checkout or inference key and requests only `pull-requests: write` permission.
  It validates the PR number and reviewed SHA before inserting the review text into the comment body.
  Empty or oversized reviews fail rather than posting partial text, and oversized output remains in the artifact.
  Before posting, the publisher checks the current PR head and skips a stale result if it differs from the reviewed SHA.
  Published comments identify the reviewed commit and workflow run.
  The model supplies only the review body and must not reproduce the publisher's header or metadata from an earlier review.

## Local diagnostics

Use the local launcher when a review stalls or the normal progress counters do not explain what it is doing.
It prints OpenCode events, all-level CLI/server logs, complete provider requests, decoded response chunks, and request lifecycle timings as they arrive.
It retains the working state and captures on the host instead of losing them when the container exits.
GitHub reviews do not enable this mode or upload these files.

Export `CSCS_INFERENCE_API_KEY` and run with Python 3.12:

```sh
uv run --no-project --python 3.12 python .github/review/diagnostics/run-local.py \
  --engine podman \
  --image "$OPENCODE_REVIEW_IMAGE_PIN" \
  --checkout /path/to/pr-checkout \
  --context /path/to/prepared/context.md \
  --instructions /path/to/trusted/AGENTS.md \
  --output "$HOME/opencode-debug/run-1"
```

Use a new or empty output directory outside the source checkout for each run.
The context is the prepared PR input, not a prompt to fetch the PR or execute its code.
The instructions file is chosen explicitly by the local caller.
Pull a published image with your container engine before launching it; the diagnostic launcher does not refresh local images.
The launcher uses the selected image's configuration and changes only the provider URL to route requests through the recorder.
Models, agents, token allowances, and step caps are unchanged.
The source checkout remains read-only, while `/work` and the recording directory are writable host mounts.
Interrupting a run stops its container and retains captures already written.

The image must match the local machine's architecture.
On an ARM node, build a native image and pass `localhost/opencode-review:local-debug` to `--image`:

```sh
podman build --format docker \
  -f .github/review/image/Containerfile \
  -t localhost/opencode-review:local-debug .
```

| Path under the output directory | Contents |
| --- | --- |
| `recording/trace.log` | Provider request bodies, decoded response data, headers, and lifecycle timings |
| `recording/opencode.events.jsonl` | OpenCode JSON events, including exposed reasoning, tool inputs/results, model text, and phase markers |
| `recording/opencode.stderr.log` | CLI/server logs from `--log-level all --print-logs` |
| `recording/NNNN.request`, `NNNN.response`, `NNNN.json` | Per-request body, decoded response, and metadata |
| `work/` | Persistent OpenCode database, cache, state, and native log files |
| `staging/` | Context, instructions, original image configuration, recorder configuration, and launch wrappers |
| `image.json`, `invocation.json` | Image identity and the commands used |
| `engine.stdout.log`, `engine.stderr.log` | Container-engine output and the complete live terminal streams |
| `review.md` | Extracted final review, written only after successful completion |

The recorder masks authentication headers and the known inference key.
The private runtime configuration and native state are retained as-is.
Prompts, source code, tool payloads, and response content are not omitted.
Keep this directory local rather than publishing it as a GitHub artifact.
The logger cannot expose reasoning that the provider does not return.

Use the lifecycle messages to locate a delay:

- No provider request has started: inspect the CLI/server logs for startup or local tool activity.
- A request was uploaded but response headers have not arrived: the gateway or provider has not started its HTTP response.
  This can include queuing, prompt processing, or generation before the first response bytes.
- Raw response data arrives but OpenCode emits no events: inspect the CLI's handling or buffering of that response.
- A response ends with `max_tokens` and another request follows: the model exhausted that request's allowance and OpenCode continued.

`first_body_ms` measures response bytes, not necessarily the first generated token.
A `message_start` or keepalive frame does not mean model content has arrived.
A silent provider stream cannot distinguish queuing, prompt processing, or unexposed generation.
Errors and partial responses are retained.
The recorder's default mode still captures only response bodies and limited metadata without printing full prompts.

Run the recorder's isolated tests with `node --test .github/review/diagnostics/test-provider-trace.mjs`.
They use local HTTPS fixtures and do not call an inference service.
Run the phase runner's failure-boundary tests with `node --test .github/review/image/test-run-review.mjs`.
Those tests use local child processes and require no model credentials.

## Request a review

Post a new top-level comment in the PR's Conversation tab:

```text
/oc-review Focus on cache invalidation and registry permissions.
```

The command is case-insensitive, and request text is optional.
A space, tab, CR, or LF separates request text from the command.
After changing the PR, post another request when the updated code is ready for review.

## Configure inference access

Provide the repository or organization secret `CSCS_INFERENCE_KEY` to the caller workflow.
The worker supplies it only to the inference step as `CSCS_INFERENCE_API_KEY`.
The GitHub jobs use their automatically issued `GITHUB_TOKEN` for metadata requests, request reactions, and comment publication.

## Web search

[opencode.json](image/opencode.json) explicitly selects TinyFish so the headless worker does not prompt for a search provider.
Search queries can include repository-specific text and are sent to TinyFish, separately from CSCS model inference.
The worker supplies no TinyFish API key; the [OpenCode integration](https://opencode.ai/v2/docs/websearch) supports keyless access.
Treat keyless access as best-effort: TinyFish's [published account quotas](https://www.tinyfish.ai/pricing) do not establish a dedicated allowance for this repository.
The provider setting applies to `websearch`; `webfetch` remains a separate tool.

## Publish or update the image

The [image pipeline](../../ci-pipelines/build-opencode-review-image.yaml) builds on CSCS's AMD64/Zen2 builder and pushes directly to GHCR.
Register that YAML file in CSCS CI with the pipeline name `build-opencode-image`.
GitHub's AMD64 `ubuntu-latest` runner pulls and runs the public image pinned by digest in the worker workflow.
Changes to `host/` and the repository's `AGENTS.md` are deployed through the default branch and do not require an image rebuild.
The image and host helpers share a phase-marked JSONL event contract.
Publish and pin a compatible image before deploying host changes that require a new event format.

Supply these variables to both image-pipeline jobs through CSCS CI configuration:

- `GITHUB_ACTOR`: the GitHub username owning the registry token.
- `GITHUB_TOKEN`: a secret personal access token (classic) with `write:packages` and permission to publish under `eth-cscs`.
  Authorize organization SSO if required and avoid the unnecessary `repo` scope.

See [GitHub's registry authentication instructions](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-with-a-personal-access-token-classic) for token setup.

1. Select a new `OPENCODE_REVIEW_IMAGE` tag in the image pipeline when changing files under `image/` or the copied package helper.
2. Build from a reviewed commit, since the job receives registry write credentials.
   A user allowed to trigger this CSCS pipeline can post the PR comment `cscs-ci run build-opencode-image`, then run its manual `build-opencode-review-image` job.
   The builder's `if-not-exists` policy reuses existing versioned tags.
3. Set the GHCR package visibility to public so the reporting job can verify anonymous pull access.
   If the reporting job ran before visibility was changed, retry `report-opencode-review-image`.
4. The reporting job verifies AMD64 architecture and matching authenticated and anonymous digests.
   Set the worker's `OPENCODE_REVIEW_IMAGE` to `OPENCODE_REVIEW_IMAGE_PIN` from the reporting log or `opencode-review-image.env` artifact to deploy that image.

