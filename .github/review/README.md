# PR reviews

## Purpose

OpenCode inspects requested PR changes and publishes code-review findings.
Inspection gathers evidence from the supplied context and checkout; a separate, tools-disabled phase writes the final response.
PR content is untrusted, so model execution is separated from GitHub authorization and publication.
The reviewer must not execute PR code or tests, install packages, modify the checkout, or publish anything itself.
The [reviewer policy](image/grumpy-reviewer.md) defines its grumpy voice, evidence standards, coverage requirements, and response format.
These instructions do not guarantee complete coverage or correct findings.

## Request a review

Post a new top-level comment in the PR's Conversation tab:

```text
/oc-review Focus on cache invalidation and registry permissions.
```

- Same-repository and fork PRs are supported.
  Inline diff comments and edits to existing comments do not trigger reviews.
- The command is case-insensitive, and request text is optional.
  A space, tab, CR, or LF separates request text from the command.
- The requester must have effective repository write access, as reported by GitHub's repository-permission API.
  Maintainers and administrators qualify, including access inherited through teams or roles with a write base permission.
- An `eyes` reaction confirms acceptance, not completion.
  A reaction API error stops the review before inference.
- A `confused` reaction indicates an acknowledgement, review, or publication failure, including a timeout.
  Open the run in the Actions tab and select **Re-run failed jobs**, or post a new command.
  A successful retry removes only the workflow's failure reaction; clicking a reaction does not restart the workflow.
- After updating the PR, post another command to request a review of the new head.
  Only authorized requests can cancel an earlier review for the same PR; other PRs run independently.
  Cancellation does not add a failure reaction, and all reaction updates depend on GitHub API access.

## Workflow and trust boundaries

The [caller workflow](../workflows/opencode-review.yml) runs an `authorize` job to validate the command and check write access.
Invalid commands and users without write access are skipped; a missing collaborator means no access, while other API errors fail authorization.
Authorized requests enter the per-PR concurrency group and call the [worker workflow](../workflows/opencode-review-worker.yml).

| Job | Responsibility | GitHub token permissions |
| --- | --- | --- |
| `authorize` | Validate the request before starting the worker | `contents: read` |
| `acknowledge` | React to the accepted request | `pull-requests: write` |
| `review` | Recheck access, prepare context, run the container, and extract review text | `contents: read`, `pull-requests: read` |
| `finish` | Publish a successful review or report failure on the request | `pull-requests: write` |

The write-enabled jobs have no source checkout, inference key, or model process.
The model receives no GitHub token.
The finishing job runs after failed dependencies too, but skips cancelled runs.
It also reports artifact-download and publication failures.

GitHub loads Conversation-comment workflows from the repository's **default branch**, currently `main`.
The caller selects the worker explicitly from `main`, and host helpers, configuration, and repository instructions are checked out from the default branch into `default-branch/`.
The PR's **base branch** is its merge target, which may differ from the default branch and is used to compute the review diff, not to select trusted helpers.
PR changes to workflows, helpers, configuration, or image sources are review material; they do not replace the code or settings running that review.

### Context and history

[context.py](host/context.py) selects the current head of an open PR through the GitHub API and verifies that the separate `pr-source/` checkout matches that SHA.
Context contains PR metadata, the full merge-base diff against the base branch, and the history collected by [history.py](host/history.py).
History includes bot and human reviews, Conversation comments, and inline threads with all replies, including authors, timestamps, commit links, and resolved or outdated status.
API failures, malformed pages, or incomplete history stop preparation rather than silently omit context.
Checkout and diff operations use validated commit SHAs; branch names are display metadata.
Git hooks and executable diff drivers are disabled when rendering context.
The reviewer treats prior discussions and source-local instructions as untrusted material and checks earlier findings against current code.

### Repository and image layout

| Location | Role |
| --- | --- |
| [`.github/workflows/`](../workflows/) | Authorization, job permissions, checkout selection, publication, and reactions |
| [`host/`](host/) | Python helpers executed on the GitHub runner; regression tests in `host/tests/` |
| [`config/`](config/) | Trusted runtime configuration from the default branch; not baked into the image |
| [`image/`](image/) | Image definition, runtime, reviewer policy, and phase-runner tests |

The [Containerfile](image/Containerfile) copies runtime files explicitly; tests are not packaged into the image.
The worker pulls the published image by digest, so changing a tag cannot change the image used by a review.

The reviewer receives these trusted instructions:

| Source | Responsibility | Delivery |
| --- | --- | --- |
| [grumpy-reviewer.md](image/grumpy-reviewer.md) | Voice, evidence, coverage, read-only rules, and final-response format | Baked into the image and used by both agents |
| Default-branch `AGENTS.md` | Repository-specific engineering guidance | Read-only runtime mount loaded through configuration |
| [run-review.mjs](image/run-review.mjs) | Phase selection, input location, current allowance, and incomplete-inspection status | Separate CLI tasks in the same session |

The phase runner reapplies the same policy file in the final task so earlier reviews and inspection summaries do not displace it in a long session.
[opencode.json](config/opencode.json) defines the model, provider, agents, permissions, and generation settings.
The provider uses CSCS's OpenAI-compatible Chat Completions API.
Its compatibility settings select `max_tokens` for the output limit and `reasoning` for streamed reasoning and retained assistant history.
The inspection agent cannot launch subagents.
The finalizer denies tool permissions and selects the `review-final` variant with provider-level `tool_choice: none`, since permissions alone still advertise skill discovery in the CLI.
[entrypoint.sh](image/entrypoint.sh) disables PR-supplied configuration, plugins, agents, and automatic instruction discovery with `OPENCODE_DISABLE_PROJECT_CONFIG=1` before starting the phase runner.

### Container isolation

[run.py](host/run.py) accepts repository instructions and the configuration directory separately from the PR checkout and rejects paths that resolve inside that checkout.
It reads one configuration snapshot, validates the execution limits, and stages read-only copies of the configuration, context, and instructions before launching the container.

| Container path | Origin | Access |
| --- | --- | --- |
| `/opt/reviewer/` | Published image runtime and reviewer policy | Read-only |
| `/opt/reviewer/config/opencode/` | Default-branch `config/opencode.json` and `config/limits.json` | Read-only mount |
| `/instructions/AGENTS.md` | Default-branch repository instructions | Read-only mount |
| `/input/pr-source/` | Validated PR-head checkout | Read-only mount |
| `/input/context.md` | Prepared review context | Read-only mount |
| `/work/`, `/tmp/` | Private runtime state and scratch space | Writable temporary filesystems |

The container runs as UID/GID 1000 with a read-only root filesystem, dropped capabilities, and `no-new-privileges`.
Only the inference key is forwarded from the host environment, without GitHub credentials, SSH agents, the host home directory, or a container socket.
The host compares `HEAD` and Git status before and after execution and attempts bounded process and container cleanup on failure or interruption.
Cleanup failures are logged, and checkout verification still runs.
Prepared inputs live under `$RUNNER_TEMP/opencode-review/input/`; captured output lives under `$RUNNER_TEMP/opencode-review/model-output/`.
Both PR context and model output remain untrusted text.

## Limits and output

Change runtime settings in [`config/`](config/) on the default branch to apply them to subsequent reviews without rebuilding the image.
Each run snapshots these files, so edits do not change an active review.

| Definition | Meaning |
| --- | --- |
| Agent `steps` in [opencode.json](config/opencode.json) | Maximum model iterations for each phase, not individual tool calls |
| Model and variant `body.max_tokens` in the same configuration | Per-request output allowances, including reasoning tokens, not a review-wide token quota |
| Model and variant `body.reasoning_effort` in the same configuration | Inspection effort and the finalizer's separate override |
| Provider `settings.headerTimeout` in the same configuration | Time allowed for response headers to arrive |
| Provider `settings.chunkTimeout` in the same configuration | Maximum wait between streamed response chunks |
| `inspection_seconds`, `finalization_seconds` in [limits.json](config/limits.json) | Separate phase deadlines |
| `execution_overhead_seconds` in the same file | Extra time for container startup and phase transitions |
| `termination_grace_seconds` in the same file | Grace before force-killing a stopped process |
| Worker `timeout-minutes` | GitHub job timeout |

Reasoning effort is a qualitative setting, not a token quota or completion-time guarantee.
There is no separate total-request deadline, so a response can keep streaming until its phase deadline.
Each phase can finish before reaching its iteration allowance.
Finalization starts a new message and has its own allowance, with room to avoid the CLI's generic last-step summary prompt on its first response.
Recovery from empty or token-truncated responses can add provider requests, so step limits are not hard HTTP request-count quotas.
Provider retries do not extend phase deadlines, and a job timeout does not guarantee a publishable response.
The host deadline is calculated as `inspection_seconds + finalization_seconds + 2 × termination_grace_seconds + execution_overhead_seconds`.
It grows automatically with the phase deadlines; leave time within the GitHub job timeout for setup, host cleanup, and artifact upload.
Durations in `limits.json` must be positive finite seconds; provider timeout settings use milliseconds.
The workflow's job timeout is configured separately; changing it alone does not extend inspection.

The image and host share a phase-marked JSONL event contract.
Inspection must complete a qualifying `read`, `grep`, `shell`, or Code Mode `execute` call successfully before finalization; this proves neither coverage nor correctness.
An inspection deadline, exhausted step allowance, or token-truncated response can proceed to finalization only with that evidence and an existing session.
The runner marks this path as incomplete, and extraction prepends a fixed coverage warning.
Other inspection failures and any finalization failure do not produce a publishable fallback.
[extract.py](host/extract.py) requires ordered phase markers and one session, and selects only completed final-response text, excluding reasoning, tool output, and inspection notes.
It rejects malformed events, unexpected sessions, tool use during finalization, and incomplete final output.

The finishing job reads `review.md` as text, validates the PR number and reviewed SHA, and rejects empty or oversized comments.
It rechecks the current PR head and skips stale reviews while retaining their artifact.
The publisher adds the reviewed commit and workflow-run link; the model supplies only the review body.

### Progress and failure diagnostics

The review step reports phase, elapsed time, captured byte counts, known event/tool counts, finish reason, and exit status.
Code Mode errors and nonzero shell exits count as errors even when OpenCode labels a call `completed`.
Heartbeats continue during silence but do not prove inference progress.
Logs contain fixed labels, counts, and statuses, not raw model text, tool arguments, source, or stderr.

The worker retains two separate artifacts with retention configured in its upload steps:

- `opencode-review` contains only `review.md` for publication.
- `opencode-review-diagnostics` contains sanitized counters, phase, exit and cleanup outcomes, review/extraction step outcomes, and bounded recent error details.
  Error details include type, HTTP status, explicit timeout classification, and a truncated message excerpt.
  Known credentials and common credential assignments are redacted before truncation, but masking cannot guarantee removal of arbitrary sensitive text.
  Structured response bodies, headers, stack traces, and raw stderr are excluded.

Inspect the diagnostic artifact's `errors` array before changing token, iteration, or time settings after a failure.
Raw events and stderr remain private runner files and are not uploaded, nor is prepared context.
Provider-exposed reasoning is captured in the private event stream but never published; missing reasoning events do not prove the model generated none.

## Deployment

Provide the repository or organization secret `CSCS_INFERENCE_KEY` to the caller workflow.
The worker exposes it only to the inference step as `CSCS_INFERENCE_API_KEY`; GitHub API operations use the jobs' automatically issued tokens.
For this workflow to run, its caller, worker, host helpers, and configuration must be on `main`, and the pinned image must be publicly pullable.
Changes to host helpers, runtime configuration, and repository instructions take effect through the default branch without rebuilding the image.
Image runtime and reviewer-policy changes still require a new image and digest pin.
The image requires the mounted configuration; there are no baked fallback settings.
Deploy compatible host and image code together when changing configuration-file formats or the event protocol.

### Web search

The configuration explicitly selects TinyFish so headless execution does not prompt for a provider.
Search queries may contain repository-specific text and are sent to TinyFish separately from CSCS inference.
The worker supplies no TinyFish API key; the [OpenCode integration](https://opencode.ai/v2/docs/websearch) supports keyless access.
This access is best-effort: [published account quotas](https://www.tinyfish.ai/pricing) do not establish a dedicated allowance for this repository.
The search-provider setting does not configure the separate `webfetch` tool.

### Publish or update the image

Register [build-opencode-review-image.yaml](../../ci-pipelines/build-opencode-review-image.yaml) in CSCS CI as `build-opencode-image`.
It builds on the AMD64/Zen2 builder and pushes directly to GHCR for GitHub's AMD64 `ubuntu-latest` runner.
Supply `GITHUB_ACTOR` and a secret `GITHUB_TOKEN` to both image-pipeline jobs through CSCS CI configuration.
The token must be a classic PAT with `write:packages` and permission to publish under `eth-cscs`; authorize organization SSO if required and avoid unnecessary `repo` scope.
See [GitHub's registry authentication instructions](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-with-a-personal-access-token-classic).

1. Select a new `OPENCODE_REVIEW_IMAGE` tag in the image pipeline when changing image inputs, including the copied package helper.
   Its `if-not-exists` policy reuses existing tags.
2. Build from a reviewed commit because the builder receives registry write credentials.
   An authorized user can post `cscs-ci run build-opencode-image`, then start the manual `build-opencode-review-image` job.
3. Set the GHCR package visibility to public so the reporting job can verify anonymous pulls.
   Retry `report-opencode-review-image` if it ran before visibility was changed.
4. The reporting job verifies AMD64 architecture and matching authenticated and anonymous digests.
   Set the worker's image to the reported `OPENCODE_REVIEW_IMAGE_PIN` from the log or `opencode-review-image.env` artifact.

## Local checks

Run host regressions with `pytest .github/review/host/tests` and phase-runner checks with `node --test .github/review/image/test-run-review.mjs`.
These checks use controlled fixtures and child processes, not inference credentials or live provider requests.
They run in preflight and are not included in the published image.
