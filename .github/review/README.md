# PR reviews

## Purpose

OpenCode inspects PR changes and publishes code-review findings.
Explicit requests control when inference runs and which changes receive feedback.
PR content is untrusted input, so model execution is separated from GitHub authorization and publication.
The reviewer inspects code but must not execute PR code or tests, install packages, change the checkout, or publish anything itself.
The review reports static findings and states any verification limits.

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
  It checks previous concerns against the current code, avoids repeating fixed findings, and does not treat approval or thread resolution as proof of correctness.

## Repository layout

| Location | Role |
| --- | --- |
| [`.github/workflows/`](../workflows/) | GitHub orchestration, job permissions, and checkout selection |
| [`host/`](host/) | Python helpers executed on the GitHub runner, with regression tests in `host/tests/` |
| [`image/`](image/) | Containerfile and reviewer configuration packaged into the image |

The runner executes `host/` helpers from the repository's default branch, not from the PR.
The container uses the software and configuration from the image pinned in the worker workflow.
Changes to these files in a PR are reviewed but are not used to run that review.

## Reviewer instructions

The reviewer combines three trusted instruction sources:

| Source | Purpose | Delivery |
| --- | --- | --- |
| [grumpy-reviewer.md](image/grumpy-reviewer.md) | Reviewer role, tone, and read-only review rules | Baked into the image and selected with `--agent grumpy-reviewer` |
| Default-branch `AGENTS.md` | Repository-specific engineering and review guidance | Read-only runtime mount loaded through `opencode.json` |
| [entrypoint.sh](image/entrypoint.sh) | Task, input paths, and output requirements | Baked into the image and passed as the `opencode run` prompt |

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

The `Run isolated CSCS reviewer` step logs the container launch, exit status, and elapsed time.
While waiting, it reports captured-output byte counts, counts of known JSON event types, and the last known step-finish reason every 30 seconds.
The report distinguishes bookkeeping events such as `step_start` from `tool_use`, `text`, or `error` events.
Tool-result counts use fixed tool names and status labels, with unknown values reported as `other`.
Code Mode errors and nonzero shell exits count as errors even when OpenCode marks the call `completed`.
A final event summary is printed when the process exits, including failed or interrupted runs.
Reports continue when no output arrives, but do not prove that an inference request is making progress.
Model text, reasoning content, tool arguments, tool errors, and tool output are not printed in these reports.

- [extract.py](host/extract.py) selects completed text parts from the final reviewer response.
  Extraction runs only after a successful reviewer exit and rejects missing, malformed or incomplete final output.
  Only `review.md` is uploaded to the `opencode-review` artifact, with seven-day retention.
  Raw events and prepared context are not uploaded as artifacts.
- A separate publisher job reads `review.md` and posts it through the GitHub API.
  It has no source checkout or inference key and requests only `pull-requests: write` permission.
  It validates the PR number and reviewed SHA before inserting the review text into the comment body.
  Empty or oversized reviews fail rather than posting partial text, and oversized output remains in the artifact.
  Before posting, the publisher checks the current PR head and skips a stale result if it differs from the reviewed SHA.
  Published comments identify the reviewed commit and workflow run.

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

