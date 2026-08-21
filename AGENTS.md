# AGENTS.md

## Purpose

This repo builds, tests, and publishes Alps-optimized container images. Preserve reproducible content hashing, canonical base-image selection, generated child-pipeline behavior, validation markers, multi-node validation, and strict stable promotion.

## Project Map

- `Alps-Images/NGC/`: CUDA/NGC base-image family. Keep NVIDIA-specific assumptions here.
- `Alps-Images/ROCm/`: AMD ROCm base-image family for MI300-class systems. Keep ROCm SDK, RCCL/aws-ofi-rccl, and Beverin assumptions here.
- `Alps-Images/common/`: shared Alps network-stack installers, package helpers, runtime env fragments, and source/version defaults used by base-image families.
- `Alps-Images/apps/<app>/`: app Containerfiles, `profile.env`, optional app-local patches, and optional tests copied into `/opt/tests/<app>/`.
- `ci-pipelines/build-alps-extended-images.yaml`: executable parent CI pipeline source and best high-level CI overview; prefer this over README prose when they differ.
- `ci-pipelines/child-templates.yaml`: static templates used by generated child jobs for runner selection, build/test bodies, validation markers, and publish policy.
- `ci-pipelines/helpers/generate-child-pipeline.py`: emits the registry-dependent child pipeline.
- `ci-pipelines/helpers/meta.sh`: image refs, content hashes, validation hashes, and generated job env. `ci-pipelines/helpers/skopeo.sh`: registry digest, marker, copy, and promotion behavior.
- `manual-build/manual-build.sh`: local `podman build` script generation using the same ref/hash logic as CI.

## Core Invariants

- Content hashes must not include timestamps or commit SHAs. OCI labels may include CI metadata.
- Base hashes must cover the selected base Containerfile, family/profile inputs, shared common inputs, patches, and logical build inputs used by `meta.sh`.
- App hashes must cover selected Containerfile/profile/test/patch/helper inputs plus the canonical base ref, so unchanged app content rebuilds when its base changes.
- App images must build from the canonical base ref returned by the family base-ref helpers, never stable tags.
- Validation hashes must cover test metadata/templates/helpers. A tested marker is valid only when `<canonical-tag>-tested-<validation-hash>` points to the same digest as the canonical image.
- The generated child pipeline must emit concrete job `image:` refs and `needs:` based on registry state.
- App builds must depend on base validation, not only base build completion.
- Publish jobs must re-check tested markers and stable-tag policy at execution time because registry state can change after generation.
- Stable non-dev tags are immutable. Dev stable tags may overwrite only on the default branch.

## Family Boundaries

- Add new accelerator families as sibling family directories, and register their family-specific metadata/build/test/publish plumbing in `meta.sh`, `generate-child-pipeline.py`, and `child-templates.yaml`.
- Do not generalize CUDA-only assumptions such as `NVCR_PREFIX`, `REMOVE_HPCX_DIRS`, SM90 flags, NCCL/NVSHMEM, NVIDIA hooks, GH200 runners, or CUDA selector parsing.
- Keep shared Slingshot/CXI/HPC stack logic in `Alps-Images/common/`; keep accelerator-specific orchestration and pins in family directories or profiles.
- ROCm profiles own wheel indexes, target lists, and whether RCCL is rebuilt. Keep ROCm installers free of hardcoded package indexes if it can be avoided.

## Runtime Env

- Runtime env fragments use `defvar` defaults; preserve user overrides, including intentionally empty values.
- `source-alps-env.sh` must stay idempotent and safe for `/etc/profile.d`, NVIDIA entrypoint hooks, and non-interactive shells.
- Do not add `set -e`, `set -u`, or `pipefail` to scripts sourced through `BASH_ENV`.
- Keep shared runtime warning text in `alps-runtime-warning.txt` so Bash and Python diagnostics stay aligned.

## Container Build Notes

- Preserve `# syntax=docker/dockerfile:1`, `ARG BASE_IMAGE` followed by `FROM ${BASE_IMAGE}`, and OCI/CSCS labels in image Containerfiles.
- Source `Alps-Images/common/package-helpers.sh` before pip installs; use its `apt_cmd`, `apt_get`, `pip_install`, and cleanup helpers instead of open-coded package handling.
- Prefer pinned tags/commits for downloaded sources.
- Put app-local files copied into images under `Alps-Images/apps/<app>/sources/`; app hashes include this directory by default when it exists.
- Prefer app patch files under `Alps-Images/apps/<app>/patches/` over inline source rewrites in app Containerfiles.
- If tests must run without a repo checkout, copy them into `/opt/tests/<image>/` and include them in the app hash.

## Verification

- After shell edits: `git ls-files '*.sh' | xargs -r -n1 bash -n`.
- If ShellCheck is available, run it on changed shell scripts; no repo-local ShellCheck config exists.
- After CI YAML edits, parse `ci-pipelines/build-alps-extended-images.yaml` and `ci-pipelines/child-templates.yaml` locally, syntax-check embedded shell snippets, and use GitLab CI lint when credentials/network are available.
- For image-affecting changes, smoke-check `meta.sh` refs/hashes and `manual-build/manual-build.sh` script generation.

## Task-Specific Docs

- `README.md`: user-facing image matrix, runtime behavior, and CI overview.
- `manual-build/README.md`: local manual build workflow.
