# AGENTS.md

## Purpose

This repo builds, tests, and publishes Alps-optimized container images. Preserve reproducible content hashing, canonical base-image selection, CI retagging for tests, multi-node validation, and strict stable promotion.

## Project Map

- `Alps-Images/NGC/`: CUDA/NGC base-image family. Keep NVIDIA-specific assumptions here.
- `Alps-Images/ROCm/`: AMD ROCm base-image family for MI300-class systems. Keep ROCm SDK, RCCL/aws-ofi-rccl, and Beverin assumptions here.
- `Alps-Images/common/`: shared Alps network-stack installers, package helpers, runtime env fragments, and source/version defaults used by base-image families.
- `Alps-Images/apps/<app>/`: app Containerfiles, `profile.env`, optional app-local patches, and optional tests copied into `/opt/tests/<app>/`.
- `ci-pipelines/build-alps-extended-images.yaml`: executable CI pipeline source; prefer this over README prose when they differ.
- `ci-pipelines/helpers/meta.sh`: image refs and content hashes. `ci-pipelines/helpers/skopeo.sh`: registry copy and promotion behavior.
- `manual-build/manual-build.sh`: local `podman build` script generation using the same ref/hash logic as CI.

## Core Invariants

- Content hashes must not include timestamps or commit SHAs. OCI labels may include CI metadata.
- Base hashes must cover the selected base Containerfile, family/profile inputs, shared common inputs, patches, and logical build inputs used by `meta.sh`.
- App hashes must cover selected Containerfile/profile/test/patch/helper inputs plus the canonical base ref, so unchanged app content rebuilds when its base changes.
- App images must build from the canonical base ref returned by the family base-ref helpers, never stable tags or commit-SHA test tags.
- GitLab resolves job `image:` before dotenv artifacts exist; tests must use predictable commit-SHA test tags created by `tag-*-for-ci` jobs.
- Matrix jobs with `needs:` must map every identity field through `needs:parallel:matrix` to avoid mixed dotenv artifacts.
- Publishing is gated by explicit `publish-gate.needs`; update it whenever tests are added, removed, or renamed.
- Stable non-dev tags are immutable. Dev stable tags may overwrite only on the default branch.

## Family Boundaries

- Add new accelerator families as sibling family directories and family-specific metadata/build/test/publish plumbing.
- Do not generalize CUDA-only assumptions such as `NVCR_PREFIX`, `REMOVE_HPCX_DIRS`, SM90 flags, NCCL/NVSHMEM, NVIDIA hooks, GH200 runners, or CUDA selector parsing.
- Keep shared Slingshot/CXI/HPC stack logic in `Alps-Images/common/`; keep accelerator-specific orchestration and pins in family directories or profiles.
- ROCm profiles own upstream base refs, ROCm versions, wheel indexes, target lists, and whether RCCL is rebuilt. Keep ROCm installers free of hardcoded package indexes.

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
- After CI YAML edits, parse `ci-pipelines/build-alps-extended-images.yaml` locally and use GitLab CI lint when credentials/network are available.
- For image-affecting changes, smoke-check `meta.sh` refs/hashes and `manual-build/manual-build.sh` script generation.

## Task-Specific Docs

- `README.md`: user-facing image matrix, runtime behavior, and CI overview.
- `manual-build/README.md`: local manual build workflow.
