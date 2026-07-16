# AGENTS.md

## Purpose

This repo builds, tests, and publishes Alps-optimized container images. Preserve reproducible content hashing, canonical base usage for apps, CI retagging for tests, multi-node validation, and strict stable promotion.

## Layout

- `Alps-Images/NGC/` is the current CUDA/NGC base-image family; do not treat its assumptions as universal.
- `Alps-Images/common/` holds shared Alps CUDA/HPC installer and runtime setup used by NGC bases.
- `Alps-Images/apps/<app>/` holds app Containerfiles, `profile.env`, optional app-local patches, and optional tests copied into `/opt/tests/<app>/`.
- `Alps-Images/patches/` contains upstream patches consumed by base builds and included in base hashes; `Alps-Images/apps/<app>/patches/` contains app-specific source patches included in app hashes.
- `ci-pipelines/build-alps-extended-images.yaml` is the executable pipeline source; prefer it over README prose when they disagree.
- `ci-pipelines/helpers/meta.sh` derives image refs and hashes; `ci-pipelines/helpers/skopeo.sh` owns registry copy/promotion behavior.
- `Alps-Images/common/apt-helpers.sh` owns apt wrappers, rootless sandbox fallback, and build-dependency cleanup helpers shared by base and app builds.
- `manual-build/manual-build.sh` emits local `podman build` scripts using the same ref/hash logic as CI; keep `manual-build/README.md` aligned with it.

## Image Refs

- Each build has `CANON_IMAGE_REF` with a 16-char content hash, `TEST_IMAGE_REF` using `${ALPS_REV}-${CI_COMMIT_SHORT_SHA}`, and `STABLE_IMAGE_REF` for release tags.
- GitLab resolves job `image:` before dotenv artifacts exist, so tests must use predictable commit-SHA test tags created by `tag-*-for-ci` jobs.
- Application images must build from the canonical base ref returned by `base_refs`, never from stable or commit-SHA tags.
- App hashes must include the canonical base ref so unchanged app plus changed base rebuilds the app.

## Hashing Rules

- Base hashes must cover the base Containerfile, variant directory/profile/hooks, `Alps-Images/common`, `Alps-Images/patches`, and logical build inputs used by `meta.sh`.
- App hashes must cover the app Containerfile, `profile.env`, app-local `patches/` when present, copied tests when present, copied shared helper inputs when present, and the canonical base ref.
- Do not add timestamps or commit SHAs to content hashes; they defeat reuse. OCI labels may still receive CI metadata.
- Required hash inputs should fail when missing. Do not silently skip a declared Containerfile/profile/patch directory.
- Any helper output field that can contain whitespace must be encoded or moved to dotenv; current `REMOVE_HPCX_DIRS` is base64 encoded for this reason.

## CI Rules

- Pipeline stages are `build-base -> test-base -> build-apps -> test-apps -> publish`.
- Build-stage dependency chain is `meta -> build -> retag-for-CI`; explicit `needs` can bypass stage barriers, so include every correctness dependency.
- Matrix jobs must use matching `needs:parallel:matrix` mappings for every identity field (`NGC_NAME`/`NGC_TAG` or `NAME`) to avoid mixed dotenv artifacts.
- Keep matrix values short because GitLab includes them in job names.
- Current NGC base matrix in CI includes `pytorch` tags `26.06-py3`, `26.02-py3`, `26.01-py3`, and `25.12-py3`, plus `nemo` tags `26.02` and `25.11.01`, and `physicsnemo` tag `25.11`.
- Current app matrix in CI is `apertus-1p5`, `apertus-2`, `sfttrainer`, `verl`, and `vllm`.
- When adding or renaming tests, update `publish-gate.needs`; publishing depends on this gate, not just stage order.

## Promotion

- Use `skopeo.sh` helpers instead of open-coded registry logic.
- `copy_if_needed` is for ephemeral CI test tags.
- Non-dev stable tags are immutable: `promote_strict` must fail if stable exists with a different digest.
- `*-dev` revisions intentionally overwrite stable dev tags only on the default branch; non-default branches are dry-run only.
- Promote the same canonical digest to JFrog and GHCR; derive GHCR refs from the stable JFrog ref.

## Containerfiles

- Preserve `# syntax=docker/dockerfile:1`, `ARG BASE_IMAGE` followed by `FROM ${BASE_IMAGE}`, and OCI/CSCS labels in image Containerfiles.
- Use `RUN set -eux; ...` for build steps and clean apt lists, temporary clones, build trees, and caches in the same layer.
- Use `apt_cmd`, `apt_get`, and apt cleanup helpers from `Alps-Images/common/apt-helpers.sh` instead of raw apt commands in shared installer and app build steps.
- Prefer pinned tags/commits for downloaded sources; avoid unbounded `pip -U` unless the existing image already requires it.
- `install-alps-hpc-stack.sh` programmatically purges preinstalled generic network stacks before rebuilding the Alps stack; keep `REMOVE_HPCX_DIRS` only as an optional profile escape hatch for image-specific leftovers.
- NGC variant `profile.env` is declarative and may carry optional `REMOVE_HPCX_DIRS` or `NVCR_PREFIX`; use `hooks.d/*.sh` only for deterministic late image fixes.
- If tests must run without the repo checkout, copy them into `/opt/tests/<image>/` and include them in the app hash.
- Prefer patch files under `Alps-Images/apps/<app>/patches/` over inline source rewrites in app Containerfiles, so compatibility changes are reviewable and hash-tracked.

## Runtime Gotchas

- `alps-runtime.env` uses `defvar`, which sets defaults only when variables are unset; preserve user overrides, including intentionally empty values.
- `source-alps-env.sh` must stay idempotent and Bash-safe because it is linked into NVIDIA entrypoint hooks and `/etc/profile.d`.
- Runtime warning hooks run through non-interactive Bash `BASH_ENV` and Python `sitecustomize.py`; they must never break shell or interpreter startup.
- Do not add `set -e`, `set -u`, or `pipefail` to scripts sourced through `BASH_ENV`.
- Keep shared warning text in `alps-runtime-warning.txt` so Bash and Python diagnostics remain aligned.

## Adding Families

- Add sibling family directories and family-specific metadata functions rather than spreading NGC conditionals through the pipeline.
- Do not generalize NGC/CUDA assumptions such as `NVCR_PREFIX`, `REMOVE_HPCX_DIRS`, SM90 flags, NCCL/NVSHMEM/aws-ofi-nccl, NVIDIA hooks, GH200 runners, or `parse_ngc_base_image`.
- New families need their own runner/hardware assumptions, tests, hash inputs, build args, retag jobs, publish jobs, and `publish-gate.needs` entries.

## Verification

- After shell edits, run `git ls-files '*.sh' | xargs -r -n1 bash -n` from the repo root.
- If ShellCheck is available, run it on changed shell scripts; no repo-local ShellCheck config is present.
- After CI YAML edits, validate `ci-pipelines/build-alps-extended-images.yaml` with GitLab CI lint when credentials/network are available.
- Before submitting image-affecting changes, confirm hashes include all changed inputs, app bases are canonical, test `image:` refs use commit-SHA tags, and README tables/stage descriptions are updated when user-facing variants or versions change.
