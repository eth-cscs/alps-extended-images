# Manual Builds On Alps

Images are normally built by GitLab CI, but they can also be built manually from a repository checkout on Alps. Run these commands on a node where rootless `podman` is available and where the target architecture matches the image you want to build.

The helper uses the same reference and hash logic as CI. By default it writes local tags under `localhost/alps-images`; override `IMAGE_PREFIX` if you want registry-style tags.

## Generate A Build Script

Generate a base-image build script:

```bash
manual-build/manual-build.sh base pytorch 26.06-py3 /tmp/build-pytorch-26.06.sh
```

Run it:

```bash
bash /tmp/build-pytorch-26.06.sh
```

Generate an app-image build script:

```bash
manual-build/manual-build.sh app vllm /tmp/build-vllm.sh
```

Run it:

```bash
bash /tmp/build-vllm.sh
```

For app images, the generated script uses the canonical base image reference computed from `profile.env`. If you built that base locally first, keep the default `IMAGE_PREFIX` consistent so the app build can find the local base tag.

## Useful Overrides

Set `IMAGE_PREFIX` to choose the output namespace:

```bash
IMAGE_PREFIX=localhost/my-alps-images manual-build/manual-build.sh base pytorch 26.06-py3 /tmp/build-base.sh
```

Set `ALPS_REV` to use a different revision suffix:

```bash
ALPS_REV=alps7-dev manual-build/manual-build.sh app vllm /tmp/build-vllm.sh
```

Set `CSCS_CI_ORIG_CLONE_URL`, `CI_COMMIT_SHA`, or `CI_COMMIT_SHORT_SHA` if you need labels and hashes to match a specific CI context.

## Direct Podman Build

The generated scripts are plain shell scripts. Inspect them if you need to customize `podman build` arguments, for example to add `--no-cache`, use a different storage location, or tag an additional image name.

For base images, the important build arguments are `BASE_IMAGE`, `NGC_VARIANT_DIR`, and `REMOVE_HPCX_DIRS`. For app images, the important build argument is `BASE_IMAGE`, which should be the canonical base image reference from `ci-pipelines/helpers/meta.sh` rather than a stable or commit-SHA tag.
