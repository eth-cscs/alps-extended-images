# CI Generator Image

This image provides the runtime for `ci-pipelines/helpers/generate-child-pipeline.py`:

- Python 3 from the `quay.io/skopeo/stable:v1.24.0` base distribution
- PyYAML from Debian (`python3-yaml`)
- `skopeo` copied from `quay.io/skopeo/stable:v1.24.0`
- Bash and CA certificates

It is pipeline bootstrap infrastructure and is built by the separate manual pipeline:

```text
ci-pipelines/build-ci-generator-image.yaml
```

The main image pipeline pins this image via `CI_GENERATOR_IMAGE` in `ci-pipelines/build-alps-extended-images.yaml`.

When changing `Containerfile`, update the version/tag in both pipeline files and run the manual generator-image pipeline on GH200/ARM.
