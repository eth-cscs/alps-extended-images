# `nemo-rl` container image

[NeMo-RL](https://github.com/NVIDIA-NeMo/RL) with a Megatron policy and the Megatron
generation backend (no vLLM, so no weight refit), plus the MoE kernel stack the
Apertus2 MoE checkpoints need.

Everything NeMo-RL imports at run time is installed in the image, so jobs need no
`PYTHONPATH` overlays. Validated 2026-09-14 with a two-node GRPO run through NeMo
Gym on 8x GH200.

## Building locally

The build context is the repository root, not this directory: the
`COPY Alps-Images/...` lines only resolve from there.

```bash
podman build -f Alps-Images/apps/nemo-rl/Containerfile \
  --build-arg BASE_IMAGE=jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/pytorch-cuda:25.12-py3-alps7-dev \
  -t nemo-rl:local .
```

## What the image contains

| Component | Pin | Why |
|---|---|---|
| UCCL-EP | `swiss-ai/uccl@fa4c325c` | `uccl.ep` + the `deep_ep` wrapper Megatron's flex dispatcher imports, so jobs skip the per-job srun build |
| TransformerEngine | `v2.17` | CUDA graph support; this is megachonk's TE, one minor above the `te212` sibling image |
| DeepGEMM | `FFGGSSJJ/DeepGEMM@559d79fb` | FP8 grouped GEMM |
| grouped_gemm | `FFGGSSJJ/grouped_gemm@45118e54` | MoE GEMM with gradient-accumulation fusion |
| nvidia-resiliency-ext | `0.6.0` | the version Megatron-LM pins; older ones break async checkpoint save |
| Emerging-Optimizers | `FFGGSSJJ@cc1385ee` | decoupled Muon (`md_decoupling`) |
| flash-linear-attention | `v0.5.2` | KDA kernels |
| ray | `2.56.1` | NeMo-RL worker runtime; not in the base image |
| flashinfer-python | `0.6.18.post1` | **required**, not optional: NeMo-RL hardcodes `sampling_backend="flashinfer"` in its Megatron worker, so `InferenceConfig.__post_init__` raises `ImportError` without it |
| openai | `2.7.2` | nemo-gym requires `<=2.7.2` and pins each child server venv to the *parent* version, so a newer parent makes every Gym venv unresolvable |
| transformers / megatron-energon / hydra-core / math-verify / mlflow / tensordict / swanlab | pinned | NeMo-RL import-time dependencies |
| uv | `0.11.23` | installs everything in this image, and is what NeMo Gym shells out to at run time to build its per-server venvs |

Installs go through `uv pip install --system`, not pip: it resolves the whole
dependency set at once rather than package by package. Versions that affect the
validated kernel and NeMo-RL stack are pinned inline. The image intentionally
does not use `--exclude-newer`: JFrog metadata for some required build packages
does not include upload dates, causing uv to exclude those packages entirely.
uv is bootstrapped with the repository's `pip_install` helper because nothing
else exists at that point, and `UV_DEFAULT_INDEX` points at the same CSCS JFrog
mirror so Gym venvs built at run time resolve through it too.

Every install is additionally run against `/opt/alps/base-pins.txt`, a constraints
file generated from the selected base image's own
`torch`/`torchvision`/`triton`/`numpy` versions. A transitive dependency therefore
cannot drag in a PyPI torch wheel and shadow that NGC stack, and a final build
check verifies that those versions remained unchanged during installation.

## What the image does not contain

NeMo-RL, Megatron-Bridge and Megatron-LM are not vendored: bind-mount the checkouts
and put them on `PYTHONPATH`. They are actively developed forks, and baking them in
would force an image rebuild for every source change.

NeMo Gym's per-server venvs are built at run time too, since they derive from the
Gym checkout's own `requirements.txt`.
