# Apertus2 Megatron-LM

Three independent targets provide Apertus2 Megatron software for Alps.
The CUDA targets expect an external Megatron-LM checkout; the native ROCm
target includes a pinned Swiss AI checkout. Data and checkpoints remain external.

| Target | Containerfile | Base | Purpose |
| --- | --- | --- | --- |
| `megatron-base-cuda` | `Containerfile.megatron-base` | `pytorch-cuda:25.12-py3` | Minimal Chonk and Megachonk training stack |
| `megatron-cuda` | `Containerfile.megatron` | `pytorch-cuda:26.06-py3` | Full training, test, server and inference stack |
| `megatron-rocm` | `Containerfile.megatron.rocm` | `pytorch-rocm:rocm10.0-ubuntu24.04-py3.12-torch2.13` | Native MI300 training with ROCm TE, UCCL EP, KDA and Muon |

Target names select separate images, not image inheritance. The selected base
determines the accelerator family and build/test runner; the two CUDA targets
use GH200, while `megatron-rocm` uses Beverin MI300. Each target must end in its
base-derived family; a mismatch is rejected. `APP_VARIANTS` lists these flat
targets, with no separate logical-variant property. Profile keys uppercase target
names and replace hyphens with underscores; colliding normalized names are rejected.

Image names are `apertus-2-megatron-base-cuda`, `apertus-2-megatron-cuda`, and
`apertus-2-megatron-rocm`.
CI uses content-addressed tags. Stable dev tags use `alps7-dev`.

## Minimal image: megatron-base-cuda

The recipe retains the existing training stack.
Only the Containerfile name and test paths changed.
The base provides Python 3.12 and CUDA 13.1.

| Package | Version | Source | Commit |
| --- | --- | --- | --- |
| UCCL EP and `deep_ep` | source checkout | `swiss-ai/uccl` | `fa4c325c6113b798ae0a009ef8239e07352264fb` |
| Transformer Engine | `2.12.0` | `NVIDIA/TransformerEngine` | `5671fd3675906cda1ade26c24a65d3dedd88eb89` |
| DeepGEMM | `2.6.1` | `deepseek-ai/DeepGEMM` | `559d79fb6994a58b8a15b4b93bf13ccc16edf247` |
| nv_grouped_gemm | `1.1.4.post8` | `swiss-ai/grouped_gemm` | `45118e547923803ed52f80f240cf974620c9bdc5` |
| Emerging-Optimizers | `0.2.0` | `FFGGSSJJ/Emerging-Optimizers` | `cc1385eec1966aea2aebfb6dd6fc66a898312a55` |
| flash-linear-attention | `0.5.2` | `swiss-ai/flash-linear-attention` | `1820dba7e15fb927294779f23ac5097eb3927b25` |
| causal-conv1d | `1.7.0` | `Dao-AILab/causal-conv1d` | `cd81f0413cad2fc1e6f17e785ac39f59aae690cd` |
| nvidia-resiliency-ext | `0.6.0` | package index | — |

Native packages are built from pinned sources against the base image's Torch.
Torch, Torchvision, Triton and NumPy versions are kept as installation
constraints. Source pins and base constraints are stored in
`/opt/alps/apertus2-*.txt`.

The SwissAI 0.5.2 FLA fork adds per-channel `A_log` support for KDA decay gates.

UCCL is built for SM90, four NVLink peers per node and libfabric CXI.
The runtime environment must set `UCCL_EP_TRANSPORT=cxi`.

### Emerging-Optimizers fork

The pinned fork adds Muon's `"moonlight"` scaling mode:

```text
0.2 * sqrt(max(size_out, size_in))
```

Upstream supports the same scaling with `scale_mode="spectral"` and
`extra_scale_factor=0.2`, but does not accept `scale_mode="moonlight"` directly.
The image keeps the fork to preserve this configuration without changes.

### Python dependencies

| Package | Version |
| --- | --- |
| Transformers | `5.17.0` |
| Tokenizers | `0.23.2` |
| Hugging Face Hub | `1.31.0` |
| W&B | `0.30.0` |
| gRPC and gRPC tools | `1.84.0` |
| Google API common protos | `1.75.3` |
| Protobuf | `7.36.1` |
| OpenTelemetry API and SDK | `1.44.0` |
| OpenTelemetry HTTP exporter, common exporter and proto | `1.44.0` |
| OpenTelemetry semantic conventions | `0.65b0` |

All runtime additions are pinned in [`requirements.txt`](sources/requirements.txt).

## Full image: megatron-cuda

The base provides Python 3.12, CUDA 13.3 and the NGC Torch stack.
The installer keeps the inherited Torch, Torchvision, Triton and NumPy versions.
It builds native extensions against that stack.

### Native packages

| Package | Version | Configuration |
| --- | --- | --- |
| Transformer Engine | `2.17.0` | NCCL EP disabled |
| UCCL | `0.1.1+cu13` | Collective RDMA plugin, EP and P2P |
| DeepEP wrapper | `0.1.0` | Built from the UCCL checkout |
| DeepGEMM | `2.6.1` | Pinned source build |
| nv_grouped_gemm | `1.1.4.post8` | SwissAI fork |
| Emerging-Optimizers | `0.2.0` | Muon `moonlight` scaling |
| flash-linear-attention | `0.5.2` | SwissAI per-channel KDA gates |
| nvidia-resiliency-ext | `0.6.0` | Async checkpoint support |
| Mamba | `2.3.1` | Mamba and Mamba2; forced source build |
| causal-conv1d | `1.7.0` | Forced source build |
| fast-hadamard-transform | `1.1.0` | Pinned source build |
| FlashAttention 3 | `3.0.0` | Pinned Hopper source build |
| FlashInfer Python | `0.6.16.post3` | Matched to the JIT cache |
| FlashInfer JIT cache | `0.6.16.post3` | Built for this CUDA stack |

| Build group | Target | Default maximum jobs |
| --- | --- | --- |
| UCCL, TE and other native builds | Includes SM90 | `8` |
| FA3 | SM90 | `8`, with two NVCC threads |
| FlashInfer cache | SM90a | `16`, with one NVCC thread |

Mamba 2.3.1 avoids the TVM-FFI dependency conflict in 2.3.2.post1.
Mamba3 is not included. Mamba uses its upstream CUDA architecture list.

See [`versions.env`](sources/megatron/versions.env) for source repositories,
commit pins and versions.

### Python packages

The full Megatron test dependency group comes from
[`Megatron-LM-MoE` at `0789b8e429553c2f17c350afa8619c69d01bec25`](https://github.com/swiss-ai/Megatron-LM-MoE/blob/0789b8e429553c2f17c350afa8619c69d01bec25/pyproject.toml).

| Test dependency | Version or source |
| --- | --- |
| coverage | `7.16.2` |
| nltk | `3.10.3` |
| wrapt | `2.5.1` |
| pytest | `8.3.5` |
| pytest-mock | `3.16.0` |
| mock | `5.2.0` |
| pytest-cov | `7.1.0` |
| pytest-random-order | `1.2.0` |
| pytest-asyncio | `0.26.0` |
| PyGithub | `2.10.0` |
| Pydantic | `2.14.1` |
| TensorBoard | `2.20.0` |
| PyYAML | `6.0.3` |
| NeMo-Run | Git commit `01a9a8ba360f` |

| Additional package | Version | Use |
| --- | --- | --- |
| Transformers | `5.17.0` | Model interfaces |
| Tokenizers | `0.23.2` | Tokenization |
| Hugging Face Hub | `1.31.0` | Model and tokenizer storage |
| W&B | `0.30.0` | Experiment tracking |
| `openai[aiohttp]` | `2.7.2` | API client |
| httpx | `0.27.2` | HTTP client |
| FastAPI | `0.141.1` | API server |
| Uvicorn | `0.54.0` | ASGI server |
| Flask | `3.1.3` | HTTP server |
| Flask-RESTful | `0.3.10` | REST server |
| Quart | `0.22.0` | Asynchronous HTTP server |
| Hypercorn | `0.18.0` | ASGI server |
| aiohttp | `3.14.5` | Asynchronous HTTP client/server |
| multi-storage-client | `0.51.0` | Storage access |
| boto3 and botocore | `1.43.114` | S3 access |
| TensorStore | `0.1.85` | Tensor storage |
| CuPy CUDA 13.x | `14.2.0` | CUDA arrays |
| TorchMemorySaver | `0.0.10` | CUDA memory management |
| Griffe | `2.3.2` | API checks |
| msgpack | `1.2.3` | Serialization |
| pyzmq | `27.1.0` | Messaging |
| OpenTelemetry API and SDK | `1.43.0` | Telemetry |
| OpenTelemetry HTTP and gRPC exporters | `1.43.0` | Trace export |
| OpenTelemetry semantic conventions | `0.64b0` | Telemetry attributes |
| Protobuf | `6.33.6` | Serialization |
| Ninja | `1.13.2` | Native extension builds |
| CUDA Tile | `1.6.0` | FlashInfer dependency |
| NCCL4Py | `0.6.0` | FlashInfer dependency; uses base NCCL |
| cuDNN Frontend | `1.31.0` | FlashInfer dependency |
| NVIDIA ML Python | `13.615.71` | GPU inspection |
| gRPC tools | `1.81.1` | Resiliency extension dependency |
| NVIDIA DLFW Inspect | `0.2.2` | TE inspection support |

TensorRT-LLM is not installed by this recipe.

| File | Contents |
| --- | --- |
| [`requirements.in`](sources/megatron/requirements.in) | Direct dependencies and compatibility pins |
| [`requirements.lock`](sources/megatron/requirements.lock) | Resolved Python dependencies |
| [`versions.env`](sources/megatron/versions.env) | Native source pins and tool versions |

uv 0.8.22 installs the locked dependencies and native packages.
Newer uv releases reject the pinned NeMo-Run source metadata.
Pip bootstraps uv and builds the FlashInfer cache wheel.
Upstream build scripts can also use pip internally.

The locked install does not resolve extra packages. This prevents replacement
of the inherited NGC GPU libraries. Build-time checks verify declared
dependencies and their version constraints after installation.

### System tools

| Tool | Version or source | Use |
| --- | --- | --- |
| Go yq | `4.44.1` | Standalone ARM64 binary; SHA-256 checked |
| GNU envsubst | `gettext-base` | Environment substitution |
| uv | `0.8.22` | Python environment and package setup |
| curl | System package | Downloads |
| git | System package | Source and commit tracking |

### Build files and runtime

`Containerfile.megatron` copies `sources/megatron/` into `/opt/alps/megatron/`.
It runs `install-system.sh` once, then calls `install-kernels.sh` for each
native build group. Separate build steps allow layer reuse.

The image stores inherited package constraints in
`/opt/alps/megatron/base-pins.txt`. Native Git source pins are stored in
`/opt/alps/megatron/source-pins.txt`.

UCCL uses upstream commit `18b588a048a3a3fe3d4bd60aa5678c66e284df31`.
This includes [PR #1085](https://github.com/uccl-project/uccl/pull/1085), which keeps
the EP D2H handle table in device memory. This prevents `combine` faults caused
by managed-memory eviction under GPU memory pressure.
The build uses `BUILD_TYPE=all`.
It also installs the public P2P library and headers, and builds the DeepEP wrapper.

The runtime defaults are `UCCL_EP_TRANSPORT=cxi`, `UCCL_P2P_TRANSPORT=cxi`
and `UCCL_CXI_THREADING=safe`. Runtime settings can override these values.

The image preloads the CUDA 13 TorchMemorySaver library.
Set `LD_PRELOAD=` to disable this hook when it is not needed.
The TorchMemorySaver smoke test requires the default preload.

## Native ROCm image: megatron-rocm

This target is native ROCm training, **not feature parity with the CUDA Chonk
production configuration**. It builds against the canonical Alps ROCm 10.0 /
Torch 2.13 base on x86_64 and targets MI300 `gfx942`. Inherited Torch, TorchVision,
TorchAudio, Triton and NumPy distributions are constrained during installation.
Portable Python additions are resolved in `sources/rocm/requirements.lock` and
installed without dependency resolution. `requirements.in` lists direct inputs;
`requirements.overrides` excludes PyPI Torch/Triton and matches the inherited
NumPy ABI during lock generation. Regenerate from the repository root with:

```bash
APP=Alps-Images/apps/apertus-2/sources/rocm
uv pip compile "$APP/requirements.in" --overrides "$APP/requirements.overrides" \
    --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
    --no-emit-package torch --no-emit-package triton --no-emit-package numpy \
    --no-annotate --no-header --output-file "$APP/requirements.lock"
```


Source pins live in [`sources/rocm/versions.env`](sources/rocm/versions.env).
The stack includes:

- ROCm Transformer Engine at `b56021a0e4a2e4a7822482ffdf5353368a777e0e`,
  with CK fused attention and runtime CK JIT; AOTriton is not compiled.
  Hipified GPU `.cpp` and CK attention translation units are explicitly declared
  HIP; ordinary host-only C++ sources retain their language.
  The build uses the base's canonical RCCL headers and resolves the packaged
  ROCTX ABI-4 runtime even when its unversioned development link is absent.
  The app-local CK-JIT patch keeps AITER's interceptor discovery environment,
  but removes `ROCM_PATH`/`HIP_PATH` from real compiler/linker subprocesses at
  build time and during runtime CK JIT: those overrides break clang 23
  device-math discovery with the wheel SDK.
  The base SDK compatibility layout provides development links in the physical
  wheel library directories as well as `/opt/rocm`, so HIP compiler detection
  and shared-library linking work without overriding the compiler's SDK root.
  MIOpen's public development headers come from the base, matching its bundled
  runtime; hipified ATen MIOpen includes need no app-local workaround.
- UCCL EP and its real DeepEP wrapper at
  `18b588a048a3a3fe3d4bd60aa5678c66e284df31`. This retains the D2H handle-table
  memory-pressure fix used by the full CUDA target. App-local patches retain
  physical-GPU CXI selection, serialized barrier cleanup/provider initialization,
  ROCR GPU-memory registration, HIP launch and ROCm 10 compiler compatibility.
  Proxies borrow the wrapper's RDMA allocation; its DLPack/Torch owner frees it.
  Communication uses a PyTorch pool stream so recorded tensors remain safe to
  release after EP buffer destruction.
  The older runtime-top-k patch is already upstream and is not applied twice.
- The same Swiss AI FLA per-channel KDA gates and Emerging-Optimizers Muon
  `moonlight` scaling as the CUDA targets, plus a native causal-conv1d build.
- Swiss AI Megatron-LM-MoE at
  `0789b8e429553c2f17c350afa8619c69d01bec25`, installed from
  `/opt/megatron-lm`, with its CPU dataset helper prebuilt.

`/opt/alps/apertus2-rocm/` records base constraints, native source revisions and
the installed package inventory. Compiler tools and headers remain installed
for runtime JIT. UCCL handles expert communication; ordinary collectives retain
the base image's Alps aws-ofi-rccl transport. No UCCL collective plugin is selected.

**Excluded:** Swiss AI's CUDA/CUTLASS `nv_grouped_gemm` expert-offloading APIs,
DeepGEMM FP8 expert offloading, Hopper FA3, FlashInfer, CuPy CUDA wheels,
TorchMemorySaver's CUDA preload, and NVIDIA resiliency extensions. Native
training should use TE experts rather than the custom offloading experts.
Do not run `example/chonk.sh` unchanged with this image: it enables those
CUDA-only offloading paths and is a 256-node production launcher.

The ROCm target declares three image-local CI checks:

1. `check-stack.py`: GPU-free metadata/ELF checks during build; on GPU, numerical
   TE Linear/GroupedLinear forward/backward, KDA gate and recurrent-kernel
   gradients, causal convolution, and stateful Muon updates.
2. `train-smoke.sh`: single-GPU TE/CK BF16 GPT training with synthetic data,
   checkpoint at iteration 2, then a fresh process resuming through iteration 4.
   It checks finite loss and optimizer/scheduler/RNG checkpoint state.
3. `uccl-smoke.py`: two-node/eight-rank RCCL all-reduce and BF16
   **high-throughput** UCCL dispatch/combine with numerical reference values.
   It also destroys the EP buffer, releases recorded tensor storage, runs fresh
   GPU/RCCL work, and requires clean process exit.
   It needs exclusive nodes, block rank order, and all four GPUs visible to
   each rank (`--gpu-bind=none`). Same-user concurrent UCCL jobs cannot share
   nodes because the upstream POSIX-shm barriers are UID-scoped.

Run inside the native image:

```bash
python3 /opt/tests/apertus-2/rocm/check-stack.py
bash /opt/tests/apertus-2/rocm/train-smoke.sh
```

The distributed check runs directly under Slurm, not via Ray or torchrun.
Its rendezvous address and port are derived from the Slurm allocation.
CSCS EDF launches use `srun --environment` with the saved SquashFS as `image`,
`/ritom/` mounted, `writable = true`, `entrypoint = true`, and
`PMIX_MCA_psec = "native"`. Set `com.hooks.cxi.enabled = "false"` under
`[annotations]` to retain the image's own CXI stack.

Manual container launches must expose `/dev/kfd`, `/dev/dri` and the allocated
`/dev/cxi*` devices, and forward the allocation's `SLINGSHOT_*`/`FI_CXI_*`
environment. The base's RCCL transport requires CXI initialization even for
the single-rank training smoke.

These checks do not establish full-size hybrid-model convergence or scaling.

## Build

From the repository root, select `megatron-base-cuda`, `megatron-cuda`, or `megatron-rocm`:

```bash
TARGET=megatron-cuda
BUILD_DIR="$SCRATCH/tmp/apertus2-image-build"

mkdir -p "$BUILD_DIR"

manual-build/manual-build.sh \
    app apertus-2 "$TARGET" "$BUILD_DIR/build-$TARGET.sh"

bash "$BUILD_DIR/build-$TARGET.sh"
```

Build the CUDA targets on an aarch64 GH200 compute node; build `megatron-rocm`
on an x86_64 node with the ROCm SDK supplied by its base image. Native extensions
use explicit architecture targets, so builds need no visible GPU. The generated
script selects the canonical, content-addressed base image, which must already
exist locally or in the selected registry. Keep `IMAGE_PREFIX` and clone-URL
inputs consistent when generating base and app scripts.
See the [manual build guide](../../../manual-build/README.md).

## Run Chonk pretraining

[`example/chonk.sh`](example/chonk.sh) is based on the
[Chonk launcher in `swiss-ai/pretrain`](https://github.com/swiss-ai/pretrain/blob/6cb927e8b153e2fe9cd660fe0a6332c2ee7ce2a2/apertus-2/pretraining/chonk/chonk.sh).
It uses [`environments/chonk.toml`](environments/chonk.toml), which selects
`megatron-base-cuda` by its content-addressed tag. The Megachonk and LC-extension
EDFs select the same minimal target; `environments/megatron.toml` selects the full
`megatron-cuda` target. Set `CONTAINER_ENV` to use a different EDF.

The CUDA EDF pins use the current canonical refs with the upstream repository
clone URL. Build and publish those refs before launching; renaming targets does
not retag existing images. Regenerate pins when their build inputs change.

**This is a production configuration:** training requests 256 nodes and 1,024
GPUs. The warm-up requests 32 nodes. It is not a small-scale test.

## Tests

The two CUDA targets declare separate CI tests and validation markers:

- vetnode hardware and software checks;
- package versions and GPU imports;
- UCCL dispatch/combine on two GH200 nodes with eight tasks.

Both CUDA images check TE, Muon scaling and per-channel KDA gate gradients.
The full image adds Hadamard, FA3, FlashInfer, Mamba, CuPy and
TorchMemorySaver GPU checks. Its FlashInfer check requires the packaged cache
module to be loaded, not a JIT fallback.

Tests are grouped under `tests/megatron-base/` and `tests/megatron/`.
Each image contains its own group under `/opt/tests/apertus-2/`.

Build-time metadata checks do not require a GPU:

```bash
python3 /opt/tests/apertus-2/megatron/check-stack.py --metadata-only
```

Run the full GPU checks inside the built `megatron-cuda` image on a GH200 node:

```bash
python3 /opt/tests/apertus-2/megatron/check-stack.py
```

These tests do not cover training convergence, every kernel or an end-to-end
Megatron training run. Python dependency resolution alone does not establish
CUDA or binary compatibility.
