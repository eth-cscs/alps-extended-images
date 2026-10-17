# Apertus2 Megatron-LM

Two images provide Megatron-LM software for Alps GH200 nodes.
Megatron-LM, pretrain, data and checkpoints remain in external directories.

| Target | Containerfile | Base | Purpose |
| --- | --- | --- | --- |
| `megatron-base` | `Containerfile.megatron-base` | `pytorch-cuda:25.12-py3` | Minimal Chonk and Megachonk training stack |
| `megatron` | `Containerfile.megatron` | `pytorch-cuda:26.06-py3` | Full training, test, server and inference stack |

The target names select separate images. Both use the CUDA build runners.
The full image does not inherit from `megatron-base`.

Image names are `apertus-2-megatron-base` and `apertus-2-megatron`.
CI uses content-addressed tags. Stable dev tags use `alps7-dev`.

## Minimal image: megatron-base

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

## Full image: megatron

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

UCCL uses upstream commit `3ec055f3f9a6d1cdd48901abd4d430abc3530445`,
the repository HEAD checked on 2026-10-16. The build uses `BUILD_TYPE=all`.
It also installs the public P2P library and headers, and builds the DeepEP wrapper.

The runtime defaults are `UCCL_EP_TRANSPORT=cxi`, `UCCL_P2P_TRANSPORT=cxi`
and `UCCL_CXI_THREADING=safe`. Runtime settings can override these values.

The image preloads the CUDA 13 TorchMemorySaver library.
Set `LD_PRELOAD=` to disable this hook when it is not needed.
The TorchMemorySaver smoke test requires the default preload.

## Build

From the repository root, select `megatron-base` or `megatron`:

```bash
TARGET=megatron
BUILD_DIR="$SCRATCH/tmp/apertus2-image-build"

mkdir -p "$BUILD_DIR"

manual-build/manual-build.sh \
    app apertus-2 "$TARGET" "$BUILD_DIR/build-$TARGET.sh"

bash "$BUILD_DIR/build-$TARGET.sh"
```

Run Podman on an aarch64 GH200 compute node. The build does not need a visible
GPU. The generated script selects the canonical, content-addressed base image.
See the [manual build guide](../../../manual-build/README.md).

## Run Chonk pretraining

[`example/chonk.sh`](example/chonk.sh) is based on the
[Chonk launcher in `swiss-ai/pretrain`](https://github.com/swiss-ai/pretrain/blob/6cb927e8b153e2fe9cd660fe0a6332c2ee7ce2a2/apertus-2/pretraining/chonk/chonk.sh).
It uses [`environments/chonk.toml`](environments/chonk.toml), which selects the
earlier training image by its content-addressed tag. The supplied EDFs do not
select the full `megatron` image. Set `CONTAINER_ENV` to use an EDF for a different
image.

**This is a production configuration:** training requests 256 nodes and 1,024
GPUs. The warm-up requests 32 nodes. It is not a small-scale test.

## Tests

Both targets declare separate CI tests and validation markers:

- vetnode hardware and software checks;
- package versions and GPU imports;
- UCCL dispatch/combine on two GH200 nodes with eight tasks.

Both images check TE, Muon scaling and per-channel KDA gate gradients.
The full image adds Hadamard, FA3, FlashInfer, Mamba, CuPy and
TorchMemorySaver GPU checks. Its FlashInfer check requires the packaged cache
module to be loaded, not a JIT fallback.

Tests are grouped under `tests/megatron-base/` and `tests/megatron/`.
Each image contains its own group under `/opt/tests/apertus-2/`.

Build-time metadata checks do not require a GPU:

```bash
python3 /opt/tests/apertus-2/megatron/check-stack.py --metadata-only
```

Run the full GPU checks inside the built `megatron` image on a GH200 node:

```bash
python3 /opt/tests/apertus-2/megatron/check-stack.py
```

These tests do not cover training convergence, every kernel or an end-to-end
Megatron training run. Python dependency resolution alone does not establish
CUDA or binary compatibility.
