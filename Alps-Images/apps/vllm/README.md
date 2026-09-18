# vLLM App Image

This image currently builds vLLM `v0.27.1` from source on top of `pytorch-cuda:26.02-py3` (CUDA variant) and `pytorch-rocm:rocm7.14-ubuntu24.04-py3.12-torch2.11` (ROCm variant).

The app-local patches in `patches/` are required for NVIDIA PyTorch `26.02` and `26.03`, whose Torch `2.11.0a0` snapshots do not expose all APIs assumed by vLLM `v0.27.1` after the stable-libtorch migration. In particular, `torch::stable::Tensor::layout()`, the stable `from_blob` deleter overload, and `torch._opaque_base` are missing or incompatible.

Revisit these patches when vLLM moves to Torch `2.12` as its baseline. At that point we may be able to move to a newer NVIDIA PyTorch container and drop some or all compatibility patches.

Older vLLM releases may need fewer changes. Releases `0.22` and older are suspected to work without these compatibility patches, but that still needs to be verified against the Alps CUDA/HPC stack.

## Mooncake (CXI KV transfer)

Both variants additionally install the [Mooncake](https://github.com/kvcache-ai/Mooncake) transfer engine with the HPE Slingshot (CXI) backend (`USE_CXI`, [kvcache-ai/Mooncake#2535](https://github.com/kvcache-ai/Mooncake/pull/2535)). It is built from a pinned tag by `sources/install-mooncake.sh` (which is part of the app content hash) and installed as the `mooncake-transfer-engine` wheel, providing the `mooncake.engine` and `mooncake.store` modules used by vLLM's `MooncakeConnector` and `MooncakeStoreConnector` KV connectors.

Mooncake is compiled against the libfabric installed by the Alps base image (`/usr`), so it always matches the stack's libfabric version and shares a single in-process libfabric instance with the aws-ofi-nccl NCCL plugin. This is deliberate: a second, bundled libfabric would race for the CXI devices and leave one consumer with an empty provider list.

On Slingshot, select the CXI transport in the extra config instead of the default `rdma` protocol, e.g.:

```bash
vllm serve <model> \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeConnector", "kv_role": "kv_producer",
      "kv_connector_extra_config": {"mooncake_protocol": "cxi"}}'
```

`device_name` may be left empty; Mooncake auto-discovers CXI devices. The image also ships the `mooncake_master` store service and the `transfer_engine_bench` utility for transport debugging. A single-node CXI smoke test runs in CI (`mooncake_smoke.sh`).
