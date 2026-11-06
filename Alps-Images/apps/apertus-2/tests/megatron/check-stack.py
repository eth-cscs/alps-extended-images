"""Check the full Megatron image and selected GH200 kernels."""

import argparse
import importlib
import json
import os
import subprocess
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path("/opt/alps/megatron")

NATIVE_VERSIONS = {
    "transformer-engine": "2.17.0",
    "deep-gemm": "2.6.1",
    "nv-grouped-gemm": "1.1.4.post8",
    "emerging-optimizers": "0.2.0",
    "flash-linear-attention": "0.5.2",
    "causal-conv1d": "1.7.0",
    "fast-hadamard-transform": "1.1.0",
    "flash-attn-3": "3.0.0",
    "nvidia-resiliency-ext": "0.6.0",
    "mamba-ssm": "2.3.1",
    "uccl": "0.1.1",
    "flashinfer-python": "0.6.16.post3",
    "flashinfer-jit-cache": "0.6.16.post3",
}

MODULES = (
    "nltk",
    "coverage",
    "pytest_mock",
    "pytest_asyncio",
    "nemo_run",
    "openai",
    "fastapi",
    "uvicorn",
    "flask",
    "flask_restful",
    "quart",
    "hypercorn",
    "aiohttp",
    "httpx",
    "multistorageclient",
    "griffe",
    "boto3",
    "botocore",
    "msgpack",
    "zmq",
    "opentelemetry.sdk.trace",
    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
    "opentelemetry.exporter.otlp.proto.http.trace_exporter",
    "uccl.ep",
    "uccl.p2p",
    "deep_ep",
    "deep_gemm",
    "grouped_gemm",
    "grouped_gemm_backend",
    "emerging_optimizers",
    "fla",
    "nvidia_resiliency_ext",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "wandb",
    "torch_memory_saver",
    "mamba_ssm",
    "causal_conv1d",
    "flash_attn_3._C",
)


def check_dependencies(packages, base_pins):
    for name in packages:
        for dependency in metadata.requires(name) or []:
            requirement = Requirement(dependency)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue

            distribution = requirement.name
            if distribution == "triton" and "triton" not in base_pins:
                distribution = "pytorch-triton"

            version = metadata.version(distribution)
            assert requirement.specifier.contains(version, prereleases=True), (
                f"{name} requires {requirement}; found {distribution}=={version}"
            )


def check_metadata():
    base_pins = dict(
        line.split("==", 1)
        for line in (ROOT / "base-pins.txt").read_text().splitlines()
    )
    for name, expected in base_pins.items():
        assert metadata.version(name) == expected, f"Base package changed: {name}"

    packages = set(NATIVE_VERSIONS) | {"deep-ep"}
    for line in (ROOT / "requirements.lock").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        packages.add(requirement.name)
        version = metadata.version(requirement.name)
        assert requirement.specifier.contains(version, prereleases=True), line

    for name, expected in NATIVE_VERSIONS.items():
        version = metadata.version(name)
        print(f"{name}={version}", flush=True)
        assert version.split("+")[0] == expected, name

    check_dependencies(sorted(packages), base_pins)

    uccl = metadata.distribution("uccl")
    assert Path(str(uccl.locate_file("uccl/lib/libnccl-net-uccl.so"))).is_file()
    assert Path("/usr/local/lib/libuccl_p2p.so").is_file()
    assert Path("/usr/local/include/uccl_engine.h").is_file()
    metadata.distribution("deep-ep")
    assert metadata.version("pytest") == "8.3.5"
    assert metadata.version("opentelemetry-api") == "1.43.0"
    assert metadata.version("opentelemetry-sdk") == "1.43.0"
    assert metadata.version("opentelemetry-semantic-conventions") == "0.64b0"

    nemo_run = metadata.distribution("nemo-run").read_text("direct_url.json")
    assert nemo_run is not None
    source = json.loads(nemo_run)
    assert source["vcs_info"]["commit_id"] == "01a9a8ba360f7b2908728ad0516e0ad9d936966d"

    preload = Path(
        "/usr/local/lib/python3.12/dist-packages/"
        "torch_memory_saver_hook_mode_preload_cu13.abi3.so"
    )
    assert preload.is_file()
    assert os.environ["NVIDIA_PYTORCH_VERSION"] == "26.06"


def check_tools():
    expected = {
        "yq": "v4.44.1",
        "envsubst": "GNU gettext-runtime",
        "uv": "uv 0.8.22",
        "curl": "curl ",
        "git": "git version ",
    }
    for command, version in expected.items():
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        assert version in result.stdout, result.stdout
        print(result.stdout.splitlines()[0], flush=True)


def set_test_caches():
    job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    rank = os.environ.get("SLURM_PROCID", "0")
    root = Path(f"/tmp/apertus2-megatron-{job}/rank-{rank}")
    for variable, name in {
        "XDG_CACHE_HOME": "xdg",
        "TORCH_HOME": "torch",
        "TORCH_EXTENSIONS_DIR": "torch-extensions",
        "CUDA_CACHE_PATH": "cuda",
        "CUPY_CACHE_DIR": "cupy",
        "DG_CACHE_DIR": "deep-gemm",
        "HF_HOME": "huggingface",
        "TRITON_CACHE_DIR": "triton",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer",
        "WANDB_CACHE_DIR": "wandb",
        "WANDB_CONFIG_DIR": "wandb-config",
    }.items():
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)


def check_gpu():
    set_test_caches()

    import cupy
    import torch
    import transformer_engine.pytorch as te
    from emerging_optimizers.orthogonalized_optimizers.muon import get_muon_scale_factor
    from fast_hadamard_transform import hadamard_transform
    from fla.ops.kda.gate import fused_kda_gate
    from flash_attn_3.flash_attn_interface import flash_attn_func
    from flashinfer import silu_and_mul
    from flashinfer_jit_cache import get_jit_cache_dir
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from torch_memory_saver import torch_memory_saver

    assert torch.cuda.is_available()
    torch.cuda.set_device(int(os.environ.get("SLURM_LOCALID", "0")))
    assert torch.cuda.get_device_capability() == (9, 0)
    assert torch.version.cuda is not None and torch.version.cuda.startswith("13.3")

    for name in MODULES:
        module = importlib.import_module(name)
        print(f"{name}: {module.__file__}", flush=True)

    layer = te.Linear(64, 64, params_dtype=torch.bfloat16, device="cuda")
    x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    layer(x).float().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert get_muon_scale_factor(64, 64, mode="moonlight") == 1.6

    g = torch.randn(1, 8, 2, 64, device="cuda", requires_grad=True)
    a_log = torch.randn(2, 64, device="cuda", requires_grad=True)
    bias = torch.zeros(128, device="cuda", requires_grad=True)
    actual = fused_kda_gate(g, a_log, bias)
    expected = -a_log.exp() * torch.nn.functional.softplus(g + bias.view(2, 64))
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    actual_grads = torch.autograd.grad(actual.sum(), (g, a_log, bias))
    expected_grads = torch.autograd.grad(expected.sum(), (g, a_log, bias))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-3, rtol=1e-3)

    x = torch.randn(4, 128, device="cuda", dtype=torch.float32)
    restored = hadamard_transform(hadamard_transform(x)) / x.shape[-1]
    torch.testing.assert_close(restored, x, atol=1e-4, rtol=1e-4)

    q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
    actual = flash_attn_func(q, q, q, causal=True)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        q.transpose(1, 2),
        q.transpose(1, 2),
        is_causal=True,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    cache_module = Path(get_jit_cache_dir()) / "silu_and_mul/silu_and_mul.so"
    assert cache_module.is_file(), cache_module

    x = torch.randn(4, 16, device="cuda", dtype=torch.float16)
    actual = silu_and_mul(x)
    expected = torch.nn.functional.silu(x[:, :8]) * x[:, 8:]
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    assert str(cache_module) in Path("/proc/self/maps").read_text()

    u = torch.randn(1, 4, 16, device="cuda", requires_grad=True)
    delta = torch.randn_like(u)
    a = -torch.ones(4, 8, device="cuda")
    b = torch.randn(1, 8, 16, device="cuda")
    c = torch.randn_like(b)
    output = selective_scan_fn(u, delta, a, b, c, delta_softplus=True)
    output.sum().backward()
    assert torch.isfinite(output).all()
    assert u.grad is not None and torch.isfinite(u.grad).all()

    array = cupy.arange(16, dtype=cupy.float32).reshape(4, 4)
    assert cupy.allclose(array @ cupy.eye(4, dtype=cupy.float32), array)

    with torch_memory_saver.region(tag="megatron-smoke", enable_cpu_backup=True):
        saved = torch.ones(16, device="cuda")
    torch.cuda.synchronize()
    torch_memory_saver.pause(tag="megatron-smoke")
    torch_memory_saver.resume(tag="megatron-smoke")
    torch.testing.assert_close(saved, torch.ones_like(saved))
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    check_metadata()
    check_tools()
    if not args.metadata_only:
        check_gpu()
    print("Megatron image checks passed", flush=True)


if __name__ == "__main__":
    main()
