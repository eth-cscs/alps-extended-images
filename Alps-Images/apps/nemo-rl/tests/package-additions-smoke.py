"""Offline imports and focused GPU smoke checks for validated image additions."""

from __future__ import annotations

import importlib
import pathlib
import subprocess
import tempfile
from importlib import metadata

import numpy as np
import soundfile as sf
import torch

PACKAGES = {
    "aniso8601": ("aniso8601", "10.0.1"),
    "comet-ml": ("comet_ml", "3.58.7"),
    "coverage": ("coverage", "7.16.2"),
    "cupy-cuda13x": ("cupy", "14.2.0"),
    "fastokens": ("fastokens", "0.3.2"),
    "fast-hadamard-transform": ("fast_hadamard_transform", "1.1.0"),
    "flashinfer-jit-cache": ("flashinfer_jit_cache", "0.6.16.post3"),
    "flake8": ("flake8", "7.4.1"),
    "Flask-RESTful": ("flask_restful", "0.3.10"),
    "hatchling": ("hatchling", "1.32.4"),
    "mock": ("mock", "5.2.0"),
    "mypy": ("mypy", "2.4.0"),
    "nemo-lens": ("nemo.lens", "0.1.0"),
    "nemo-run": ("nemo_run", "0.11.1"),
    "nltk": ("nltk", "3.10.3"),
    "nvidia-mathdx": ("nvidia", "25.6.0"),
    "open-clip-torch": ("open_clip", "3.3.0"),
    "peft": ("peft", "0.21.2"),
    "pre-commit": ("pre_commit", "4.6.2"),
    "PyGithub": ("github", "2.10.0"),
    "pylint": ("pylint", "4.1.2"),
    "pyrefly": ("pyrefly", "1.3.2"),
    "pytest-asyncio": ("pytest_asyncio", "1.4.0"),
    "pytest-cov": ("pytest_cov", "7.1.0"),
    "pytest-mock": ("pytest_mock", "3.16.0"),
    "pytest-random-order": ("random_order", "1.2.0"),
    "pytest-runner": (None, "6.0.1"),
    "pytest-shard": ("pytest_shard", "0.1.2"),
    "pytest-testmon": ("testmon", "2.2.0"),
    "pytest-timeout": ("pytest_timeout", "2.4.0"),
    "ruff": ("ruff", "0.16.10"),
    "soundfile": ("soundfile", "0.14.0"),
    "tensorstore": ("tensorstore", "0.1.85"),
    "timm": ("timm", "1.0.30"),
    "torch-memory-saver": ("torch_memory_saver", "0.0.10"),
    "ty": ("ty", "0.0.85"),
    "types-requests": ("requests", "2.31.0.6"),
    "wget": ("wget", "3.2"),
}

for distribution, (module, expected) in PACKAGES.items():
    actual = metadata.version(distribution)
    assert actual == expected, f"{distribution}: expected {expected}, got {actual}"
    if module:
        importlib.import_module(module)
    print(f"{distribution} {actual}: import/version PASS")

for command in (["ruff", "--version"], ["nemo", "--help"]):
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=60
    )
    assert result.stdout or result.stderr, f"empty output from {command[0]}"
    print(f"{' '.join(command)} CLI PASS")

assert metadata.version("urllib3") == "1.26.20"
assert metadata.version("types-urllib3") == "1.26.25.14"
assert torch.cuda.is_available(), "GPU smoke requires a visible CUDA device"

# SoundFile native library: deterministic WAV roundtrip, no external fixture.
with tempfile.TemporaryDirectory() as tmp:
    path = pathlib.Path(tmp) / "roundtrip.wav"
    samples = np.sin(np.arange(160, dtype=np.float32) / 10) * 0.5
    sf.write(path, samples, 8000, subtype="PCM_16")
    restored, rate = sf.read(path, dtype="float32")
    assert rate == 8000 and restored.shape == samples.shape
    np.testing.assert_allclose(restored, samples, atol=4e-5, rtol=0)
print("SoundFile WAV roundtrip PASS")

# Fastokens against an entirely local, known-good ByteLevel BPE fixture.
import fastokens
import tokenizers

vocab = {
    "<unk>": 0,
    "h": 1,
    "e": 2,
    "l": 3,
    "o": 4,
    "Ġ": 5,
    "w": 6,
    "r": 7,
    "d": 8,
    "he": 9,
    "hel": 10,
    "hell": 11,
    "hello": 12,
    "Ġw": 13,
    "Ġwo": 14,
    "Ġwor": 15,
    "Ġworl": 16,
    "Ġworld": 17,
}
merges = [
    ("h", "e"),
    ("he", "l"),
    ("hel", "l"),
    ("hell", "o"),
    ("Ġ", "w"),
    ("Ġw", "o"),
    ("Ġwo", "r"),
    ("Ġwor", "l"),
    ("Ġworl", "d"),
]
reference = tokenizers.Tokenizer(
    tokenizers.models.BPE(vocab=vocab, merges=merges, unk_token="<unk>")
)
reference.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
reference.decoder = tokenizers.decoders.ByteLevel()
fast = fastokens.Tokenizer.from_json_str(reference.to_str())
texts = ["hello world", "hello", "world hello"]
ids = [reference.encode(text).ids for text in texts]
assert [fast.encode(text).ids for text in texts] == ids
assert [item.ids for item in fast.encode_batch(texts)] == ids
expected = [reference.decode(item) for item in ids]
assert [fast.decode(item) for item in ids] == expected
assert fast.decode_batch(ids) == expected
print("fastokens ByteLevel BPE PASS")

# Actual SM90 FWHT kernel against a dense reference in three supported dtypes.
from fast_hadamard_transform import hadamard_transform

for size, dtype in ((8, torch.float16), (16, torch.bfloat16), (32, torch.float32)):
    x = torch.arange(1, size + 1, device="cuda", dtype=dtype).reshape(1, size)
    got = hadamard_transform(x, scale=size**-0.5)
    h = torch.ones((1, 1), device="cuda")
    while h.shape[0] < size:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    torch.testing.assert_close(
        got.float(), x.float() @ h / size**0.5, rtol=0.02, atol=0.02
    )
torch.cuda.synchronize()
print("fast-hadamard-transform FP16/BF16/FP32 GPU references PASS")

from torch_memory_saver import torch_memory_saver as saver

original = torch.arange(1024, device="cuda", dtype=torch.float32)
with saver.region(tag="smoke", enable_cpu_backup=True):
    saved = original.clone()
torch.cuda.synchronize()
saver.pause(tag="smoke")
saver.resume(tag="smoke")
torch.testing.assert_close(saved, original)
torch.testing.assert_close(
    original + 1, torch.arange(1, 1025, device="cuda", dtype=torch.float32)
)
print("TorchMemorySaver default preload pause/resume PASS")

# The packaged SM90a cache should serve FlashInfer's kernel without a JIT compile.
import flashinfer as fi
import flashinfer_jit_cache as fi_cache

cache_module = (
    pathlib.Path(fi_cache.get_jit_cache_dir()) / "silu_and_mul" / "silu_and_mul.so"
)
assert cache_module.is_file(), cache_module
cache_x = torch.randn((4, 16), device="cuda", dtype=torch.float16).contiguous()
cache_y = fi.silu_and_mul(cache_x)
torch.cuda.synchronize()
torch.testing.assert_close(
    cache_y,
    torch.nn.functional.silu(cache_x[:, :8]) * cache_x[:, 8:],
    rtol=1e-3,
    atol=1e-3,
)
cache_maps = pathlib.Path("/proc/self/maps").read_text()
assert str(cache_module) in cache_maps, (
    "FlashInfer did not map the packaged cache module"
)
print("FlashInfer cached SM90a silu_and_mul FP16/reference + mapped module PASS")

# NemoLens trace API using an in-memory exporter only (no telemetry network).
import os

os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
os.environ.pop("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", None)
from nemo import lens
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

assert metadata.metadata("nemo-lens")["Requires-Python"] == ">=3.13"
assert metadata.version("opentelemetry-sdk") == "1.45.1"
assert metadata.version("protobuf") == "6.33.6"
exporter = InMemorySpanExporter()
handle = lens.setup_telemetry(
    lens.NemoLensConfig(
        enabled=True,
        service_name="nemo-rl-package-smoke",
        export_strategy="single_rank",
        traces_enabled=True,
        metrics_enabled=False,
        logs_enabled=False,
        span_groups="per_step",
        run_id="package-smoke",
    ),
    rank=0,
    world_size=1,
    span_exporter=exporter,
)
with lens.managed_span("step", "offline-smoke", tracer=handle.tracer, iteration=3):
    pass
handle.shutdown()
spans = exporter.get_finished_spans()
assert len(spans) == 1 and spans[0].name == "offline-smoke", spans
assert spans[0].attributes["iteration"] == 3
print("NemoLens in-memory trace export PASS (Python 3.12 metadata override)")

# CuPy's CUDA 13 wheel must interoperate with the active CUDA device.
import cupy

array = cupy.arange(16, dtype=cupy.float32).reshape(4, 4)
cupy.testing.assert_allclose(array @ array.T, np.asarray(array.get() @ array.get().T))
assert float(array.sum().get()) == 120.0
print("CuPy CUDA reduction/GEMM PASS")

# TensorStore in-memory Zarr roundtrip; no backing files or network access.
import tensorstore as ts

store = ts.open(
    {"driver": "zarr", "kvstore": "memory://", "path": "smoke"},
    create=True,
    dtype=ts.float32,
    shape=[2, 2],
).result()
store.write(np.array([[1, 2], [3, 4]], dtype=np.float32)).result()
np.testing.assert_array_equal(store.read().result(), [[1, 2], [3, 4]])
print("TensorStore in-memory roundtrip PASS")

# Compile and execute a representative cuFFTDx SM90 kernel from the shipped SDK.
mathdx = metadata.distribution("nvidia-mathdx")
include = pathlib.Path(str(mathdx.locate_file("nvidia/mathdx/include")))
cutlass = pathlib.Path(
    str(mathdx.locate_file("nvidia/mathdx/external/cutlass/include"))
)
assert (include / "cufftdx.hpp").is_file(), include
source = pathlib.Path(__file__).with_name("mathdx-cufftdx-smoke.cu")
with tempfile.TemporaryDirectory() as tmp:
    executable = pathlib.Path(tmp) / "mathdx-cufftdx-smoke"
    subprocess.run(
        [
            "/usr/local/cuda/bin/nvcc",
            "-std=c++17",
            "-arch=sm_90",
            f"-I{include}",
            f"-I{cutlass}",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
print("MathDx cuFFTDx SM90 compile/runtime PASS (cuBLASDx/cuSolverDx not covered)")
