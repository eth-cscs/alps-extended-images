"""Validate the pinned native gfx942 training stack; --metadata-only needs no GPU.

Numerical checks deliberately use torch.cuda: that is PyTorch's ROCm device API.
Run one process per visible GPU; all writable JIT caches are private under /tmp.
"""

import argparse
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path("/opt/alps/apertus2-rocm")
SOURCES = {
    "transformer-engine": ("TE", "b56021a0e4a2e4a7822482ffdf5353368a777e0e"),
    "uccl": ("UCCL", "18b588a048a3a3fe3d4bd60aa5678c66e284df31"),
    "megatron": ("MEGATRON", "0789b8e429553c2f17c350afa8619c69d01bec25"),
    "emerging-optimizers": (
        "EMERGING_OPTIMIZERS", "cc1385eec1966aea2aebfb6dd6fc66a898312a55"
    ),
    "flash-linear-attention": ("FLA", "1820dba7e15fb927294779f23ac5097eb3927b25"),
    "causal-conv1d": ("CAUSAL_CONV1D", "cd81f0413cad2fc1e6f17e785ac39f59aae690cd"),
}
# Public versions read from the pinned sources, not CUDA wheel release numbers.
NATIVE_VERSIONS = {
    "transformer-engine": "2.20.0.dev0",
    "emerging-optimizers": "0.2.0",
    "flash-linear-attention": "0.5.2",
    "causal-conv1d": "1.7.0",
    "uccl": "0.1.1",
    "deep-ep": "0.1.0",
    "megatron-core": "0.17.0rc0",
}
MODULES = (
    "torch", "triton", "numpy", "transformer_engine.pytorch",
    "emerging_optimizers", "fla.ops.kda",
    "causal_conv1d", "causal_conv1d_cuda", "uccl.ep", "deep_ep",
    "megatron.core", "megatron.training", "accelerate", "einops",
    "transformers", "tokenizers", "huggingface_hub", "wandb", "yaml",
    "omegaconf", "tensorstore", "tensorboard", "sentencepiece", "tiktoken",
)
CUDA_PACKAGES = {
    "deep-gemm", "nv-grouped-gemm", "flashinfer-python", "flashinfer-jit-cache",
    "torch-memory-saver", "flash-attn", "flash-attn-3", "transformer-engine-cu12",
    "transformer-engine-cu13", "nvidia-modelopt", "nvidia-resiliency-ext",
}
CUDA_LIBRARY = re.compile(
    r"lib(?:cuda|cudart|cublas(?:Lt)?|cudnn[^.]*|cufft|curand|cusolver|cusparse|"
    r"nvrtc[^.]*|nvJitLink|nvToolsExt|nccl)\.so(?:\.|$)|"
    r"torch_memory_saver.*(?:cu12|cu13)|/cuda(?:/|-)"
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def module_spec(name):
    """Resolve children without importing parents, including PEP 660 editables."""
    parts = name.split(".")
    spec = importlib.util.find_spec(parts[0])
    for index in range(1, len(parts)):
        if spec is None or spec.submodule_search_locations is None:
            return None
        path = spec.submodule_search_locations
        spec = None
        for finder in sys.meta_path:
            spec = finder.find_spec(".".join(parts[:index + 1]), path)
            if spec is not None:
                break
    return spec


def check_metadata():
    pins = {}
    for line in (ROOT / "base-pins.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        pins[canonicalize_name(name)] = expected
        actual = metadata.version(name)
        require(actual == expected, f"Base pin changed: {name}: {actual} != {expected}")
    require({"torch", "numpy"} <= pins.keys(), "Missing torch/numpy base pins")
    require(any("triton" in name for name in pins), "Missing inherited Triton base pin")
    require(Version(pins["torch"]).release[:2] == (2, 13), "Expected Torch 2.13 base")

    for line in (ROOT / "requirements.lock").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        actual = metadata.version(requirement.name)
        require(requirement.specifier.contains(actual, prereleases=True),
                f"Unsatisfied training dependency: {requirement}; installed {actual}")
    for name, expected in NATIVE_VERSIONS.items():
        actual = metadata.version(name)
        require(Version(actual).public == expected,
                f"Pinned source version mismatch: {name}: {actual} != {expected}")
        print(f"{name}={actual}", flush=True)
    for name in MODULES:
        spec = module_spec(name)
        require(spec is not None, f"Missing installed module: {name}")
        print(f"module {name}: {spec.origin or list(spec.submodule_search_locations)}", flush=True)

    configured = {}
    for line in (ROOT / "versions.env").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            key, value = line.split("=", 1)
            configured[key] = shlex.split(value)[0]
    manifest = {}
    for line in (ROOT / "source-pins.txt").read_text().splitlines():
        name, repo, ref = line.split()
        require(name not in manifest, f"Duplicate source manifest entry: {name}")
        manifest[name] = (repo, ref)
    require(manifest.keys() == SOURCES.keys(), f"Unexpected source manifest: {manifest}")
    for name, (prefix, expected) in SOURCES.items():
        require(configured[f"{prefix}_REF"] == expected,
                f"Configured {name} ref is not the audited pin {expected}")
        require(manifest[name] == (configured[f"{prefix}_REPO"], expected),
                f"Source manifest disagrees with versions.env: {name}: {manifest[name]}")
    for name, path in (("megatron", "/opt/megatron-lm"), ("uccl", "/opt/uccl")):
        actual = subprocess.check_output(
            ["git", "-c", f"safe.directory={path}", "-C", path, "rev-parse", "HEAD"],
            text=True, timeout=10,
        ).strip()
        require(actual == manifest[name][1], f"Retained {name} checkout changed: {actual}")
    direct_url = metadata.distribution("megatron-core").read_text("direct_url.json")
    require(direct_url is not None, "Missing Megatron editable installation metadata")
    direct_url = json.loads(direct_url)
    require(direct_url.get("dir_info", {}).get("editable") is True
            and direct_url.get("url") == "file:///opt/megatron-lm",
            f"Megatron must be editable from /opt/megatron-lm: {direct_url}")

    distributions = list(metadata.distributions())
    forbidden = sorted({canonicalize_name(dist.metadata["Name"]) for dist in distributions
                        if canonicalize_name(dist.metadata["Name"]) in CUDA_PACKAGES})
    require(not forbidden, f"CUDA-only packages installed: {forbidden}")
    preloads = os.environ.get("LD_PRELOAD", "")
    system_preload = Path("/etc/ld.so.preload")
    if system_preload.exists():
        preloads += " " + system_preload.read_text()
    require(not CUDA_LIBRARY.search(preloads), f"CUDA preload configured: {preloads}")
    # Inspect ELF metadata, never dlopen extensions during the GPU-less build.
    libraries = set()
    for name in ("torch", "transformer-engine", "uccl", "causal-conv1d"):
        dist = metadata.distribution(name)
        for file in dist.files or ():
            if ".so" in file.name:
                path = Path(dist.locate_file(file))
                if path.is_file():
                    libraries.add(path.resolve())
    for name in ("uccl.ep", "causal_conv1d_cuda"):
        spec = module_spec(name)
        require(spec is not None and spec.origin, f"Missing native extension: {name}")
        libraries.add(Path(spec.origin).resolve())
    for path in sorted(libraries):
        dynamic = subprocess.check_output(
            ["readelf", "-d", str(path)], text=True, timeout=30
        )
        needed = re.findall(r"\(NEEDED\).*\[([^]]+)\]", dynamic)
        require(not any(CUDA_LIBRARY.search(name) for name in needed),
                f"CUDA binary dependency: {path}: {needed}")
    print(f"metadata OK: preserved base pins, source manifests, modules; "
          f"{len(libraries)} ELF libraries free of CUDA dependencies", flush=True)


def set_test_caches(directory):
    root = Path(directory)
    for variable, directory in {
        "XDG_CACHE_HOME": "xdg", "TORCH_HOME": "torch",
        "TORCH_EXTENSIONS_DIR": "torch-extensions", "TRITON_CACHE_DIR": "triton",
        "TORCHINDUCTOR_CACHE_DIR": "inductor", "MIOPEN_USER_DB_PATH": "miopen-db",
        "MIOPEN_CUSTOM_CACHE_DIR": "miopen", "CK_JIT_CACHE_DIR": "te-ck",
    }.items():
        path = root / directory
        path.mkdir()
        os.environ[variable] = str(path)
    # CK_JIT_ROOT is the installed blob source root, not a writable cache. Leave
    # it alone; ck-jit e0b3031 supports CK_JIT_CACHE_DIR for runtime compilation.
    print(f"Private runtime/JIT caches: {root}", flush=True)


def close(torch, label, actual, expected, *, atol=1e-4, rtol=1e-4, rms=None):
    require(actual is not None, f"{label}: missing tensor/gradient")
    require(bool(torch.isfinite(actual).all()), f"{label}: nonfinite result")
    require(bool(torch.isfinite(expected).all()), f"{label}: nonfinite reference")
    delta = actual.float() - expected.float()
    maximum = delta.abs().max().item()
    relative_rms = (delta.square().mean().sqrt()
                    / expected.float().square().mean().sqrt().clamp_min(1e-8)).item()
    print(f"{label}: max_abs={maximum:.6g}, relative_rms={relative_rms:.6g}", flush=True)
    if rms is None:
        torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol,
                                   msg=lambda message: f"{label}: {message}")
    else:
        require(relative_rms <= rms,
                f"{label}: relative RMS error {relative_rms:.6g} exceeds {rms}")


def check_te(torch):
    import transformer_engine.pytorch as te

    for grouped in (False, True):
        label = "TE GroupedLinear" if grouped else "TE Linear"
        splits = [16, 32]
        if grouped:
            layer = te.GroupedLinear(2, 64, 64, bias=True, params_dtype=torch.bfloat16,
                                     device="cuda", use_grouped_tensor=False)
            weights = [layer.weight0, layer.weight1]
            biases = [layer.bias0, layer.bias1]
        else:
            layer = te.Linear(64, 64, bias=True, params_dtype=torch.bfloat16, device="cuda")
            weights, biases = [layer.weight], [layer.bias]
        x = torch.randn(sum(splits), 64, device="cuda", dtype=torch.bfloat16,
                        requires_grad=True)
        ref_x = x.detach().float().requires_grad_()
        ref_weights = [p.detach().float().requires_grad_() for p in weights]
        ref_biases = [p.detach().float().requires_grad_() for p in biases]
        actual = layer(x, splits) if grouped else layer(x)
        if grouped:
            expected = torch.cat([
                torch.nn.functional.linear(part, weight, bias)
                for part, weight, bias in zip(ref_x.split(splits), ref_weights, ref_biases,
                                              strict=True)
            ])
        else:
            expected = torch.nn.functional.linear(ref_x, ref_weights[0], ref_biases[0])
        close(torch, f"{label} forward", actual, expected, atol=3e-3, rtol=2e-2)
        upstream = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, [x, *weights, *biases], upstream)
        ref_grads = torch.autograd.grad(expected, [ref_x, *ref_weights, *ref_biases],
                                        upstream.float())
        names = ["input", *[f"weight{i}" for i in range(len(weights))],
                 *[f"bias{i}" for i in range(len(biases))]]
        for name, grad, ref in zip(names, actual_grads, ref_grads, strict=True):
            close(torch, f"{label} d{name}", grad, ref, atol=5e-2, rtol=2e-2)


def check_kda(torch):
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.gate import fused_kda_gate
    from fla.ops.kda.naive import naive_recurrent_kda

    shape = (1, 128, 2, 64)  # Two chunks exercise the recurrent state boundary.
    g = torch.randn(shape, device="cuda", requires_grad=True)
    a_log = (torch.randn(2, 64, device="cuda") * 0.3).requires_grad_()
    bias = torch.randn(128, device="cuda", requires_grad=True)
    # The pinned naive_kda_gate only handles per-head A_log, so use an explicit
    # torch expression for this SwissAI per-channel extension.
    for lower_bound in (None, -5.0):
        actual = fused_kda_gate(g, a_log, bias, lower_bound=lower_bound)
        shifted = g + bias.view(2, 64)
        expected = (-a_log.exp() * torch.nn.functional.softplus(shifted)
                    if lower_bound is None
                    else lower_bound * torch.sigmoid(a_log.exp() * shifted))
        suffix = "softplus" if lower_bound is None else "lowerbound"
        close(torch, f"KDA per-channel gate {suffix}", actual, expected)
        upstream = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, (g, a_log, bias), upstream)
        ref_grads = torch.autograd.grad(expected, (g, a_log, bias), upstream)
        for name, grad, ref in zip(("g", "A_log", "bias"), actual_grads, ref_grads,
                                  strict=True):
            close(torch, f"KDA gate {suffix} d{name}", grad, ref, atol=2e-3, rtol=2e-3)

    q = torch.nn.functional.normalize(torch.randn(shape, device="cuda"), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(shape, device="cuda"), dim=-1)
    v = torch.randn(shape, device="cuda")
    gate = -torch.rand(shape, device="cuda") * 0.05
    beta = torch.rand(shape[:-1], device="cuda")
    state = torch.randn(1, 2, 64, 64, device="cuda") * 0.1
    inputs = [p.to(torch.bfloat16).requires_grad_() for p in (q, k, v)]
    inputs += [gate.requires_grad_(), beta.to(torch.bfloat16).requires_grad_(),
               state.requires_grad_()]
    reference = [p.detach().float().requires_grad_() for p in inputs]
    q, k, v, gate, beta, state = inputs
    actual, final = chunk_kda(q, k, v, gate, beta, initial_state=state,
                              output_final_state=True, chunk_size=64)
    # Same reference used by pinned tests/ops/test_kda.py::test_chunk.
    expected, ref_final = naive_recurrent_kda(
        *reference[:5], initial_state=reference[5], output_final_state=True
    )
    close(torch, "chunk_kda output", actual, expected, rms=0.02)
    close(torch, "chunk_kda final state", final, ref_final, rms=0.02)
    upstream, final_upstream = torch.randn_like(actual), torch.randn_like(final)
    actual_grads = torch.autograd.grad((actual, final), inputs, (upstream, final_upstream))
    ref_grads = torch.autograd.grad((expected, ref_final), reference,
                                    (upstream.float(), final_upstream.float()))
    for name, grad, ref in zip(("q", "k", "v", "gate", "beta", "initial_state"),
                              actual_grads, ref_grads, strict=True):
        close(torch, f"chunk_kda d{name}", grad, ref, rms=0.03)


def check_causal_conv(torch):
    from causal_conv1d import causal_conv1d_fn

    for activation in (None, "silu"):
        x = torch.randn(2, 64, 128, device="cuda", requires_grad=True)
        weight = torch.randn(64, 4, device="cuda", requires_grad=True)
        bias = torch.randn(64, device="cuda", requires_grad=True)
        actual = causal_conv1d_fn(x, weight, bias, activation=activation)
        expected = torch.nn.functional.conv1d(x, weight[:, None], bias,
                                              padding=3, groups=64)[..., :128]
        if activation:
            expected = torch.nn.functional.silu(expected)
        close(torch, f"causal-conv1d {activation} forward", actual, expected,
              atol=2e-4, rtol=2e-4)
        upstream = torch.randn_like(actual)
        grads = torch.autograd.grad(actual, (x, weight, bias), upstream)
        refs = torch.autograd.grad(expected, (x, weight, bias), upstream)
        for name, grad, ref in zip(("input", "weight", "bias"), grads, refs, strict=True):
            close(torch, f"causal-conv1d {activation} d{name}", grad, ref,
                  atol=2e-3, rtol=2e-3)


def check_muon(torch):
    from emerging_optimizers.orthogonalized_optimizers.muon import Muon, get_muon_scale_factor

    lr, momentum, decay = 0.02, 0.8, 0.1
    for shape in ((32, 64), (64, 32)):
        scale = 0.2 * max(shape) ** 0.5
        require(get_muon_scale_factor(*shape, mode="moonlight") == scale,
                f"Incorrect moonlight scaling for {shape}")
        parameter = torch.nn.Parameter(torch.randn(*shape, device="cuda"))
        expected = parameter.detach().clone()
        buffer = torch.zeros_like(parameter)
        optimizer = Muon([parameter], lr=lr, momentum=momentum, weight_decay=decay,
                         nesterov=True, scale_mode="moonlight", coefficient_type="simple",
                         num_ns_steps=3, fp32_matmul_prec="highest", use_syrk=False)
        for step in range(2):
            gradient = torch.randn_like(parameter)
            parameter.grad = gradient.clone()
            # Independent torch reference for pinned EMA/Nesterov, simple NS
            # polynomial and decoupled decay. Two steps exercise optimizer state.
            buffer = momentum * buffer + (1 - momentum) * gradient
            update = (1 - momentum) * gradient + momentum * buffer
            transpose = shape[0] > shape[1]
            update = update.mT if transpose else update
            update = update / update.norm().clamp_min(1e-7)
            for _ in range(3):
                gram = update @ update.mT
                update = 3.4445 * update + (-4.7750 * gram + 2.0315 * (gram @ gram)) @ update
            update = update.mT if transpose else update
            expected = expected * (1 - lr * decay) - lr * scale * update
            optimizer.step()
            close(torch, f"Muon moonlight {shape} step {step + 1}", parameter, expected,
                  atol=2e-5, rtol=2e-5)
            close(torch, f"Muon momentum {shape} step {step + 1}",
                  optimizer.state[parameter]["momentum_buffer"], buffer)


def check_gpu(cache_root):
    set_test_caches(cache_root)
    os.environ["NVTE_FUSED_ATTN_CK"] = "1"
    os.environ["NVTE_FUSED_ATTN_AOTRITON"] = "0"
    import torch

    require(torch.version.hip is not None and torch.version.cuda is None,
            f"Expected HIP Torch, got hip={torch.version.hip}, cuda={torch.version.cuda}")
    require(torch.cuda.is_available(), "No accessible HIP GPU")
    # SLURM_LOCALID may be nonzero when each task sees only its own GPU.
    device = 0 if torch.cuda.device_count() == 1 else int(os.environ.get("SLURM_LOCALID", "0"))
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    require(properties.gcnArchName.split(":")[0] == "gfx942",
            f"Expected gfx942, got {properties.gcnArchName}")
    print(f"HIP {torch.version.hip}, {properties.name}, {properties.gcnArchName}", flush=True)
    torch.manual_seed(1234)
    torch.set_float32_matmul_precision("highest")
    for check in (check_te, check_kda, check_causal_conv, check_muon):
        print(f"Running {check.__name__}", flush=True)
        check(torch)
        torch.cuda.synchronize()
    print("Native ROCm forward/backward and optimizer numerical checks passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true",
                        help="Inspect package/source/ELF metadata without importing GPU packages")
    args = parser.parse_args()
    check_metadata()
    if not args.metadata_only:
        with tempfile.TemporaryDirectory(prefix="apertus2-rocm-", dir="/tmp") as cache_root:
            check_gpu(cache_root)


if __name__ == "__main__":
    main()
