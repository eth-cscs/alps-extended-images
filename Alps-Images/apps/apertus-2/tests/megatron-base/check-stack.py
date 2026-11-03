"""Check the training stack; GPU imports run only outside builds."""

import argparse
import importlib
import os
import sysconfig
from importlib import metadata
from pathlib import Path

EXPECTED = {
    "transformer-engine": "2.12.0",
    "deep-gemm": "2.6.1",
    "nv-grouped-gemm": "1.1.4.post8",
    "emerging-optimizers": "0.2.0",
    "flash-linear-attention": "0.5.2",
    "causal-conv1d": "1.7.0",
    "nvidia-resiliency-ext": "0.6.0",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    site_dir = Path(sysconfig.get_paths()["purelib"]).resolve()
    for name, expected in EXPECTED.items():
        distribution = metadata.distribution(name)
        print(f"{name}={distribution.version}", flush=True)
        assert distribution.version.split("+")[0] == expected, name
        assert Path(str(distribution.locate_file(""))).resolve().is_relative_to(site_dir), name

    for pin in Path("/opt/alps/apertus2-base-pins.txt").read_text().splitlines():
        name, expected = pin.split("==", 1)
        assert metadata.version(name) == expected, f"Base package changed: {pin}"
    assert "nv25.12" in metadata.version("torch"), metadata.version("torch")
    for pin in Path("/opt/alps/apertus2-requirements.txt").read_text().splitlines():
        name, expected = pin.split("==", 1)
        assert metadata.version(name) == expected, f"Runtime package changed: {pin}"
    if args.metadata_only:
        return
    assert os.environ.get("UCCL_EP_TRANSPORT") == "cxi"

    import torch
    import transformer_engine.pytorch as te
    from emerging_optimizers.orthogonalized_optimizers.muon import get_muon_scale_factor
    from fla.ops.kda.gate import fused_kda_gate

    assert torch.cuda.is_available()
    torch.cuda.set_device(int(os.environ.get("SLURM_LOCALID", "0")))
    assert torch.cuda.get_device_capability() == (9, 0), torch.cuda.get_device_name()
    for name in (
        "uccl.ep",
        "deep_ep",
        "deep_gemm",
        "grouped_gemm",
        "grouped_gemm_backend",
        "emerging_optimizers",
        "fla",
        "causal_conv1d",
        "nvidia_resiliency_ext",
        "transformers",
        "tokenizers",
        "huggingface_hub",
        "wandb",
    ):
        module = importlib.import_module(name)
        module_path = module.__file__
        print(f"{name}: {module_path}", flush=True)
        assert module_path is not None, name
        assert Path(module_path).resolve().is_relative_to(site_dir), name

    # Exercise a native TE forward/backward, not just its package metadata.
    layer = te.Linear(64, 64, params_dtype=torch.bfloat16, device="cuda")
    x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    layer(x).float().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert get_muon_scale_factor(64, 64, mode="moonlight") == 1.6

    # Check the per-channel decay gate used by KDA, including its gradients.
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
    torch.cuda.synchronize()
    print("Apertus2 GPU stack smoke passed", flush=True)


if __name__ == "__main__":
    main()
