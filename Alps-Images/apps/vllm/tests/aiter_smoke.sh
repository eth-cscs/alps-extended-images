#!/usr/bin/env bash
# Single-node smoke test for the AITER install in the ROCm vLLM image:
# the package imports, vLLM resolves its AITER MLA prefill backend, and one
# kernel JIT-compiles and runs on a GPU (exercises hipcc and the ROCm prefix).
# The first step runs against the cache baked into the image and fails if it recompiles.
set -euo pipefail

baked_dir="${AITER_JIT_DIR:?AITER_JIT_DIR unset; not the ROCm vLLM image?}"
work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT
export VLLM_ROCM_USE_AITER=1

touch "${work_dir}/marker"
python -c "
import aiter
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum
MLAPrefillBackendEnum.ROCM_AITER_FA.get_class()
"
rebuilt="$(find "${baked_dir}" -maxdepth 1 -name '*.so' -newer "${work_dir}/marker")"
[[ -z "${rebuilt}" ]] || { echo "FATAL: baked AITER cache was recompiled: ${rebuilt}" >&2; exit 1; }
echo "OK: baked AITER cache in ${baked_dir} used without recompiling"

# JIT into a private copy so the image cache stays untouched and concurrent runs cannot clash.
export AITER_JIT_DIR="${work_dir}/jit"
mkdir -p "${AITER_JIT_DIR}"
cp -r "${baked_dir}/." "${AITER_JIT_DIR}/"

python - <<'PY'
import torch
import aiter
from vllm._aiter_ops import rocm_aiter_ops
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum

assert torch.cuda.is_available(), "no GPU visible"
assert rocm_aiter_ops.is_enabled(), "vLLM does not consider AITER enabled"
backend = MLAPrefillBackendEnum.ROCM_AITER_FA.get_class()
assert backend.is_available(), "AITER MLA prefill backend unavailable"

x = torch.randn(64, 4096, dtype=torch.bfloat16, device="cuda")
w = torch.ones(4096, dtype=torch.bfloat16, device="cuda")
out = aiter.rms_norm(x, w, 1e-6)
ref = torch.nn.functional.rms_norm(x.float(), (4096,), w.float(), 1e-6).to(torch.bfloat16)
torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
print("OK: aiter rms_norm JIT-compiled and matches torch")
PY
