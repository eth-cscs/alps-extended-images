#!/usr/bin/env bash
set -euo pipefail

python -c "import vllm.models.deepseek_v32; print('clean exit')"
python -c "from vllm.model_executor.models.registry import ModelRegistry; print(len(ModelRegistry.get_supported_archs()))"
python -c 'from vllm.model_executor.models.registry import ModelRegistry; info = ModelRegistry.models["DeepseekV32ForCausalLM"].inspect_model_cls(); print(info)'
python -c 'from vllm.utils.import_utils import import_triton_kernels; import_triton_kernels(); from triton_kernels.matmul_ogs import PrecisionConfig; print(PrecisionConfig)'
