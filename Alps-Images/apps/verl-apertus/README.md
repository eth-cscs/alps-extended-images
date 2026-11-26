# `verl-apertus` container image

This image trains Apertus 1.0 (`ApertusForCausalLM`) and Apertus 1.5 (`Apertus1p5ForConditionalGeneration`) with verl, using Megatron training and SGLang generation.
It is the [`verl`](../verl) image plus the Apertus support that the [`rl-bench-apertus-v1.5-70B-sglang-megatron-v1-separate-async.sh`](../verl/apertus-benchmarks/rl-bench-apertus-v1.5-70B-sglang-megatron-v1-separate-async.sh) recipe downloaded, built and patched at job start.
Here it is pinned and applied at build time, so launches need no wheel builds, downloads or runtime patching.

Up to the transformer-engine rebuild, the [Containerfile](Containerfile) matches the `verl` one: verl v0.9.1, SGLang 0.5.16, Megatron-Core 0.19.0, Megatron-Bridge 0.6.1, FlashInfer 0.6.14 and transformer-engine 2.12.0.
Keep the two in step.

## Apertus additions

| Component | Revision | Change |
|-----------|----------|--------|
| transformers | swiss-ai/transformers `986d6dfa` (5.15.0.dev0) | Replaces 5.12.1. Adds Apertus 1.5; Apertus 1.0 is upstream. Installed without dependencies, with a build check that its requirements are met. safetensors is 0.8.0, as the fork requires. |
| xIELU | rubber-duck-debug/xielu `2a55f6b9` | CUDA extension, built for `sm_90`. The Apertus bridges' MLP requires it. |
| Megatron-Core 0.19.0 | `patches/megatron-core` | theely/Megatron-LM `9898eb2b` over `core_v0.19.0`: MLP activation module and the xIELU TP grad-sum. |
| Megatron-Bridge 0.6.1 | `patches/megatron-bridge` | theely/Megatron-Bridge `53bea08c` over `v0.6.0`: the Apertus and Apertus 1.5 bridges, and `safe_config_loader` without `flock` (Lustre has no file locks). |
| SGLang 0.5.16 | `patches/sglang` | sgl-project/sglang#32979 at `9595c6a2`, the Apertus files only. Then: sub-configs given as dicts, unmatched vision/audio tower weights skipped, and `qwen3_asr` registered with `exist_ok`. Apertus 1.0 is upstream. |
| verl v0.9.1 | `patches/verl` | verl-project/verl#7421, #7661 and #7881; `empty_cache()` before the gradient reduce-scatter; `rollout.repetition_penalty` read on the V1 separate-async path; no `flush=` to `logger.info`; and the Apertus 1.5 text-only forward. |

The Apertus 1.5 forward fix turns off verl's vision path for `apertus1p5` only.
The recipe turned it off for every model, which would break other vision-language models in this image.
Apertus 1.5 trains text-only here: the bridge builds the language model, and SGLang loads the vision and audio towers without using them.

sgl-project/sglang#32979 was rebased on 2026-11-22 and its head has moved past `9595c6a2`.
The new head handles sub-configs given as dicts itself, the first of the local fixes, but it is not tested with this image.

## What stays in the job

The recipe's runtime settings are not part of the image.
They include the FlashInfer and Triton cache locations, `SGLANG_DISABLE_CUDA_GRAPH=1`, the reward function, dataset preparation and the verl config.
The recipe also sets `attn_implementation: eager`, because the Apertus 1.5 vision tokenizer supports neither FlashAttention 2 nor SDPA.

## Tests

[`tests/import-smoke.sh`](tests/import-smoke.sh) runs on one GPU node.
It checks the versions, the xIELU CUDA kernel against its Python reference, and the Megatron bridges.
It also checks the SGLang Apertus models and the verl patches.

## Build

Run the [manual-build workflow](../../../manual-build/README.md) from the repository root, for example `manual-build/manual-build.sh app verl-apertus cuda /tmp/build-verl-apertus.sh`.
This workflow selects the canonical base image.
