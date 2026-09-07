# [CSCS, 2026-09-04] Runtime patch for sglang 0.5.16's DeepSeek weight loader
# (sglang/srt/models/deepseek_common/deepseek_weight_loader.py, DeepseekV2WeightLoaderMixin
# .do_load_weights), found debugging train-gsm8k-deepseek-v3-671B-v1-separate-async-megatron.sh
# run 3279810 (19 steps, reward 0.0 / grad_norm 0.0 on every step, ppo_kl ~1.5: the SGLang
# rollout was serving a different model than the trainer).
#
# Mechanism: for q_lora_rank models (DeepSeek-V2/V3) sglang fuses q_a_proj and
# kv_a_proj_with_mqa into one fused_qkv_a_proj_with_mqa parameter at load time, using a
# `cached_a_proj = {}` dict that is LOCAL to each do_load_weights() call -- the fused
# parameter is only written once BOTH halves of a layer arrived in the SAME call. That holds
# for a from-disk load (one call, all tensors), but verl's delta_sharded weight sync
# (verl/workers/rollout/sglang_rollout/delta_loader.py) feeds model.load_weights() in 512 MB
# chunks -- run 3279810's seed sync was 2386 flushes for 45395 tensors -- so any layer whose
# pair straddles a chunk boundary silently keeps its dummy-init (load_format=dummy) fused
# projection. Garbage attention in a handful of layers = garbage generations = zero reward.
#
# Fix: (1) persist the cache on the model object across calls, so a pair completes whenever
# its second half arrives; (2) when a still-unpaired half arrives again (steady syncs ship only
# CHANGED params as NaN-masked full-shape tensors, and skip unchanged ones entirely), MERGE the
# new NaN-masked delta into the cached one instead of overwriting it, so no changed positions
# are lost while waiting for the partner. Dense (seed) tensors carry no NaNs and take the plain
# overwrite path. Correct for both the seed and steady paths; a no-op for non-q_lora models.
#
# Applied on every node inside the main srun (dist-packages is per-container), BEFORE the
# SGLang servers start. Fatal if the anchors are missing (load-bearing for DeepSeek-V3, unlike
# the best-effort watchdog diagnostic) -- re-anchor against the new sglang source if it moves.
# Idempotent. Usage: python3 sglang-deepseek-qkv-a-cache-fix.py [path-to-file-for-testing]
import importlib.util
import py_compile
import sys

MARK = "_cscs_cached_a_proj"

ANCHOR_INIT = "        cached_a_proj = {} if fuse_qkv_a_proj else None\n"
NEW_INIT = (
    "        # [CSCS] cache persists across load_weights calls: verl delta_sharded feeds weights\n"
    "        # in 512 MB chunks, and a per-call cache silently drops any layer whose q_a_proj /\n"
    "        # kv_a_proj_with_mqa pair straddles a chunk boundary (fused projection left at its\n"
    "        # dummy-init value). See sglang-deepseek-qkv-a-cache-fix.py.\n"
    "        if fuse_qkv_a_proj:\n"
    "            if not hasattr(self, \"" + MARK + "\"):\n"
    "                self." + MARK + " = {}\n"
    "            cached_a_proj = self." + MARK + "\n"
    "        else:\n"
    "            cached_a_proj = None\n"
)

ANCHOR_STORE = (
    "                            cached_a_proj[name] = _clone_if_runai_streamed_tensor(\n"
    "                                loaded_weight\n"
    "                            )\n"
)
NEW_STORE = (
    "                            _cscs_new = _clone_if_runai_streamed_tensor(loaded_weight)\n"
    "                            _cscs_old = cached_a_proj.get(name)\n"
    "                            if (\n"
    "                                _cscs_old is not None\n"
    "                                and _cscs_old.shape == _cscs_new.shape\n"
    "                                and _cscs_new.is_floating_point()\n"
    "                            ):\n"
    "                                # [CSCS] steady-sync deltas are NaN-masked; merge, do not overwrite\n"
    "                                _cscs_new = torch.where(torch.isnan(_cscs_new), _cscs_old, _cscs_new)\n"
    "                            cached_a_proj[name] = _cscs_new\n"
)


def main() -> int:
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        spec = importlib.util.find_spec("sglang.srt.models.deepseek_common.deepseek_weight_loader")
        if spec is None or not spec.origin:
            print("FATAL: sglang.srt.models.deepseek_common.deepseek_weight_loader not found")
            return 1
        path = spec.origin
    with open(path) as f:
        src = f.read()
    if MARK in src:
        print("sglang DeepSeek qkv_a cache fix already applied in " + path)
        return 0
    if src.count(ANCHOR_INIT) != 1 or src.count(ANCHOR_STORE) != 1:
        print(
            "FATAL: sglang DeepSeek qkv_a cache fix anchors not found (init=%d store=%d) in %s -- sglang version drift, re-anchor"
            % (src.count(ANCHOR_INIT), src.count(ANCHOR_STORE), path)
        )
        return 1
    if "import torch\n" not in src:
        print("FATAL: expected `import torch` in " + path)
        return 1
    src = src.replace(ANCHOR_INIT, NEW_INIT, 1).replace(ANCHOR_STORE, NEW_STORE, 1)
    with open(path, "w") as f:
        f.write(src)
    py_compile.compile(path, doraise=True)
    print("Applied sglang DeepSeek qkv_a cache fix to " + path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
