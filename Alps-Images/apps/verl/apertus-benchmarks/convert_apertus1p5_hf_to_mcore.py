"""One-time offline conversion: Apertus-v1.5-70B HF checkpoint -> Megatron dist-checkpoint.

Why this exists (see CLAUDE.md "Model load phase" analysis, 2026-09-08, and the follow-up
session that produced this script): megatron-bridge's load_hf_weights (called at the start of
every training run via AutoBridge.load_hf_weights -> Apertus1p5Bridge.load_weights_hf_to_megatron)
has every trainer rank read the ENTIRE HF safetensors checkpoint from Lustre with
safe_open().get_tensor(), even though only tp_rank==0 actually needs the raw tensor (other TP
ranks get their shard via in-memory split/scatter; replicated params via broadcast). At this
recipe's scale (TP=8, ~12 DP replicas) that means ~14 TB of redundant reads for a 146 GB
checkpoint, and 30-57 minutes of wall-clock on every single run regardless of node count or
Lustre tier (confirmed: /capstor/scratch and /capstor/store show the same load time).

The fix: convert once to Megatron's native distributed checkpoint format
(megatron.core.dist_checkpointing), which is pre-sharded per rank on disk -- each rank then reads
only its own shard directly, no redundant scanning, no split/scatter at load time. This format is
also reshardable (megatron.core.dist_checkpointing.load can remap saved shards onto a DIFFERENT
TP/PP/EP layout than what was used to save), so the conversion does not need to match the real
training recipe's TP=8 exactly -- this script defaults to TP=4/PP=1 (fits one 4-GPU GH200 node,
avoiding a multi-node torchrun launch) and the training recipe can still load it at TP=8.

This deliberately mirrors verl's own real training-time model construction path exactly
(verl/workers/engine/megatron/transformer_impl.py TransformerEngine._build_tf_config /
_build_megatron_module, non-vanilla-bridge branch), rather than the older, apertus1p5-unaware
generic converter (scripts/converter_hf_to_mcore.py, which uses
verl.models.mcore.loader.load_state_dict_to_megatron_gptmodel and has no apertus1p5 case):

    bridge = AutoBridge.from_hf_pretrained(hf_model_path, trust_remote_code=False)
    provider = bridge.to_megatron_provider(load_weights=False)     # shape only
    ...apply the same provider_overrides (TP/PP/EP, sequence_parallel, etc.)...
    module, _ = make_megatron_module(..., bridge=bridge, provider=provider, ...)
    bridge.load_hf_weights(module, hf_model_path, allowed_mismatched_params=[])  # the real, slow
                                                                                  # HF load, ONCE
    dist_checkpointing.save(unwrap_model(module[0]).sharded_state_dict(), output_path)

AutoBridge.from_hf_pretrained/to_megatron_provider/load_hf_weights are megatron-bridge's own
public API (verl.models.mcore.bridge.AutoBridge is a re-export of megatron.bridge.AutoBridge,
confirmed against the real NVIDIA-NeMo/Megatron-Bridge v0.6.0 source); AutoBridge.from_hf_pretrained
dispatches to Apertus1p5Bridge (registered by the same Megatron-Bridge compare-diff patch this
repo's training recipe already applies) purely from the checkpoint's `architectures` field, so no
apertus1p5-specific code needs to be written here -- it reuses the exact bridge already validated
by every real training run.

To USE the converted checkpoint at training time, add to the recipe's actor.megatron block:
    use_dist_checkpointing: True
    dist_checkpointing_path: <output_path>
(verl/workers/engine/megatron/transformer_impl.py:_build_megatron_module branches on
use_dist_checkpointing to call load_mcore_dist_weights(module, dist_checkpointing_path, ...)
INSTEAD OF bridge.load_hf_weights(...) -- the model SHAPE is still built via the same bridge
either way; only the weight-loading step changes.) ref.megatron would need the same treatment.

NEVER RUN END-TO-END. Verified only against real verl v0.9.0 / megatron-bridge v0.6.0 source
(fetched and read directly, not from memory) -- the API calls are real and match what verl's own
trainer does every day, but the full save -> reload -> train round-trip on this specific model has
not been exercised on a cluster. Run the round-trip self-test (RUN_ROUNDTRIP_TEST=1) before
trusting the output for anything beyond a probe.
"""

import os

import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing
from megatron.core import parallel_state as mpu
from megatron.core.dist_checkpointing.serialization import StrictHandling
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from transformers import AutoConfig

from verl.models.mcore.bridge import AutoBridge
from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module, unwrap_model


def log(msg: str) -> None:
    if dist.get_rank() == 0:
        print(f"[convert] {msg}", flush=True)


def main() -> None:
    hf_model_path = os.environ["HF_MODEL_PATH"]
    output_path = os.environ["OUTPUT_PATH"]
    tp_size = int(os.environ.get("CONVERT_TP", "4"))
    pp_size = int(os.environ.get("CONVERT_PP", "1"))
    ep_size = int(os.environ.get("CONVERT_EP", "1"))
    run_roundtrip_test = os.environ.get("RUN_ROUNDTRIP_TEST", "0") == "1"

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    if world_size != tp_size * pp_size * ep_size:
        raise SystemExit(
            f"world_size={world_size} != tp_size*pp_size*ep_size={tp_size * pp_size * ep_size} "
            "-- fix --nproc_per_node/--nnodes or CONVERT_{TP,PP,EP}"
        )

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=ep_size,
    )
    model_parallel_cuda_manual_seed(0)

    log(f"Loading HF config/bridge from {hf_model_path} (TP={tp_size} PP={pp_size} EP={ep_size})")
    hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=False)
    bridge = AutoBridge.from_hf_pretrained(hf_model_path, trust_remote_code=False)

    # Shape only (load_weights=False): identical to verl's own
    # transformer_impl.py:_build_tf_config non-vanilla-bridge branch.
    provider = bridge.to_megatron_provider(load_weights=False)
    provider_overrides = {
        "tensor_model_parallel_size": tp_size,
        "pipeline_model_parallel_size": pp_size,
        "expert_model_parallel_size": ep_size,
        "expert_tensor_parallel_size": 1,
        "virtual_pipeline_model_parallel_size": None,
        "context_parallel_size": 1,
        "sequence_parallel": False,  # matches actor.megatron.sequence_parallel: False in the recipe
        "overlap_p2p_comm": False,
        "batch_p2p_comm": False,
        "variable_seq_lengths": True,
        "attention_backend": AttnBackend.flash,
        "moe_token_dispatcher_type": "alltoall",
        "moe_router_load_balancing_type": "none",
    }
    provider.apply_overrides_and_finalize(dtype=torch.bfloat16, overrides=provider_overrides)

    wrap_config = McoreModuleWrapperConfig(
        is_value_model=False,
        wrap_with_ddp=False,
        use_distributed_optimizer=False,
        use_layer_wise_distributed_optimizer=False,
        use_megatron_fsdp=False,
    )

    log("Building Megatron module (shape only)")
    module, _tf_config = make_megatron_module(
        wrap_config=wrap_config,
        tf_config=None,
        hf_config=hf_config,
        bridge=bridge,
        provider=provider,
        override_model_config={},
        override_ddp_config={},
        peft_cls=None,
        peft_config=None,
    )

    log("Loading HF weights via Apertus1p5Bridge (the one real, slow read -- happens once here)")
    bridge.load_hf_weights(module, hf_model_path, allowed_mismatched_params=[])

    dist.barrier()
    if dist.get_rank() == 0:
        os.makedirs(output_path, exist_ok=True)
    dist.barrier()

    log(f"Saving Megatron dist-checkpoint to {output_path}")
    sharded_state_dict = unwrap_model(module[0]).sharded_state_dict()
    dist_checkpointing.save(sharded_state_dict, output_path, sharded_strategy=None, async_sharded_save=False)
    dist.barrier()
    log("Save complete.")

    if run_roundtrip_test:
        # A real (non-tautological) check: reloading into the SAME live tensors without first
        # corrupting them would trivially "match" even if the save were a no-op, since
        # ShardedTensor wraps the existing buffers by reference. Save a CPU clone of a sample of
        # tensors, zero the live buffers, reload from disk, and confirm the reload actually
        # restored the original values -- proves the round trip, not just that memory is memory.
        raw_model = unwrap_model(module[0])
        sample_names = []
        for name, tensor in raw_model.state_dict().items():
            if torch.is_tensor(tensor) and tensor.numel() > 0:
                sample_names.append(name)
        # First, last, and every ~50th param in between -- enough spread to catch a systematic
        # bug (e.g. one layer type mismapped) without cloning the whole 70B-param model.
        sample_names = sample_names[::50] or sample_names[:1]
        log(f"Round-trip self-test: sampling {len(sample_names)} of {len(raw_model.state_dict())} tensors")

        originals = {}
        for name in sample_names:
            tensor = raw_model.state_dict()[name]
            originals[name] = tensor.detach().clone().cpu()
            tensor.data.zero_()

        reload_ssd = raw_model.sharded_state_dict()
        dist_checkpointing.load(reload_ssd, output_path, strict=StrictHandling.ASSUME_OK_UNEXPECTED)

        mismatches = 0
        for name, original in originals.items():
            reloaded = raw_model.state_dict()[name].detach().cpu()
            if reloaded.shape != original.shape or not torch.equal(reloaded, original):
                mismatches += 1
                log(f"MISMATCH: {name} original.shape={original.shape} reloaded.shape={reloaded.shape}")
        log(f"Round-trip check: {len(originals) - mismatches}/{len(originals)} sampled tensors restored correctly")
        if mismatches:
            raise SystemExit(f"FATAL: round-trip self-test found {mismatches} mismatched tensors")
        log("Round-trip self-test PASSED (save -> zero -> reload restored the sampled tensors exactly).")

    dist.barrier()
    log("Done.")


if __name__ == "__main__":
    main()
