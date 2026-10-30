#!/usr/bin/env python3
"""RCCL and UCCL DeepEP HT correctness: exclusive two nodes, four Slurm ranks/node.

Run directly with srun (not torchrun); expose all four node GPUs to every rank
(--gpu-bind=none). MASTER_ADDR/PORT are derived from Slurm unless already set.
The pinned UCCL POSIX-shm barriers are UID-scoped, so same-UID concurrent HT
jobs must not share these nodes. No Ray, serving stack, or benchmark loop.
"""

import faulthandler
import os
import socket
import subprocess
import sys
import tempfile
from datetime import timedelta
from pathlib import Path


def configure_slurm():
    if len(sys.argv) != 1:
        raise ValueError("uccl-smoke.py takes no arguments; use the Slurm environment")
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world = int(os.environ["SLURM_NTASKS"])
    if world != 8 or not 0 <= rank < world or not 0 <= local_rank < 4:
        raise ValueError("UCCL HT smoke requires exactly 8 ranks: 2 nodes x 4 GPUs")
    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(local_rank),
        LOCAL_WORLD_SIZE="4",
        UCCL_EP_TRANSPORT="cxi", UCCL_SOCKET_IFNAME="hsn", UCCL_CXI_THREADING="safe",
        TORCH_NCCL_ASYNC_ERROR_HANDLING="1", TORCH_NCCL_BLOCKING_WAIT="1",
    )
    if "MASTER_ADDR" not in os.environ:
        nodes = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
            text=True, timeout=30,
        ).splitlines()
        if len(nodes) != 2:
            raise ValueError(f"Expected an allocation of two nodes, got {nodes}")
        os.environ["MASTER_ADDR"] = nodes[0]
    if "MASTER_PORT" not in os.environ:
        job_id = int(os.environ["SLURM_JOB_ID"].split("_", 1)[0])
        os.environ["MASTER_PORT"] = str(20000 + job_id % 20000)
    return rank, local_rank, world


def run(rank, local_rank, world):
    # Import only after each rank has its private cache paths and Slurm identity.
    import torch
    import torch.distributed as dist

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("UCCL smoke requires ROCm PyTorch with available HIP GPUs")
    if torch.cuda.device_count() != 4:
        raise RuntimeError("Expose all four node GPUs to each rank with --gpu-bind=none")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
    if arch != "gfx942":
        raise RuntimeError(f"Expected gfx942, got {arch}")
    torch.manual_seed(1234 + rank)
    torch.set_num_threads(1)

    # Exercise the public DeepEP API consumed by Megatron's fused dispatcher.
    from deep_ep import Buffer, Config

    buffer = None
    try:
        # "nccl" selects RCCL in ROCm PyTorch. Preserve the image's canonical
        # aws-ofi-rccl plugin and LD_LIBRARY_PATH; UCCL is only the EP transport.
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world,
            device_id=device, timeout=timedelta(seconds=180),
        )
        group = dist.group.WORLD
        topology = [None] * world
        dist.all_gather_object(topology, (socket.gethostname(), local_rank), group=group)
        hosts = [item[0] for item in topology]
        if not (
            len(set(hosts[:4])) == 1 and len(set(hosts[4:])) == 1
            and hosts[0] != hosts[4]
            and [item[1] for item in topology] == list(range(4)) * 2
        ):
            raise RuntimeError(f"Expected block rank order on two 4-GPU nodes: {topology}")

        values = torch.tensor([rank + 1, (rank + 1) ** 2], dtype=torch.float32, device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM, group=group)
        torch.testing.assert_close(
            values, torch.tensor([36.0, 204.0], device=device), rtol=0, atol=0,
        )

        # Match the pinned HIP internode benchmark's conservative HT config.
        # Buffer hints depend on the compiled four-peer gfx942 configuration.
        config = Config(64, 8, 512, 16, 512)
        tokens, hidden, experts = 32, 512, world

        def size_with_margin(size):
            return ((int(size * 1.2) + 127) // 128) * 128

        nvl_bytes = size_with_margin(config.get_nvl_buffer_size_hint(hidden * 2, world))
        rdma_bytes = size_with_margin(config.get_rdma_buffer_size_hint(hidden * 2, world))
        buffer = Buffer(
            group=group, num_nvl_bytes=nvl_bytes, num_rdma_bytes=rdma_bytes,
            low_latency_mode=False, num_qps_per_rank=64, explicitly_destroy=True,
            is_intranode=False,
        )
        if buffer.runtime.get_num_rdma_ranks() != 2:
            raise RuntimeError("Compiled UCCL HT topology is not two RDMA nodes")

        token_ids = torch.arange(tokens, device=device, dtype=torch.int64)
        signs = (1 - 2 * (torch.arange(hidden, device=device) % 2)).to(torch.bfloat16)

        def source_tokens(source_rank):
            # Unique, exactly representable bf16 values identify source+token;
            # alternating signs also detect corruption across hidden channels.
            ids = (source_rank * tokens + token_ids + 1).to(torch.bfloat16) / 32
            return (ids[:, None] * signs[None, :]).contiguous()

        def source_routing(source_rank):
            first = (token_ids + source_rank) % world
            # Every token visits one expert on EACH node, exercising HT RDMA
            # and intra-node IPC; one expert per rank and no duplicate choices.
            return torch.stack((first, (first + 4) % world), dim=1).contiguous()

        x = source_tokens(rank)
        topk_idx = source_routing(rank)
        topk_weights = torch.full((tokens, 2), 0.5, dtype=torch.float32, device=device)
        per_rank, per_node, per_expert, membership, _ = buffer.get_dispatch_layout(
            topk_idx, experts,
        )
        expected_membership = topk_idx[:, :, None].eq(
            torch.arange(world, device=device)[None, None, :]
        ).any(dim=1)
        torch.testing.assert_close(membership, expected_membership, rtol=0, atol=0)
        torch.testing.assert_close(
            per_rank, expected_membership.sum(dim=0).to(torch.int32), rtol=0, atol=0,
        )
        torch.testing.assert_close(per_expert, per_rank, rtol=0, atol=0)
        torch.testing.assert_close(
            per_node, torch.full((2,), tokens, dtype=torch.int32, device=device),
            rtol=0, atol=0,
        )
        recv_x, recv_idx, recv_weights, counts, handle, _ = buffer.dispatch(
            x=x, num_tokens_per_rank=per_rank, num_tokens_per_rdma_rank=per_node,
            is_token_in_rank=membership, num_tokens_per_expert=per_expert,
            topk_idx=topk_idx, topk_weights=topk_weights, config=config,
            async_finish=False,
        )
        reference = torch.cat([
            source_tokens(source)[source_routing(source).eq(rank).any(dim=1)]
            for source in range(world)
        ])
        if recv_x.dtype != torch.bfloat16 or recv_x.shape != reference.shape:
            raise AssertionError(f"Unexpected bf16 dispatch: {recv_x.dtype} {recv_x.shape}")
        if counts != [reference.shape[0]]:
            raise AssertionError(f"Incorrect expert receive counts: {counts}")
        # Receive ordering is not part of the transport contract. Token values
        # are unique, so sorting compares every element without assuming order.
        torch.testing.assert_close(
            recv_x[recv_x[:, 0].argsort()], reference[reference[:, 0].argsort()],
            rtol=0, atol=0,
        )
        if not bool(((recv_idx == -1) | (recv_idx == 0)).all()):
            raise AssertionError("Dispatch returned an invalid local expert index")
        if not bool((recv_idx.eq(0).sum(dim=1) == 1).all()):
            raise AssertionError("Each received token must select exactly one local expert")
        torch.testing.assert_close(
            recv_weights[recv_idx == 0],
            torch.full_like(recv_weights[recv_idx == 0], 0.5), rtol=0, atol=0,
        )

        # HT combine ADDS outputs; unlike the LL API it does not apply routing
        # weights. Apply each identity expert's 1/2 weight before combining.
        combined, combined_weights, _ = buffer.combine(
            x=recv_x * 0.5, handle=handle, topk_weights=recv_weights,
            config=config, async_finish=False,
        )
        torch.testing.assert_close(combined, x, rtol=0, atol=0)
        torch.testing.assert_close(combined_weights, topk_weights, rtol=0, atol=0)
        torch.cuda.synchronize(device)
        # A rank-local assertion cannot be hidden by success on rank zero.
        passed = torch.ones((), dtype=torch.int32, device=device)
        dist.all_reduce(passed, op=dist.ReduceOp.SUM, group=group)
        if passed.item() != world:
            raise AssertionError("Not all eight ranks completed numerical validation")
        dist.barrier(group=group)
        buffer.destroy()
        buffer = None
        # Release recorded tensor storage AFTER destroying the EP buffer:
        # its stream must still be valid for the caching allocator's events.
        del (handle, recv_x, recv_idx, recv_weights, combined, combined_weights,
             x, topk_idx, topk_weights, per_rank, per_node, per_expert, membership)
        torch.cuda.synchronize(device)
        # The EP destructor must not corrupt subsequent GPU work or RCCL.
        after_destroy = torch.full((1,), rank + 1, dtype=torch.int32, device=device)
        dist.all_reduce(after_destroy, op=dist.ReduceOp.SUM, group=group)
        torch.testing.assert_close(
            after_destroy, torch.tensor([36], dtype=torch.int32, device=device),
            rtol=0, atol=0,
        )
        dist.barrier(group=group)
        print(
            f"PASS rank={rank} local_rank={local_rank} host={socket.gethostname()} "
            f"HIP={torch.version.hip} GPU={arch}: RCCL sum=[36,204], "
            f"UCCL HT bf16 dispatch={reference.shape[0]} tokens + identity combine exact; "
            "recorded tensor release and post-destroy RCCL passed",
            flush=True,
        )
    finally:
        try:
            if buffer is not None:
                buffer.destroy()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def main():
    faulthandler.enable()
    # Also bound GPU/native-code hangs, which a Python signal handler alone
    # cannot interrupt. Slurm should use --kill-on-bad-exit=1 for peer failure.
    faulthandler.dump_traceback_later(600, exit=True)
    rank, local_rank, world = configure_slurm()
    with tempfile.TemporaryDirectory(prefix=f"apertus2-uccl-rank{rank}-") as private:
        root = Path(private)
        for variable, subdir in (
            ("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"),
            ("TRITON_CACHE_DIR", "triton"), ("TORCH_EXTENSIONS_DIR", "torch-extensions"),
            ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
            ("MIOPEN_USER_DB_PATH", "miopen-db"), ("MIOPEN_CUSTOM_CACHE_DIR", "miopen-cache"),
        ):
            path = root / subdir
            path.mkdir()
            os.environ[variable] = str(path)
        run(rank, local_rank, world)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
