#!/usr/bin/env python3
"""UCCL CXI component/lifetime smoke: two exclusive nodes, four direct Slurm ranks per node.

All four GPUs must remain visible (--gpu-bind=none), rank placement is block.
Use --kill-on-bad-exit=1 and a step timeout; no concurrent same-UID UCCL jobs on these nodes.
This checks dispatch/combine and post-destroy tensor/RCCL lifetime, not training acceptance.
"""

import faulthandler
import os
from pathlib import Path
import socket
import subprocess
import tempfile
from datetime import timedelta


def run():
    """Exercise the real DeepEP wrapper with deterministic BF16 token/routing data."""
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    world = int(os.environ['SLURM_NTASKS'])
    assert world == 8 and 0 <= local_rank < 4
    nodes = subprocess.check_output(
        ['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']], text=True, timeout=30).splitlines()
    assert len(nodes) == 2, nodes
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(local_rank), LOCAL_WORLD_SIZE='4',
                      UCCL_EP_TRANSPORT='cxi', UCCL_SOCKET_IFNAME='hsn', UCCL_CXI_THREADING='safe')
    os.environ.setdefault('MASTER_ADDR', nodes[0])
    os.environ.setdefault('MASTER_PORT', str(20000 + int(os.environ['SLURM_JOB_ID'].split('_')[0]) % 20000))
    import torch
    import torch.distributed as dist
    from deep_ep import Buffer, Config

    assert torch.version.hip and torch.cuda.device_count() == 4
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    device = torch.device('cuda', local_rank)
    assert torch.cuda.get_device_properties(device).gcnArchName.startswith('gfx942')
    buffer = None
    try:
        dist.init_process_group(backend='nccl', init_method='env://', device_id=device,
                                timeout=timedelta(seconds=180))
        topology = [None] * world
        dist.all_gather_object(topology, (socket.gethostname(), local_rank))
        assert len({item[0] for item in topology[:4]}) == len({item[0] for item in topology[4:]}) == 1
        assert topology[0][0] != topology[4][0] and [item[1] for item in topology] == list(range(4)) * 2
        config = Config(64, 8, 512, 16, 512)
        tokens, hidden = 32, 512
        def margin(size):
            return ((int(size * 1.2) + 127) // 128) * 128
        buffer = Buffer(group=dist.group.WORLD,
                        num_nvl_bytes=margin(config.get_nvl_buffer_size_hint(hidden * 2, world)),
                        num_rdma_bytes=margin(config.get_rdma_buffer_size_hint(hidden * 2, world)),
                        num_qps_per_rank=64, low_latency_mode=False, explicitly_destroy=True, is_intranode=False)
        assert buffer.runtime.get_num_rdma_ranks() == 2
        stream = buffer.get_comm_stream()
        ids = torch.arange(tokens, device=device, dtype=torch.int64)
        signs = (1 - 2 * (torch.arange(hidden, device=device) % 2)).to(torch.bfloat16)
        def values(source):
            return (((source * tokens + ids + 1).to(torch.bfloat16) / 32)[:, None] * signs[None, :]).contiguous()
        def routing(source):
            first = (ids + source) % world
            return torch.stack((first, (first + 4) % world), dim=1).contiguous()
        x = values(rank)
        indices = routing(rank)
        weights = torch.full((tokens, 2), 0.5, device=device, dtype=torch.float32)
        per_rank, per_node, per_expert, membership, _ = buffer.get_dispatch_layout(indices, world)
        recv, recv_indices, recv_weights, counts, handle, _ = buffer.dispatch(
            x=x, num_tokens_per_rank=per_rank, num_tokens_per_rdma_rank=per_node,
            is_token_in_rank=membership, num_tokens_per_expert=per_expert,
            topk_idx=indices, topk_weights=weights, config=config, async_finish=False)
        reference = torch.cat([values(source)[routing(source).eq(rank).any(dim=1)] for source in range(world)])
        assert counts == [reference.shape[0]]
        torch.testing.assert_close(recv[recv[:, 0].argsort()], reference[reference[:, 0].argsort()], rtol=0, atol=0)
        combined, combined_weights, _ = buffer.combine(
            x=recv * 0.5, handle=handle, topk_weights=recv_weights, config=config, async_finish=False)
        torch.testing.assert_close(combined, x, rtol=0, atol=0)
        torch.testing.assert_close(combined_weights, weights, rtol=0, atol=0)
        # Explicitly keep allocator-recorded storage alive beyond Buffer.destroy().
        held = torch.ones(4096, device=device)
        held.record_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        buffer.destroy()
        buffer = None
        del held, handle, recv, recv_indices, recv_weights, combined, combined_weights
        # The borrowed stream must still be usable and its allocator events valid.
        with torch.cuda.stream(stream):
            after = torch.full((1,), rank + 1, device=device, dtype=torch.int32)
        torch.cuda.current_stream().wait_stream(stream)
        dist.all_reduce(after)
        torch.testing.assert_close(after, torch.tensor([36], device=device, dtype=torch.int32), rtol=0, atol=0)
        torch.cuda.synchronize()
        dist.barrier()
        print(f'PASS rank={rank}: UCCL CXI dispatch/combine exact, post-destroy recorded storage/stream/RCCL valid', flush=True)
    finally:
        try:
            if buffer is not None:
                buffer.destroy()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def main():
    """Bound native hangs and keep every rank's writable compiler cache private."""
    faulthandler.enable()
    faulthandler.dump_traceback_later(600, exit=True)
    with tempfile.TemporaryDirectory(prefix='apertus2-uccl-lifetime-') as private:
        root = Path(private)
        for variable in ('TMPDIR', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR',
                         'TORCHINDUCTOR_CACHE_DIR', 'AITER_JIT_DIR', 'MIOPEN_USER_DB_PATH', 'MIOPEN_CUSTOM_CACHE_DIR'):
            path = root / variable.lower()
            path.mkdir()
            os.environ[variable] = str(path)
        run()
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
