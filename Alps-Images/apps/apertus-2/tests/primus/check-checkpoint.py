#!/usr/bin/env python3
"""Strict actual-ownership, optimizer/RNG and finite-update acceptance for tiny tests."""
import json
import math
import os
from pathlib import Path
import re
import sys
import torch

root=Path(sys.argv[1]);rank=sys.argv[2];stage=sys.argv[3]
world=int(os.environ['WORLD_SIZE'])
ranks=range(world) if rank=='all' else [int(rank)]
stages=(0,1) if stage=='all' else [int(stage)]
manifest=json.loads(Path('/opt/alps/apertus2-primus/source-identity.json').read_text())
for index in stages:
    iteration=(2,4)[index]
    if index:
        # Native reload announcements belong to rank 0, not necessarily the
        # rank reporting training metrics (rank 7 for the two-node EP8 test).
        reload_text=(root/'ranks'/'0-stage1.log').read_text()
        assert re.search(r'successfully loaded checkpoint from .*?at iteration\s+2\b',reload_text), 'Missing rank-0 native reload announcement'
    reporting=[];owners=[]
    for current in ranks:
        record=json.loads((root/'ranks'/f'{current}-stage{index}.json').read_text())
        assert record['rank']==current and record['world']==world and record['iteration']==iteration
        assert record['loaded_megatron']
        if index:
            restored = record['restore']
            assert len(restored['checkpoint']) == 1, restored
            checkpoint = restored['checkpoint'][0]
            assert checkpoint['result'][0] == 2, restored
            assert not any(checkpoint[key] for key in ('no_load_optim', 'no_load_rng', 'finetune'))
            assert restored['optimizer'] and {'cpu', 'cuda'} <= set(restored['rng']), restored
        assert all(manifest['files'][name]==sha for name,sha in record['loaded_megatron'].items())
        model=Path(record['model_checkpoint']);optimizer=Path(record['optimizer_checkpoint'])
        assert model.is_relative_to(root/'checkpoints'/f'iter_{iteration:07d}')
        assert optimizer.parent==model.parent and optimizer.name==f'layer_wise_optimizer_{record["dp_rank"]}.pt'
        if index:
            relative=model.relative_to(root/'checkpoints'/f'iter_{iteration:07d}')
            loaded_optimizer=(root/'checkpoints'/'iter_0000002'/relative).with_name(optimizer.name)
            assert [Path(name).resolve() for name in record['restore']['optimizer']]==[loaded_optimizer.resolve()]
        assert optimizer.stat().st_size>0
        state=torch.load(model,map_location='cpu',weights_only=False)
        assert state['iteration']==iteration
        assert all(state.get(key) is not None for key in ('model','optimizer','opt_param_scheduler','rng_state'))
        assert state['args'].consumed_train_samples==iteration*8
        sidecar=torch.load(optimizer,map_location='cpu',weights_only=False)
        assert isinstance(sidecar,(dict,list)) and sidecar
        owners.append(str(optimizer))
        text=(root/'ranks'/f'{current}-stage{index}.log').read_text()
        seen=[]
        for line in text.splitlines():
            match=re.search(r'\biteration\s+(\d+)\s*/\s*4\s*\|',line)
            if not match: continue
            seen.append(int(match[1]))
            for label in ('lm loss','grad norm'):
                value=re.search(re.escape(label)+r':\s*([^\s|]+)',line)
                assert value and math.isfinite(float(value[1])),line
            for label in ('number of skipped iterations','number of nan iterations'):
                value=re.search(re.escape(label)+r':\s*(\d+)',line)
                assert value and int(value[1])==0,line
        if seen:
            assert seen==([1,2] if index==0 else [3,4]),seen
            reporting.append(current)
    if rank=='all':
        assert reporting, 'No rank reported finite updates'
        assert len(set(owners))==world, 'Duplicate optimizer ownership'
        inventory={str(p) for p in (root/'checkpoints'/f'iter_{iteration:07d}').rglob('layer_wise_optimizer_*.pt')}
        assert inventory==set(owners), 'Optimizer inventory/ownership mismatch'
        print(f'All {world} ranks passed stage{index}: finite updates, source identity and checkpoint ownership')
assert (root/'checkpoints/latest_checkpointed_iteration.txt').read_text().strip()==str((2,4)[stages[-1]])
