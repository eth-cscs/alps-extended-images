#!/usr/bin/env python3
"""Run unmodified image Megatron and capture actual checkpoint ownership afterward."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--record',type=Path,required=True)
parser.add_argument('--iteration',type=int,required=True)
args,training_arguments=parser.parse_known_args()
source=Path('/opt/megatron-lm')
sys.path.insert(0,str(source))
sys.argv=[str(source/'pretrain_gpt.py'),*training_arguments]
# Observe successful native restore calls without modifying the vendored source.
restore = {'checkpoint': [], 'optimizer': [], 'rng': []}
def observe(frame, event, value):
    if event != 'return' or frame.f_code.co_name not in ('load_checkpoint', 'load_state_dict_from_file', 'set_rng_state'):
        return
    file = frame.f_code.co_filename
    name = frame.f_code.co_name
    parent = frame.f_back
    in_restore = False
    while parent is not None:
        if parent.f_code.co_name == 'load_checkpoint' and parent.f_code.co_filename.endswith('/megatron/training/checkpointing.py'):
            in_restore = True
            break
        parent = parent.f_back
    if file.endswith('/megatron/training/checkpointing.py') and name == 'load_checkpoint':
        native = frame.f_locals.get('args')
        restore['checkpoint'].append(dict(result=value, no_load_optim=native.no_load_optim,
            no_load_rng=native.no_load_rng, finetune=native.finetune))
        sys.setprofile(None)
    elif in_restore and file.endswith('/megatron/core/optimizer/layer_wise_optimizer.py') and name == 'load_state_dict_from_file':
        restore['optimizer'].append(str(frame.f_locals.get('filename', frame.f_locals.get('path', 'native-sidecar'))))
    elif in_restore and name == 'set_rng_state' and ('/torch/random.py' in file or '/torch/cuda/random.py' in file):
        restore['rng'].append('cuda' if '/cuda/' in file else 'cpu')
if '--load' in training_arguments:
    sys.setprofile(observe)
try:
    runpy.run_path(sys.argv[0],run_name='__main__')
except SystemExit as exit:
    if exit.code not in (None, 0):
        raise
finally:
    sys.setprofile(None)
from megatron.training import get_args
from megatron.core import parallel_state as mpu
from megatron.training.checkpointing import get_checkpoint_name
import torch.distributed as dist
native=get_args()
model=Path(get_checkpoint_name(native.save,args.iteration)).resolve()
loaded={}
for name,module in tuple(sys.modules.items()):
    if name=='megatron' or name.startswith('megatron.'):
        file=getattr(module,'__file__',None)
        if file and Path(file).suffix=='.py':
            file=Path(file).resolve()
            assert file.is_relative_to(source), (name,str(file))
            loaded[str(file.relative_to(source))]=hashlib.sha256(file.read_bytes()).hexdigest()
record=dict(rank=dist.get_rank(),world=dist.get_world_size(),iteration=args.iteration,
    pp_rank=mpu.get_pipeline_model_parallel_rank(),dp_rank=mpu.get_data_parallel_rank(),
    ep_rank=mpu.get_expert_model_parallel_rank(),model_checkpoint=str(model),
    optimizer_checkpoint=str(model.with_name(f'layer_wise_optimizer_{mpu.get_data_parallel_rank()}.pt')),
    loaded_megatron=loaded,restore=restore)
args.record.write_text(json.dumps(record,indent=2)+'\n')
