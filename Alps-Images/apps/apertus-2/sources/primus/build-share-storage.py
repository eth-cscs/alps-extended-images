#!/usr/bin/env python3
"""Rebuild the reference CPU storage-sharing extension against this base's Torch ABI."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
from torch.utils.cpp_extension import load_inline

source = Path('/opt/megatron-lm/megatron/core/tensor_parallel/random.py')
tree = ast.parse(source.read_text())
value = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == '_SHARE_STORAGE_SRC' for target in node.targets))
root = Path('/opt/alps/apertus2-primus/extensions'); root.mkdir(exist_ok=True)
build = root/'build-share-storage'; build.mkdir(exist_ok=True)
module = load_inline(name='share_storage_ext', cpp_sources=value, functions=['share_storage'],
                     build_directory=str(build), verbose=True)
shutil.copyfile(module.__file__, root/'share_storage_ext.so')
(root/'share-storage-source.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    cpp_sha256=hashlib.sha256(value.encode()).hexdigest(),scope='Exact reference C++ source rebuilt for the image Torch ABI.'),indent=2)+'\n')
shutil.rmtree(build)
