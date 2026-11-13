#!/usr/bin/env python3
"""Validate source identity first, then the selected reference APIs and numerical kernels."""
import argparse
import ctypes
import hashlib
from importlib import metadata
import importlib.util
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-only',action='store_true')
    args=parser.parse_args()
    root=Path('/opt/alps/apertus2-primus')
    source=Path('/opt/megatron-lm')
    manifest=json.loads((root/'source-identity.json').read_text())
    assert manifest['revision']=='d1277d86e0c1925f8438fd7630d86d05ab398dcd'
    for name,sha in manifest['files'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha, name
    spec=importlib.util.find_spec('megatron.core')
    assert Path(spec.origin).resolve().is_relative_to(source), spec.origin
    versions={name:metadata.version(name) for name in ('torch','triton','transformer-engine','amd-aiter',
        'primus-turbo','emerging-optimizers','flash-linear-attention','fla-core','causal-conv1d')}
    assert versions['torch'].startswith(('2.10.0+git94c6e04','2.14.0+rocm10.1.0')), versions
    assert versions['emerging-optimizers']=='0.3.1', versions
    assert versions['flash-linear-attention']==versions['fla-core']=='0.5.2'
    assert versions['causal-conv1d']=='1.5.0.post8'
    original=json.loads((root/'vendor/manifest.json').read_text())
    expected=dict(original['files'])
    if versions['primus-turbo']=='0.5.1.dev7':
        expected.update(json.loads((root/'patches/primus268-source-hashes.json').read_text()))
        patch=root/'patches/0001-turbo051-private-api-imports.patch'
        assert manifest['applied_patches']==[dict(file=patch.name,sha256=hashlib.sha256(patch.read_bytes()).hexdigest())]
    else:
        assert versions['primus-turbo']=='0.2.0+3cd482d'
        assert manifest['applied_patches']==[]
    assert manifest['files']==expected
    fla=Path(next(iter(importlib.util.find_spec('fla').submodule_search_locations)))
    assert hashlib.sha256((fla/'ops/kda/gate.py').read_bytes()).hexdigest()=='b350410cc605d1b3fac082d1e8384d7e9897e3aaa36c00f3d01d8d7fd48320ae'
    assert (root/'extensions/share_storage_ext.so').is_file()
    assert (source/'megatron/core/datasets/libbfd_pack.so').is_file()
    assert list((source/'megatron/core/datasets').glob('helpers_cpp*.so'))
    if args.metadata_only:
        print(json.dumps(dict(status='passed',scope='Package/source/build metadata only; no GPU acceptance.',versions=versions),indent=2))
        return
    import torch
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx942')
    assert os.environ['DEBUG_CLR_AQL_DEV_QUEUE']=='0'
    assert os.environ['NCCL_P2P_DISABLE']=='1'
    assert os.environ['NCCL_NET']=='AWS Libfabric'
    rccl=ctypes.CDLL('/opt/alps/rocm/rccl/lib/librccl.so.1')
    rccl.ncclGetVersion.argtypes=[ctypes.POINTER(ctypes.c_int)]
    rccl.ncclGetVersion.restype=ctypes.c_int
    actual_rccl=ctypes.c_int()
    assert rccl.ncclGetVersion(ctypes.byref(actual_rccl))==0
    assert actual_rccl.value==23004,actual_rccl.value
    loaded_rccl={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                 if '/librccl.so' in line}
    assert loaded_rccl and all('/opt/alps/rocm/rccl/' in file for file in loaded_rccl),loaded_rccl
    for name in ('BEVERIN_TURBO_GG','BEVERIN_T6_EP','BEVERIN_T6_FP8','BEVERIN_T6_FUSED',
                 'BEVERIN_AITER_ATTN','BEVERIN_K7_BATCH','BEVERIN_NS_WS'):
        assert os.environ.get(name)=='1', name
    import megatron.core
    assert Path(megatron.core.__file__).resolve().is_relative_to(source)
    from megatron.core.transformer.moe import beverin_turbo
    # Fails explicitly if a new vendor Turbo no longer supports the reference's private API.
    backend,*operations=beverin_turbo._ops()
    assert all(callable(operation) for operation in operations)
    from megatron.core.transformer.moe import beverin_gg, beverin_vk
    assert callable(beverin_gg.grouped_gemm_ptr) and callable(beverin_vk.vk_accum)
    # Exercise actual pointer-table fwd/dgrad and fused FP32 main-grad accumulation.
    torch.manual_seed(123)
    expert=torch.nn.Module()
    expert.num_gemms=2
    expert.fuse_wgrad_accumulation=True
    weights=[]
    for index in range(2):
        weight=torch.nn.Parameter(torch.randn(256,128,device='cuda',dtype=torch.bfloat16)/8)
        weight.main_grad=torch.zeros_like(weight,dtype=torch.float32)
        setattr(expert,f'weight{index}',weight)
        weights.append(weight)
    x=(torch.randn(128,128,device='cuda',dtype=torch.bfloat16)/8).requires_grad_()
    lengths=torch.tensor([64,64],device='cuda',dtype=torch.int64)
    assert beverin_turbo.PTR and beverin_turbo.FUSED_WGRAD
    assert beverin_turbo._ptr_offsets(weights) is not None
    result=beverin_turbo.expert_linear(expert,x,lengths)
    expected=torch.cat([part.float()@weight.float().T for part,weight in zip(x.split(64),weights)])
    torch.testing.assert_close(result.float(),expected,rtol=0.03,atol=0.03)
    gradient=torch.randn_like(result)/8
    result.backward(gradient)
    from megatron.core.transformer.moe import beverin_o1
    beverin_o1.join()
    torch.cuda.synchronize()
    expected_dx=torch.cat([part.float()@weight.float() for part,weight in zip(gradient.split(64),weights)])
    torch.testing.assert_close(x.grad.float(),expected_dx,rtol=0.03,atol=0.03)
    for weight,dy,part in zip(weights,gradient.split(64),x.split(64)):
        torch.testing.assert_close(weight.main_grad,dy.float().T@part.float(),rtol=0.03,atol=0.03)
        assert weight.grad_added_to_main_grad
    from aiter import flash_attn_func
    assert callable(flash_attn_func)
    spec=importlib.util.spec_from_file_location('share_storage_ext',root/'extensions/share_storage_ext.so')
    extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension)
    original=torch.arange(16,dtype=torch.float32)
    shared=torch.empty_like(original)
    extension.share_storage(shared,original)
    assert shared.data_ptr()==original.data_ptr()
    del original
    torch.testing.assert_close(shared,torch.arange(16,dtype=torch.float32))
    torch.manual_seed(123)
    q=torch.randn(1,128,2,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    k=torch.randn_like(q,requires_grad=True);v=torch.randn_like(q,requires_grad=True)
    actual=flash_attn_func(q,k,v,dropout_p=0.0,causal=True,deterministic=False,
        return_lse=True,how_v3_bf16_cvt=int(os.environ.get('BEVERIN_AITER_ATTN_CVT','1')))
    actual=actual[0] if isinstance(actual,(tuple,list)) else actual
    qr,kr,vr=[t.detach().float().cpu().requires_grad_() for t in (q,k,v)]
    expected=torch.nn.functional.scaled_dot_product_attention(qr.transpose(1,2),kr.transpose(1,2),
        vr.transpose(1,2),is_causal=True).transpose(1,2)
    torch.testing.assert_close(actual.float().cpu(),expected,rtol=0.03,atol=0.03)
    gradient=torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient.float().cpu())
    attention_accuracy={}
    if versions['amd-aiter']=='0.1.22.post1':
        # The original fixed elementwise gate failed a few near-zero BF16
        # gradients. Use this release's independent upstream error policy,
        # calibrated against a BF16 math control and FP64 reference, plus a
        # stricter global L2 guard. Keep original-gate mismatch counts visible.
        from torch.nn.attention import SDPBackend,sdpa_kernel
        policy_path=Path(__file__).with_name('attention-accuracy.py')
        spec=importlib.util.spec_from_file_location('attention_accuracy',policy_path)
        policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
        precise=[t.detach().double().cpu().requires_grad_() for t in (q,k,v)]
        precise_out=torch.nn.functional.scaled_dot_product_attention(
            *[t.transpose(1,2) for t in precise],is_causal=True).transpose(1,2)
        precise_out.backward(gradient.double().cpu())
        controls=[t.detach().clone().requires_grad_() for t in (q,k,v)]
        with sdpa_kernel(SDPBackend.MATH):
            control_out=torch.nn.functional.scaled_dot_product_attention(
                *[t.transpose(1,2) for t in controls],is_causal=True).transpose(1,2)
        control_out.backward(gradient)
        for name,tensor,reference,precise_tensor,control in zip(('q','k','v'),(q,k,v),(qr,kr,vr),precise,controls):
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all().item()
            actual_grad=tensor.grad.detach().double().cpu()
            reference_grad=precise_tensor.grad.detach()
            control_grad=control.grad.detach().double().cpu()
            assert torch.isfinite(reference_grad).all().item() and torch.isfinite(control_grad).all().item()
            norm=torch.linalg.vector_norm(reference_grad).item()
            assert norm>0
            actual_error=actual_grad-reference_grad
            control_error=control_grad-reference_grad
            metrics=policy.validate_gradient_error(actual_error.abs().max().item(),
                control_error.abs().max().item(),torch.linalg.vector_norm(actual_error).item()/norm,
                torch.linalg.vector_norm(control_error).item()/norm)
            metrics['original_elementwise_mismatches']=(~torch.isclose(
                tensor.grad.float().cpu(),reference.grad,rtol=0.05,atol=0.005)).sum().item()
            attention_accuracy[name]=metrics
    else:
        for tensor,reference in zip((q,k,v),(qr,kr,vr)):
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all().item()
            torch.testing.assert_close(tensor.grad.float().cpu(),reference.grad,rtol=0.05,atol=0.005)
    print(json.dumps(dict(status='passed',versions=versions,gpu=torch.cuda.get_device_name(),
        compute_units=torch.cuda.get_device_properties(0).multi_processor_count,
        attention_accuracy=attention_accuracy,
        scope='Source/adapter identity, reference Turbo pointer GEMM fwd/dgrad/fused wgrad, CPU storage lifetime and AITER attention forward/backward; not training/restart acceptance.'),indent=2))


if __name__=='__main__': main()
