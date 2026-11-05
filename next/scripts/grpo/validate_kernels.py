import json, time
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as m

torch.cuda.set_device(1)
torch.manual_seed(42)

def error(actual, expected):
    a, b = actual.float(), expected.float()
    return float((a-b).square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-8))

results=[]
for length in (511, 4096):
    shape=(1,length,16,128)
    data=[torch.randn(shape,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    data += [-torch.rand((1,length,16),device='cuda'),torch.rand((1,length,16),device='cuda',dtype=torch.bfloat16)]
    weight=torch.randn(shape,device='cuda',dtype=torch.bfloat16)
    records=[]
    for function in (m.torch_chunk_gated_delta_rule,m.chunk_gated_delta_rule):
        values=[v.detach().clone().requires_grad_() for v in data]
        torch.cuda.synchronize(); started=time.monotonic()
        output,_=function(*values, use_qk_l2norm_in_kernel=True)
        (output*weight).float().mean().backward()
        torch.cuda.synchronize()
        records.append((output.detach(),[v.grad.detach() for v in values],time.monotonic()-started))
    forward=error(records[1][0], records[0][0])
    gradients=[error(a,b) for a,b in zip(records[1][1],records[0][1])]
    assert forward<.02 and max(gradients)<.05, (forward,gradients)
    results.append({'length':length,'relative_output_rmse':forward,'relative_gradient_rmse':gradients,
                    'torch_seconds':records[0][2],'fla_seconds_including_jit':records[1][2]})
ref=m.Qwen3_5RMSNormGated(128,eps=1e-6).to(device='cuda',dtype=torch.bfloat16)
fast=m.FusedRMSNormGated(128,eps=1e-6,activation='silu').to(device='cuda',dtype=torch.bfloat16)
fast.load_state_dict(ref.state_dict())
x=torch.randn(513,128,device='cuda',dtype=torch.bfloat16)
z=torch.randn_like(x)
a,b=[],[]
for mod,record in ((ref,a),(fast,b)):
    xx,zz=x.clone().requires_grad_(),z.clone().requires_grad_()
    y=mod(xx,zz); y.float().square().mean().backward()
    record.extend((y.detach(),xx.grad,zz.grad,mod.weight.grad))
errors=[error(v,u) for v,u in zip(b,a)]
assert max(errors)<.05,errors
print(json.dumps({'status':'passed','gated_delta_rule':results,'rmsnorm_relative_errors':errors}),flush=True)
