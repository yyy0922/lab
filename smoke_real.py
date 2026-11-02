"""Complete one real APK optimization step; no sampled methods or positions."""
import json
import time
import torch
from data import ROOT, apk_methods, dump
from model import Detector
from experiment import apk_forward, tensors

torch.set_num_threads(4)
torch.manual_seed(42)
rows=json.loads((ROOT/'data'/'manifest.json').read_text(encoding='utf-8'))
r=next(r for r in rows if r.get('status')=='ok')
raw=json.loads((ROOT/'data'/'methods'/(r['sha256']+'.json')).read_text(encoding='utf-8'))
# Smoke only: this vocabulary belongs to a single forward/backward check, never
# reused for the train/validation/test experiment or GA.
vocab={g:i+1 for i,g in enumerate(raw['grams'])}
methods=apk_methods(r,vocab)
net=Detector(len(vocab),'C6')
optimizer=torch.optim.AdamW(net.parameters(),lr=.001)
start=time.perf_counter()
z,h,replay,total=apk_forward(net,methods,1024,True)
loss=torch.nn.functional.cross_entropy(z,torch.tensor([r['label']]))
loss.backward()
for idx,rng in replay:
    torch.set_rng_state(rng)
    x,g,l,w=tensors(methods,idx)
    (net.methods(x,g,l)*w[:,None]/total).sum(0,keepdim=True).backward(h.grad)
grad=float(net.long.block.offset.weight.grad.abs().sum())
assert torch.isfinite(loss) and grad>0
optimizer.step()
report=dict(sha256=r['sha256'],label=r['label'],width=64,state=16,usable_methods=total,unique_methods=len(methods),positions=r['positions'],loss=float(loss.detach()),group_offset_gradient_l1=grad,seconds=time.perf_counter()-start,scope='one full real APK training step; not evaluation or evidence of accuracy')
dump(ROOT/'smoke_real.json',report)
print(json.dumps(report,indent=2),flush=True)
