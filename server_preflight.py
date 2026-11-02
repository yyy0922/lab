"""Read-only portable server readiness check; no training or test evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import torch
from data import ROOT

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--device',choices=['cpu','cuda','auto'],default='auto'); args=parser.parse_args()
    required=['model.py','data.py','experiment.py','reference/map.py','reference/run_lta.py','reference/Static Analysis/infrastructure/map.py','data/prepared.json']
    missing=[x for x in required if not (ROOT/x).exists()]
    if missing: raise FileNotFoundError(missing)
    path=ROOT/'data'/'prepared.json'; prepared=json.loads(path.read_text(encoding='utf-8'))
    for r in prepared['records']:
        cache=ROOT/'data'/'methods'/(r['sha256']+'.json')
        if not cache.exists(): missing.append(str(cache))
    if missing: raise FileNotFoundError(missing)
    groups=[{prepared['records'][i]['feature_group'] for i in prepared[k]} for k in ['train','val','test']]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]),'Cross-split feature duplicate'
    cuda=torch.cuda.is_available()
    if args.device=='cuda' and not cuda: raise RuntimeError('CUDA unavailable. Install a CUDA-enabled PyTorch build compatible with the server GPU/driver.')
    device='cuda' if cuda and args.device!='cpu' else 'cpu'
    probe=torch.ones(2,device=device); assert float(probe.sum())==2
    print(json.dumps(dict(status='ready',root=str(ROOT),python=platform.python_version(),torch=torch.__version__,device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,samples=len(prepared['records']),split={k:len(prepared[k]) for k in ['train','val','test']},features=len(prepared['vocabulary']),prepared_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw_apks_required_for_cached_training=False,official_fused_mamba_kernel=False),indent=2))
