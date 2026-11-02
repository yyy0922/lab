"""Frozen group split -> training-only GA -> APK-level training/evaluation."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from data import ROOT, dump, apk_methods
from model import Detector, OriginalLTA

def prepare(args):
    rows=json.loads((ROOT/'data'/'manifest.json').read_text(encoding='utf-8'))
    records=[r for r in rows if r.get('status')=='ok' and r['label'] is not None]
    source_root=ROOT/'data'/'apk_snapshot' if (ROOT/'data'/'apk_snapshot').exists() else ROOT/'apk'
    expected=sum(1 for folder in ['benign','malradar'] for _ in (source_root/folder).glob('*.apk'))
    if (ROOT/'data'/'snapshot_inventory.json').exists():
        inventory=json.loads((ROOT/'data'/'snapshot_inventory.json').read_text(encoding='utf-8'))
        expected=sum(r['status']=='copied' and r['snapshot'].replace('\\','/').split('/')[-2]!='new' for r in inventory)
    if len([r for r in rows if r['label'] is not None])!=expected:
        raise ValueError('Extraction incomplete or source snapshot changed; inspect manifest')
    failures=[r for r in rows if r['label'] is not None and r.get('status')!='ok']
    if failures and not args.allow_extraction_failures:
        raise ValueError('Extraction failures recorded; inspect them and explicitly allow exclusion')
    y=np.array([r['label'] for r in records]); groups=np.array([r['feature_group'] for r in records])
    trval,test=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(y,y,groups))
    tr,va=next(StratifiedGroupKFold(8,shuffle=True,random_state=43).split(trval,y[trval],groups[trval]))
    train,val=trval[tr],trval[va]
    assert all(not(set(groups[a])&set(groups[b])) for a,b in [(train,val),(train,test),(val,test)])
    # Vocabulary constructed from training APKs only, before GA selection.
    grams={r['sha256']:json.loads((ROOT/'data'/'methods'/(r['sha256']+'.json')).read_text(encoding='utf-8'))['grams'] for r in records}
    vocab=sorted({g for i in train for g in grams[records[i]['sha256']]})
    lookup={g:j for j,g in enumerate(vocab)}
    x=np.zeros((len(records),len(vocab)),dtype=np.float32)
    for i,r in enumerate(records):
        for g in grams[r['sha256']]:
            if g in lookup: x[i,lookup[g]]=1
    # Reuse LTA's exact GA operators and SVM fitness, with train-only input.
    spec=importlib.util.spec_from_file_location('lta_reference',ROOT/'reference'/'run_lta.py')
    sys.path.insert(0,str(ROOT/'reference'/'Static Analysis'))
    old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    selected,history=old.select_features(x[train],y[train],args.population,args.generations,42)
    if not selected.any(): raise ValueError('GA selected no features')
    result=dict(records=records,train=train.tolist(),val=val.tolist(),test=test.tolist(),vocabulary={g:i+1 for i,g in enumerate(np.asarray(vocab)[selected])},ga_history=history,ga_population=args.population,ga_generations=args.generations,split_seed=42,split_unit='identical global 5gram vector group',test_previously_seen_in_earlier_experiments=True)
    result['excluded_failures']=failures
    path=ROOT/'data'/'prepared.json'
    if path.exists(): raise FileExistsError('Frozen split exists; use a new experiment folder instead of overwriting')
    dump(path,result)
    print({k:len(result[k]) for k in ['train','val','test','vocabulary']},flush=True)

def chunks(methods,budget):
    order=sorted(range(len(methods)), key=lambda i:len(methods[i][0]))
    current=[]; longest=0
    for i in order:
        length=len(methods[i][0])
        if current and (len(current)+1)*length>budget:
            yield current; current=[]
        current.append(i); longest=length
    if current: yield current

def capture_rng(device):
    cpu=torch.get_rng_state()
    return (cpu,torch.cuda.get_rng_state(device),device) if device.type=='cuda' else cpu

def restore_rng(state):
    if isinstance(state,tuple):
        torch.set_rng_state(state[0]); torch.cuda.set_rng_state(state[1],state[2])
    else: torch.set_rng_state(state)

def tensors(methods,idx,device=None):
    lengths=torch.tensor([len(methods[i][0]) for i in idx]); t=int(lengths.max())
    ids=torch.zeros(len(idx),t,dtype=torch.long); groups=torch.zeros_like(ids)
    for j,i in enumerate(idx):
        ids[j,:lengths[j]]=torch.from_numpy(methods[i][0]); groups[j,:lengths[j]]=torch.from_numpy(methods[i][1])
    weights=torch.tensor([methods[i][2] for i in idx],dtype=torch.float32)
    return tuple(t.to(device) for t in (ids,groups,lengths,weights)) if device is not None else (ids,groups,lengths,weights)

def apk_forward(net,methods,budget,training=False):
    """Exact chain-rule microbatching. RNG replay reproduces first-pass dropout.
    Backprop head once, then recompute method chunks weighted by multiplicity.
    No methods or positions are truncated. Replay CPU and selected CUDA RNG.
    """
    device=next(net.parameters()).device
    total=sum(m[2] for m in methods); pooled=torch.zeros(1,3*net.width,device=device)
    replay=[]
    with torch.no_grad():
        for idx in chunks(methods,budget):
            rng=capture_rng(device); ids,g,l,w=tensors(methods,idx,device)
            h=net.methods(ids,g,l)
            pooled+=(h*w[:,None]).sum(0,keepdim=True)/total
            if training: replay.append((idx,rng))
    if not training: return net.head(pooled)
    pooled.requires_grad_(True)
    return net.head(pooled),pooled,replay,total

def score(y,p,threshold=.5):
    pred=np.array(p)>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(accuracy=float(accuracy_score(y,pred)),f1=float(f1_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),fpr=float(fp/max(1,tn+fp)),auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,confusion=[[int(tn),int(fp)],[int(fn),int(tp)]])

def train(args):
    torch.set_num_threads(args.threads)
    requested=getattr(args,'device','cpu')
    device=torch.device('cuda' if requested=='auto' and torch.cuda.is_available() else 'cpu' if requested=='auto' else requested)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable; install a suitable CUDA-enabled PyTorch build')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    prepared_path=Path(args.prepared)
    prepared=json.loads(prepared_path.read_text(encoding='utf-8'))
    records=prepared['records']; vocabulary=prepared['vocabulary']
    if args.mode=='test' and not args.confirm_frozen:
        raise ValueError('Final test requires --confirm-frozen after validation-based model selection')
    out=ROOT/'runs'/args.run
    if args.mode=='train' and (out/'config.json').exists(): raise FileExistsError('Choose a new --run; refusing to overwrite')
    out.mkdir(parents=True,exist_ok=True)
    indices=prepared['train']+prepared['val'] if args.mode=='train' else prepared['test']
    from functools import lru_cache
    @lru_cache(maxsize=1)
    def load_bag(i): return apk_methods(records[i],vocabulary)
    class Bags:
        def __getitem__(self,i): return load_bag(i)
    bags=Bags()
    lengths=[len(ids) for i in prepared['train'] for ids,g,w in apk_methods(records[i],vocabulary) for _ in range(w)]
    tau=max(1,float(np.quantile(lengths,.4)))
    net=OriginalLTA(len(vocabulary)) if args.variant=='C0' else Detector(len(vocabulary),args.variant,args.width,args.state,tau)
    if args.variant!='C0':
        # Common named tensors start identically across variants at a given seed.
        with torch.random.fork_rng():
            torch.manual_seed(args.seed)
            common=Detector(len(vocabulary),'A',args.width,args.state,tau).state_dict()
        state=net.state_dict()
        for key in state:
            if key in common and state[key].shape==common[key].shape:
                state[key]=common[key].clone()
        net.load_state_dict(state)
    net.to(device)
    def globals_(i):
        obj=json.loads((ROOT/'data'/'methods'/(records[i]['sha256']+'.json')).read_text(encoding='utf-8'))
        x=torch.zeros(1,len(vocabulary))
        for gram in obj['grams']:
            if gram in vocabulary: x[0,vocabulary[gram]-1]=1
        return x.to(device)
    def evaluate(ids):
        net.eval(); probabilities=[]; losses=[]
        with torch.no_grad():
            for i in ids:
                z=net(globals_(i)) if args.variant=='C0' else apk_forward(net,bags[i],args.tokens)
                probabilities.append(z.softmax(-1)[0,1].item())
                losses.append(F.cross_entropy(z,torch.tensor([records[i]['label']],device=device)).item())
        return probabilities,float(np.mean(losses))
    if args.mode=='test':
        config=json.loads((out/'config.json').read_text(encoding='utf-8'))
        for key in ['variant','width','state','seed']:
            if config[key]!=getattr(args,key): raise ValueError('Config mismatch: '+key)
        if (out/'test.json').exists(): raise FileExistsError('Test already evaluated')
        net.load_state_dict(torch.load(out/'best.pt',weights_only=True,map_location=device))
        p,loss=evaluate(prepared['test']); y=[records[i]['label'] for i in prepared['test']]
        validation=json.loads((out/'best_validation.json').read_text())
        dump(out/'test.json',dict(metrics=score(y,p,validation['threshold']),metrics_at_half=score(y,p),loss=loss,n=len(y),scope='local folder labels; previously explored dataset; not independent publication confirmation'))
        dump(out/'test_predictions.json',[dict(sha256=records[i]['sha256'],label=records[i]['label'],probability=p[j]) for j,i in enumerate(prepared['test'])])
        print(json.dumps(score(y,p),indent=2)); return
    config=vars(args).copy(); config.update(tau=tau,parameters=sum(p.numel() for p in net.parameters()),torch=torch.__version__,device=str(device),prepared_sha256=hashlib.sha256(prepared_path.read_bytes()).hexdigest(),scan='PyTorch Mamba-1 affine recurrence',status='running')
    dump(out/'config.json',config)
    optimizer=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=1e-4)
    best=float('inf'); stale=0; start=time.perf_counter()
    with (out/'trace.csv').open('w',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['epoch','train_loss','validation_loss','seconds'])
        for epoch in range(args.epochs):
            net.train(); losses=[]; order=prepared['train'].copy(); random.shuffle(order)
            for i in order:
                optimizer.zero_grad(set_to_none=True); label=torch.tensor([records[i]['label']],device=device)
                if args.variant=='C0':
                    loss=F.cross_entropy(net(globals_(i)),label); loss.backward()
                else:
                    logits,pooled,replay,total=apk_forward(net,bags[i],args.tokens,True)
                    loss=F.cross_entropy(logits,label); loss.backward()
                    grad=pooled.grad.detach(); after_head=capture_rng(device)
                    for idx,rng in replay:
                        restore_rng(rng); x,g,l,w=tensors(bags[i],idx,device)
                        h=net.methods(x,g,l)
                        (h*w[:,None]/total).sum(0,keepdim=True).backward(grad)
                    restore_rng(after_head)
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                optimizer.step(); losses.append(loss.item())
                print(f'epoch={epoch+1} apk={i} loss={loss.item():.4f}',flush=True)
            p,val_loss=evaluate(prepared['val'])
            elapsed=time.perf_counter()-start
            writer.writerow([epoch+1,np.mean(losses),val_loss,elapsed]); stream.flush()
            print(f'{args.run} epoch={epoch+1} val_loss={val_loss:.4f} seconds={elapsed:.1f}',flush=True)
            if val_loss<best:
                best=val_loss; stale=0; torch.save(net.state_dict(),out/'best.pt')
                y=[records[i]['label'] for i in prepared['val']]
                candidates=[float(np.nextafter(max(p),np.inf)),*sorted(set(p),reverse=True)]
                eligible=[t for t in candidates if score(y,p,t)['fpr']<=.01]
                threshold=max(eligible,key=lambda t:(score(y,p,t)['recall'],t))
                dump(out/'best_validation.json',dict(epoch=epoch+1,loss=best,threshold=threshold,metrics=score(y,p,threshold),metrics_at_half=score(y,p),predictions=[dict(index=i,label=y[j],p=p[j]) for j,i in enumerate(prepared['val'])]))
            else: stale+=1
            if stale>=args.patience: break
    config.update(status='completed',seconds=time.perf_counter()-start,epochs_completed=epoch+1)
    dump(out/'config.json',config)

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='mode',required=True)
    prep=sub.add_parser('prepare'); prep.add_argument('--population',type=int,default=100); prep.add_argument('--generations',type=int,default=20)
    prep.add_argument('--allow-extraction-failures',action='store_true')
    for mode in ['train','test']:
        t=sub.add_parser(mode); t.add_argument('--run',required=True); t.add_argument('--variant',default='C6',choices=['C0','C1','C2','C3','C4','C5','C6','A','B','C','D','F0','F1','F2','D_delta','D_bc']); t.add_argument('--width',type=int,default=64); t.add_argument('--state',type=int,default=16); t.add_argument('--seed',type=int,default=42); t.add_argument('--epochs',type=int,default=200); t.add_argument('--patience',type=int,default=20); t.add_argument('--lr',type=float,default=.001); t.add_argument('--tokens',type=int,default=2048); t.add_argument('--threads',type=int,default=4); t.add_argument('--confirm-frozen',action='store_true')
        t.add_argument('--prepared',default=str(ROOT/'data'/'prepared.json'))
        t.add_argument('--device',choices=['auto','cpu','cuda'],default='cpu')
    args=p.parse_args()
    if args.mode=='prepare': prepare(args)
    else: train(args)

if __name__=='__main__': main()
