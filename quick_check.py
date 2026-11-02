"""Eight full APKs for functional training validation, never a research result."""
import json
import sys
import subprocess
import importlib.util
from collections import defaultdict
import numpy as np
from data import ROOT,dump

if __name__=='__main__':
    rows=json.loads((ROOT/'data'/'manifest.json').read_text(encoding='utf-8'))
    records=[]; train=[]; val=[]; seen=set()
    for label in [0,1]:
        chosen=[]
        for r in sorted([r for r in rows if r.get('status')=='ok' and r['label']==label],key=lambda r:r['positions']):
            if r['feature_group'] not in seen:
                chosen.append(r); seen.add(r['feature_group'])
            if len(chosen)==4: break
        if len(chosen)<4: raise ValueError('Need four distinct feature groups per label')
        start=len(records); records+=chosen; train+=list(range(start,start+3)); val.append(start+3)
    grams={i:json.loads((ROOT/'data'/'methods'/(r['sha256']+'.json')).read_text(encoding='utf-8'))['grams'] for i,r in enumerate(records)}
    vocab=sorted({g for i in train for g in grams[i]}); lookup={g:j for j,g in enumerate(vocab)}
    x=np.zeros((len(records),len(vocab)),dtype=np.float32)
    for i in train:
        for g in grams[i]: x[i,lookup[g]]=1
    sys.path.insert(0,str(ROOT/'reference'/'Static Analysis'))
    spec=importlib.util.spec_from_file_location('reference',ROOT/'reference'/'run_lta.py'); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    selected,history=module.select_features(x[train],np.array([r['label'] for r in records])[train],8,2,42)
    path=ROOT/'data'/'quick_prepared.json'
    if path.exists(): raise FileExistsError('Quick data already exists')
    dump(path,dict(records=records,train=train,val=val,test=[],vocabulary={g:j+1 for j,g in enumerate(np.array(vocab)[selected])},ga_history=history,scope='functional check only; smallest eight APKs selected for speed; deliberately not representative'))
    summary={}
    for variant in ['C0','C2','B','D','C6']:
        run='quick_'+variant+'_seed42'
        subprocess.run([sys.executable,'-u',str(ROOT/'experiment.py'),'train','--prepared',str(path),'--run',run,'--variant',variant,'--epochs','3','--threads','2'],check=True,cwd=ROOT)
        result=json.loads((ROOT/'runs'/run/'best_validation.json').read_text())
        summary[variant]=result
        dump(ROOT/'quick_results.json',dict(scope='8 full APKs, train6/validation2, no test; functional check, no effectiveness conclusion',results=summary))
    print('QUICK CHECK COMPLETE',flush=True)
