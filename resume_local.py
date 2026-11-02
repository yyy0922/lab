"""Continue unfinished one-epoch pilot without replacing prior run records."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from data import ROOT,dump

if __name__=='__main__':
    prepared=ROOT/'data'/'prepared.json'
    digest=hashlib.sha256(prepared.read_bytes()).hexdigest()
    summary={}
    (ROOT/'pipeline.pid').write_text(str(os.getpid()))
    dump(ROOT/'pipeline_status.json',dict(status='resuming',pid=os.getpid(),prepared_sha256=digest))
    for variant in ['C0','C2','B','D','C6']:
        base=f'pilot_{variant}_seed42'
        candidates=[ROOT/'runs'/base,*sorted((ROOT/'runs').glob(base+'_restart_*'))]
        completed=[]
        for candidate in candidates:
            config=candidate/'config.json'
            if config.exists():
                c=json.loads(config.read_text(encoding='utf-8'))
                if c.get('prepared_sha256')!=digest:
                    raise ValueError('Dataset mismatch: '+str(candidate))
                if c.get('status')=='completed': completed.append(candidate)
        if completed:
            run=completed[-1].name
            print('REUSE COMPLETED',run,flush=True)
        else:
            run=base
            suffix=1
            while (ROOT/'runs'/run).exists():
                run=f'{base}_restart_{suffix:02d}'; suffix+=1
            command=[sys.executable,'-u',str(ROOT/'experiment.py'),'train','--run',run,'--variant',variant,'--epochs','1','--width','64','--state','16','--seed','42']
            dump(ROOT/'pipeline_status.json',dict(status='running',pid=os.getpid(),command=command,restarted_from_initialization=True,prepared_sha256=digest))
            print('RUN',command,flush=True)
            result=subprocess.run(command,cwd=ROOT)
            if result.returncode:
                dump(ROOT/'pipeline_status.json',dict(status='failed',pid=os.getpid(),run=run,returncode=result.returncode))
                sys.exit(result.returncode)
        folder=ROOT/'runs'/run
        summary[variant]=dict(run=run,config=json.loads((folder/'config.json').read_text(encoding='utf-8')),validation=json.loads((folder/'best_validation.json').read_text(encoding='utf-8')))
        dump(ROOT/'full_pilot_summary.json',dict(scope='one epoch, seed42, validation only; no convergence or superiority claim',results=summary))
    subprocess.run([sys.executable,str(ROOT/'length_diagnostic.py'),'--out',str(ROOT/'full_length_diagnostic.json')],cwd=ROOT,check=True)
    dump(ROOT/'pipeline_status.json',dict(status='complete',pid=os.getpid(),test_evaluated=False,summary=str(ROOT/'full_pilot_summary.json')))
    print('FULL PILOT COMPLETE; TEST UNTOUCHED',flush=True)
