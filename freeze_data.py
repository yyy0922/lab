"""Copy source bytes into an experiment-owned immutable-by-convention snapshot."""
import hashlib
import shutil
import os
from pathlib import Path
from data import ROOT, dump

if __name__=='__main__':
    base=(ROOT/'data'/'decoded').resolve()
    for p in base.glob('*'):
        target=p.resolve()
        if (ROOT/'data'/'methods'/(p.name+'.json')).exists():
            assert target.is_relative_to(base) and target!=base
            shutil.rmtree('\\\\?\\'+str(target) if os.name=='nt' else target)
    dest=ROOT/'data'/'apk_snapshot'
    if (ROOT/'data'/'snapshot_inventory.json').exists(): raise FileExistsError('Completed snapshot exists; will not modify it')
    files=[p for folder in ['benign','malradar','new'] for p in sorted((ROOT.parent/'apk'/folder).glob('*.apk'))]
    records=[]
    for p in files:
        out=dest/p.parent.name/p.name; out.parent.mkdir(parents=True,exist_ok=True)
        try:
            shutil.copy2(p,out)
            digest=hashlib.sha256(out.read_bytes()).hexdigest()
        except OSError as exc:
            records.append(dict(source=str(p),snapshot=str(out),status='copy_or_read_failed',error=repr(exc))); continue
        records.append(dict(source=str(p),snapshot=str(out),sha256=digest,bytes=out.stat().st_size,status='copied'))
    dump(ROOT/'data'/'snapshot_inventory.json',records)
    print('Frozen',sum(r['status']=='copied' for r in records),'APKs',flush=True)
