"""Static APK extraction and exact position encoding. APKs are never executed."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parent
LTA = ROOT.parent / 'LTA-main'  # optional old decoded cache only
GROUP = {'V':1, 'T':2, 'P':2, 'G':3, 'I':3, 'R':3, 'M':4}

def load_map():
    spec = importlib.util.spec_from_file_location('original_map', ROOT / 'reference' / 'map.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.bytecode_map

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def encode_method(tokens, vocabulary):
    grams = [tokens[i:i+5] for i in range(max(0, len(tokens)-4))]
    return (np.asarray([vocabulary.get(g, 0) for g in grams], dtype=np.int64),
            np.asarray([GROUP[g[2]] for g in grams], dtype=np.int64))

def parse(decoded):
    mapping = load_map()
    methods, meta = [], []
    for smali in sorted(decoded.glob('smali*')):
        if not smali.is_dir(): continue
        for directory, dirs, files in os.walk(smali):
            dirs.sort()
            for filename in sorted(files):
                if not filename.endswith('.smali'): continue
                path = Path(directory) / filename
                readable = '\\\\?\\' + str(path.resolve()) if os.name == 'nt' else str(path)
                current = None
                with open(readable, encoding='utf-8', errors='replace') as stream:
                    for line in stream:
                        parts = line.strip().split(maxsplit=1)
                        if not parts: continue
                        if parts[0] == '.method':
                            current = []
                            signature = parts[1] if len(parts)>1 else ''
                        elif parts[0] == '.end' and len(parts)>1 and parts[1]=='method':
                            if current is not None:
                                methods.append(''.join(current))
                                meta.append([path.relative_to(decoded).as_posix(), signature])
                            current = None
                        elif current is not None and parts[0] in mapping:
                            current.append(mapping[parts[0]])
    return methods, meta

def extract(apk_root, java, workers=4):
    import time
    cache = ROOT/'data'/'methods'
    cache.mkdir(parents=True, exist_ok=True)
    records = []
    labels = {}
    def worker(job):
            apk,folder,label=job
            sha = hashlib.sha256(apk.read_bytes()).hexdigest()
            item = dict(sha256=sha, apk=str(apk.resolve()), label=label, source=folder)
            if label is None:
                item['status']='unlabeled'; return item
            out = cache/(sha+'.json')
            if out.exists():
                saved = json.loads(out.read_text(encoding='utf-8'))
                item.update(saved['audit'])
            else:
                start = time.perf_counter()
                decoded = LTA/'runs'/'bundled_46'/'decoded'/sha
                if not (decoded/'apktool.yml').exists():
                    decoded = ROOT/'data'/'decoded'/sha
                    if not (decoded/'apktool.yml').exists():
                        decoded.parent.mkdir(parents=True, exist_ok=True)
                        jar=ROOT/'tools'/'apktool.jar'
                        if not jar.exists(): raise FileNotFoundError('Missing bundled tools/apktool.jar')
                        r = subprocess.run([java,'-jar',str(jar),'d','-f','-r','-o',str(decoded),str(apk)], capture_output=True, text=True)
                        if r.returncode:
                            item['status']='decode_failed'; item['error']=r.stdout[-2000:]+r.stderr[-2000:]
                            return item
                methods, meta = parse(decoded)
                valid = [m for m in methods if len(m)>=5]
                grams = sorted({m[i:i+5] for m in valid for i in range(len(m)-4)})
                audit = dict(status='ok' if valid else 'representation_unavailable', methods=len(methods), usable_methods=len(valid), empty_methods=len(methods)-len(valid), positions=sum(len(m)-4 for m in valid), feature_group=hashlib.sha256('|'.join(grams).encode()).hexdigest(), extraction_seconds=time.perf_counter()-start)
                dump(out,dict(sha256=sha,audit=audit,sequences=methods,locations=meta,grams=grams))
                item.update(audit)
                temporary=(ROOT/'data'/'decoded'/sha).resolve()
                if decoded.resolve()==temporary and temporary.is_relative_to((ROOT/'data'/'decoded').resolve()):
                    shutil.rmtree('\\\\?\\'+str(temporary) if os.name=='nt' else temporary)
            return item
    from concurrent.futures import ThreadPoolExecutor, as_completed
    jobs=[(p,folder,label) for folder,label in [('benign',0),('malradar',1),('new',None)] for p in sorted((Path(apk_root)/folder).glob('*.apk'))]
    if Path(apk_root).resolve()==(ROOT/'data'/'apk_snapshot').resolve():
        inventory=json.loads((ROOT/'data'/'snapshot_inventory.json').read_text(encoding='utf-8'))
        jobs=[]
        for r in inventory:
            if r['status']!='copied': continue
            portable=r['snapshot'].replace('\\','/').split('/')
            folder,name=portable[-2:]
            jobs.append((Path(apk_root)/folder/name,folder,{'benign':0,'malradar':1,'new':None}[folder]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(worker,job):job for job in jobs}
        for future in as_completed(futures):
            try:
                item=future.result()
            except Exception as exc:
                apk,folder,label=futures[future]
                item=dict(sha256=apk.stem,identity_verified=False,apk=str(apk.resolve()),source=folder,label=label,status='parse_failed',error=repr(exc))
                print('EXTRACTION ERROR',apk.name,repr(exc),flush=True)
            sha=item['sha256']
            if sha in labels and labels[sha]!=item['label']: raise ValueError('Conflicting labels: '+sha)
            labels[sha]=item['label']; records.append(item)
            dump(ROOT/'data'/'manifest.json',sorted(records,key=lambda x:(x['source'],x['sha256'])))
            print(f"extract {len(records)}/{len(jobs)} {sha[:10]} {item['status']} methods={item.get('usable_methods',0)}",flush=True)
    records.sort(key=lambda x:(x['source'],x['sha256']))
    dump(ROOT/'data'/'manifest.json',records)
    return records

def apk_methods(record, vocabulary):
    obj=json.loads((ROOT/'data'/'methods'/(record['sha256']+'.json')).read_text(encoding='utf-8'))
    # Compress only byte-identical sequences, retaining exact multiplicity in pooling.
    from collections import Counter
    counts=Counter(s for s in obj['sequences'] if len(s)>=5)
    return [( *encode_method(s,vocabulary), count) for s,count in counts.items()]
