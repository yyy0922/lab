"""One-shot local pipeline. Explicit pilot settings; no final test access."""
import subprocess
import sys
import argparse
import os
from data import ROOT, extract, dump

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-pid',type=int)
    args=parser.parse_args()
    (ROOT/'pipeline.pid').write_text(str(os.getpid()))
    dump(ROOT/'pipeline_status.json',dict(status='waiting_for_extraction' if args.wait_pid else 'extracting',pid=os.getpid()))
    if args.wait_pid:
        import psutil
        try: psutil.Process(args.wait_pid).wait()
        except psutil.NoSuchProcess: pass
    # Extraction is resumable. Never launch this concurrently with extraction.
    source=ROOT/'data'/'apk_snapshot' if (ROOT/'data'/'apk_snapshot').exists() else ROOT.parent/'apk'
    extract(source,r'C:\Program Files\Java\jdk-1.8\bin\java.exe',workers=2)
    commands=[]
    if not (ROOT/'data'/'prepared.json').exists():
        commands.append(['prepare','--population','20','--generations','5','--allow-extraction-failures'])
    for variant in ['C0','C2','B','D','C6']:
        commands.append(['train','--run',f'pilot_{variant}_seed42','--variant',variant,'--epochs','1','--width','64','--state','16'])
    for command in commands:
        print('RUN',command,flush=True)
        dump(ROOT/'pipeline_status.json',dict(status='running',command=command,pid=os.getpid()))
        try:
            subprocess.run([sys.executable,'-u',str(ROOT/'experiment.py'),*command],cwd=ROOT,check=True)
        except subprocess.CalledProcessError as exc:
            dump(ROOT/'pipeline_status.json',dict(status='failed',command=command,returncode=exc.returncode,pid=os.getpid()))
            raise
    dump(ROOT/'pilot_status.json',dict(status='complete',scope='one-epoch pipeline validation, not convergence or superiority evidence',test_evaluated=False))
    dump(ROOT/'pipeline_status.json',dict(status='complete',test_evaluated=False,pid=os.getpid()))
