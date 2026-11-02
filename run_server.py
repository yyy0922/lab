"""Linux/Windows runner. Uses frozen cached inputs, no Java or extraction."""
import argparse
import subprocess
import sys
from data import ROOT

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    p.add_argument('--variants',nargs='+',default=['C2','B','D','C6'])
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--prefix',default='server_pilot')
    p.add_argument('--tokens',type=int,default=2048)
    args=p.parse_args()
    subprocess.run([sys.executable,str(ROOT/'server_preflight.py'),'--device',args.device],check=True)
    for variant in args.variants:
        name=f'{args.prefix}_{variant}_seed{args.seed}'
        if (ROOT/'runs'/name).exists(): raise FileExistsError('Run exists; choose another --prefix: '+name)
        subprocess.run([sys.executable,'-u',str(ROOT/'experiment.py'),'train','--run',name,'--variant',variant,'--seed',str(args.seed),'--epochs',str(args.epochs),'--device',args.device,'--tokens',str(args.tokens)],cwd=ROOT,check=True)
    print('QUEUE COMPLETE. Validation records saved; test set not evaluated.',flush=True)
