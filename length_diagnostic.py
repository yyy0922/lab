"""Training-only length-statistics classifier; validation diagnostic, no test."""
import argparse
import json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from data import ROOT,dump
from experiment import score

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--prepared',default=str(ROOT/'data'/'prepared.json')); p.add_argument('--out',default=str(ROOT/'length_diagnostic.json')); args=p.parse_args()
    prepared=json.loads(open(args.prepared,encoding='utf-8').read()); rows=prepared['records']
    x=np.array([[np.log1p(r['positions']),np.log1p(r['methods']),np.log1p(r['usable_methods']),r['empty_methods']/max(1,r['methods'])] for r in rows])
    y=np.array([r['label'] for r in rows]); tr=prepared['train']; va=prepared['val']
    classifier=make_pipeline(StandardScaler(),LogisticRegression(random_state=42,class_weight='balanced'))
    classifier.fit(x[tr],y[tr]); probabilities=classifier.predict_proba(x[va])[:,1]
    report=dict(scope='validation only; a high score indicates that size shortcuts are plausible, not proof that the neural model uses them',features=['log_positions','log_method_count','log_usable_methods','empty_fraction'],validation=score(y[va],probabilities),predictions=[dict(index=i,label=int(y[i]),p=float(probabilities[j])) for j,i in enumerate(va)])
    dump(args.out,report); print(json.dumps(report,indent=2))
