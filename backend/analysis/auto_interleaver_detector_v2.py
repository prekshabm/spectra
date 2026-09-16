"""Conservative automatic interleaver detector for SPECTRA.

Reference-free heuristic: evaluates candidate inverse transforms in several
independent windows. A method is accepted only when the same parameterized
candidate consistently wins across windows with a clear margin.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
from backend.analysis.interleavers import (
    block_deinterleave, convolutional_deinterleave,
    diagonal_deinterleave, pseudo_random_deinterleave,
)
MIN_BITS = 256
BLOCK_ROWS = (2, 4, 8, 16, 32)
DIAGONAL_ROWS = (2, 4, 8, 16, 32)
CONV_CONFIGS = ((2,1),(3,1),(3,2),(4,2),(4,3),(5,2),(5,3))
PR_SEEDS = tuple(range(256))

def _bits(x):
    if isinstance(x, str): x=[int(b) for b in x if b in '01']
    a=np.asarray(x,dtype=np.uint8).reshape(-1)
    return a[(a==0)|(a==1)] if a.size else a

def _score(x):
    if len(x)<16: return 0.0
    z=x.astype(np.float64)*2-1
    tr=float(np.mean(x[1:]!=x[:-1]))
    run=float(np.clip(1-2*abs(tr-.5),0,1))
    vals=[]
    for lag in (1,2,4,8,16,32):
        if len(x)<=lag+4: continue
        a=z[:-lag]; b=z[lag:]; a-=a.mean(); b-=b.mean()
        d=np.sqrt(np.dot(a,a)*np.dot(b,b))
        vals.append(abs(float(np.dot(a,b)/d)) if d>1e-12 else 0.0)
    ac=max(vals) if vals else 0.0
    # repeated short patterns; deterministic payloads become more repetitive
    grams=[]
    for k in (4,8):
        if len(x)<k*4: continue
        s=[''.join(map(str,x[i:i+k])) for i in range(len(x)-k+1)]
        grams.append(len(set(s))/len(s))
    rep=1-min(grams) if grams else 0.0
    return float(np.clip(.50*ac+.30*run+.20*rep,0,1))

def _windows(bits):
    n=len(bits); w=max(64,n//4)
    out=[]
    for i in range(4):
        a=i*w; b=n if i==3 else min(n,(i+1)*w)
        if b-a>=64: out.append(bits[a:b])
    return out

def _eval(obs, kind, params):
    try:
        if kind=='BLOCK': y=block_deinterleave(obs, rows=params['rows'])
        elif kind=='CONVOLUTIONAL': y=convolutional_deinterleave(obs, branches=params['branches'], delay=params['delay'])
        elif kind=='DIAGONAL': y=diagonal_deinterleave(obs, rows=params['rows'])
        elif kind=='PSEUDO-RANDOM': y=pseudo_random_deinterleave(obs, seed=params['seed'])
        else: return None
        y=np.asarray(y,dtype=np.uint8).reshape(-1)
        if len(y)!=len(obs): return None
        return y
    except Exception:
        return None

def _candidates():
    for r in BLOCK_ROWS: yield 'BLOCK',{'rows':r}
    for b,d in CONV_CONFIGS: yield 'CONVOLUTIONAL',{'branches':b,'delay':d}
    for r in DIAGONAL_ROWS: yield 'DIAGONAL',{'rows':r}
    for s in PR_SEEDS: yield 'PSEUDO-RANDOM',{'seed':s}

def detect_interleaver(bits, min_confidence=0.70, min_margin=0.10):
    obs=_bits(bits)
    if len(obs)<MIN_BITS:
        return {'detected_type':'UNKNOWN','confidence':0.0,'applied':False,'parameters':{},'input_bits':int(len(obs)),'output_bits':0,'bitstream':'','candidates':[],'note':f'At least {MIN_BITS} bits are required.'}
    ws=_windows(obs)
    base=np.mean([_score(w) for w in ws])
    rows=[]
    for kind,p in _candidates():
        scores=[]
        for w in ws:
            y=_eval(w,kind,p)
            if y is None: scores=[]; break
            scores.append(max(0.0,_score(y)-_score(w)))
        if not scores: continue
        # consistency is crucial: reward candidates that help most windows.
        wins=sum(s>0.03 for s in scores)/len(scores)
        mean=float(np.mean(scores)); spread=float(np.std(scores))
        score=float(np.clip(.55*mean/.25 + .30*wins + .15*(1-min(spread/.20,1)),0,1))
        rows.append({'type':kind,'parameters':p,'score':score,'window_scores':[round(float(s),4) for s in scores],'windows_won':round(wins,2)})
    rows.sort(key=lambda r:r['score'],reverse=True)
    if not rows:
        return {'detected_type':'UNKNOWN','confidence':0.0,'applied':False,'parameters':{},'input_bits':int(len(obs)),'output_bits':0,'bitstream':'','candidates':[],'note':'No usable candidates.'}
    best=rows[0]; second=rows[1]['score'] if len(rows)>1 else 0.0
    margin=float(best['score']-second)
    confidence=float(np.clip(best['score']*100,0,100))
    applied=best['score']>=min_confidence and margin>=min_margin and best['windows_won']>=0.75
    if applied:
        y=_eval(obs,best['type'],best['parameters'])
        if y is not None:
            return {'detected_type':best['type'],'confidence':round(confidence,2),'margin':round(margin,4),'applied':True,'parameters':best['parameters'],'input_bits':int(len(obs)),'output_bits':int(len(y)),'bitstream':''.join(str(int(b)) for b in y),'candidates':rows[:8],'note':'Automatic reference-free heuristic; framing or known-reference information is stronger evidence.'}
    return {'detected_type':'UNKNOWN','confidence':round(confidence,2),'margin':round(margin,4),'applied':False,'parameters':best['parameters'],'input_bits':int(len(obs)),'output_bits':0,'bitstream':'','candidates':rows[:8],'note':'No candidate was sufficiently consistent and separated; no deinterleaving was applied.'}

auto_deinterleave=detect_interleaver
