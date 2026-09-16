import numpy as np
from backend.analysis.interleavers import block_interleave, convolutional_interleave, diagonal_interleave, pseudo_random_interleave
from backend.analysis.auto_interleaver_detector_v2 import detect_interleaver

def source(n=2048, seed=7):
    rng=np.random.default_rng(seed); out=[]; state=0
    while len(out)<n:
        run=int(rng.integers(8,40)); bit=int(rng.integers(0,2)); out.extend([bit]*run)
        if rng.random()<.25: out.extend([1-bit]*int(rng.integers(1,6)))
    return np.asarray(out[:n],dtype=np.uint8)

def case(name, fn, kw):
    src=source(); enc=fn(src,**kw); r=detect_interleaver(enc)
    print(f'{name:15s} -> {r["detected_type"]:14s} conf={r["confidence"]:6.2f}% applied={r["applied"]} params={r["parameters"]}')
    if not r['applied']: return False
    rec=np.asarray([int(b) for b in r['bitstream']],dtype=np.uint8)
    e=int(np.count_nonzero(rec!=src)); print('  errors=',e); return e==0

def main():
    cases=[('BLOCK',block_interleave,{'rows':8}),('CONVOLUTIONAL',convolutional_interleave,{'branches':4,'delay':3}),('DIAGONAL',diagonal_interleave,{'rows':8}),('PSEUDO-RANDOM',pseudo_random_interleave,{'seed':42})]
    p=sum(case(*c) for c in cases); print(f'PASS: {p}/4'); raise SystemExit(0 if p==4 else 1)
if __name__=='__main__': main()
