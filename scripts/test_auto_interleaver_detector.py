import numpy as np
from backend.analysis.interleavers import block_interleave, convolutional_interleave, diagonal_interleave, pseudo_random_interleave
from backend.analysis.auto_interleaver_detector import detect_interleaver

def structured_bits(length=2048):
    pat="0000000011111111001100110011001100001111000011111111111100000000"
    s=(pat*((length//len(pat))+1))[:length]
    return np.fromiter((int(b) for b in s), dtype=np.uint8)

def run(name, fn, kwargs):
    original=structured_bits(); inter=fn(original,**kwargs)
    r=detect_interleaver(inter)
    print(f"{name:15s} -> {r['detected_type']:14s} confidence={r['confidence']:6.2f}% applied={r['applied']} params={r['parameters']}")
    if not r["applied"]: return False
    rec=np.asarray([int(b) for b in r["bitstream"]],dtype=np.uint8)
    errors=int(np.count_nonzero(rec!=original)); print("  errors=",errors); return errors==0

cases=[("BLOCK",block_interleave,{"rows":8}),("CONVOLUTIONAL",convolutional_interleave,{"branches":4,"delay":3}),("DIAGONAL",diagonal_interleave,{"rows":8}),("PSEUDO-RANDOM",pseudo_random_interleave,{"seed":42})]
p=sum(run(*c) for c in cases)
print(f"PASS: {p}/{len(cases)}")
if p != len(cases): raise SystemExit(1)
print("Automatic detector test: PASS")
