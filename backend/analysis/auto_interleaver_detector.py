"""SPECTRA automatic interleaver detector.

Conservative, reference-free heuristic over the four deterministic
interleavers already present in SPECTRA. It returns UNKNOWN unless the
best candidate has sufficient score and margin over alternatives.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
from backend.analysis.interleavers import (
    block_deinterleave,
    convolutional_deinterleave,
    diagonal_deinterleave,
    pseudo_random_deinterleave,
)
MIN_BITS = 128
DEFAULT_CONFIDENCE_THRESHOLD = 0.55
DEFAULT_MARGIN_THRESHOLD = 0.08
BLOCK_ROWS = (2, 4, 8, 16, 32)
DIAGONAL_ROWS = (2, 4, 8, 16, 32)
CONVOLUTIONAL_CONFIGS = ((2,1),(3,1),(3,2),(4,2),(4,3),(5,2),(5,3))
PSEUDORANDOM_SEEDS = tuple(range(256))

def _clean_bits(bits) -> np.ndarray:
    if isinstance(bits, str): bits = [int(b) for b in bits if b in ("0","1")]
    arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    return arr[(arr == 0) | (arr == 1)] if arr.size else arr

def _transition_rate(bits):
    return 1.0 if len(bits) < 2 else float(np.mean(bits[1:] != bits[:-1]))

def _lag_correlation(bits, lag):
    if lag <= 0 or len(bits) <= lag + 4: return 0.0
    x = bits.astype(np.float64) * 2.0 - 1.0
    a, b = x[:-lag], x[lag:]
    a, b = a - np.mean(a), b - np.mean(b)
    den = float(np.sqrt(np.dot(a,a) * np.dot(b,b)))
    return 0.0 if den <= 1e-12 else abs(float(np.dot(a,b) / den))

def _run_score(bits):
    tr = _transition_rate(bits)
    return float(np.clip(1.0 - 2.0 * abs(tr - 0.50), 0.0, 1.0))

def _autocorr_score(bits):
    return max((_lag_correlation(bits, lag) for lag in (1,2,3,4,8,16,32,64)), default=0.0)

def _block_consistency_score(bits):
    n = len(bits); scores = []
    for block_size in (8,16,32,64,128):
        if block_size >= n: continue
        local = [_transition_rate(bits[s:s+block_size]) for s in range(0, n-block_size+1, block_size)]
        if len(local) >= 2:
            scores.append(float(np.clip(1.0 - 4.0 * np.std(local), 0.0, 1.0)))
    return max(scores, default=0.0)

def _structure_score(bits):
    if len(bits) < 16: return 0.0
    return float(np.clip(0.45*_autocorr_score(bits) + 0.30*_block_consistency_score(bits) + 0.25*_run_score(bits), 0.0, 1.0))

def _candidate_score(observed, recovered):
    improvement = _structure_score(recovered) - _structure_score(observed)
    return float(np.clip(0.50 + 1.50 * improvement, 0.0, 1.0))

def _add(candidates, observed, recovered, kind, params):
    if len(recovered) != len(observed) or len(recovered) < MIN_BITS: return
    candidates.append({"type": kind, "parameters": params, "score": _candidate_score(observed, recovered)})

def detect_interleaver(bits, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, margin_threshold=DEFAULT_MARGIN_THRESHOLD) -> Dict:
    observed = _clean_bits(bits)
    if len(observed) < MIN_BITS:
        return {"detected_type":"UNKNOWN","confidence":0.0,"parameters":{},"applied":False,"input_bits":int(len(observed)),"output_bits":0,"bitstream":"","candidates":[],"note":f"At least {MIN_BITS} bits are required for automatic interleaver detection."}
    candidates=[]; n=len(observed)
    for rows in BLOCK_ROWS:
        if n % rows == 0:
            try: _add(candidates, observed, block_deinterleave(observed, rows=rows), "BLOCK", {"rows":int(rows)})
            except Exception: pass
    for branches, delay in CONVOLUTIONAL_CONFIGS:
        try: _add(candidates, observed, convolutional_deinterleave(observed, branches=branches, delay=delay), "CONVOLUTIONAL", {"branches":int(branches),"delay":int(delay)})
        except Exception: pass
    for rows in DIAGONAL_ROWS:
        if n % rows == 0:
            try: _add(candidates, observed, diagonal_deinterleave(observed, rows=rows), "DIAGONAL", {"rows":int(rows)})
            except Exception: pass
    for seed in PSEUDORANDOM_SEEDS:
        try: _add(candidates, observed, pseudo_random_deinterleave(observed, seed=seed), "PSEUDO-RANDOM", {"seed":int(seed)})
        except Exception: pass
    if not candidates:
        return {"detected_type":"UNKNOWN","confidence":0.0,"parameters":{},"applied":False,"input_bits":int(n),"output_bits":0,"bitstream":"","candidates":[],"note":"No valid deterministic interleaver candidate could be evaluated."}
    candidates.sort(key=lambda x:x["score"], reverse=True)
    best=candidates[0]; second=candidates[1]["score"] if len(candidates)>1 else 0.0
    margin=float(best["score"]-second)
    confidence=float(np.clip(best["score"]*100.0,0,100))
    applied=best["score"] >= confidence_threshold and margin >= margin_threshold
    if applied:
        recovered=_recover(observed,best)
        return {"detected_type":best["type"],"confidence":round(confidence,2),"margin":round(margin,4),"parameters":best["parameters"],"applied":True,"input_bits":int(n),"output_bits":int(len(recovered)),"bitstream":"".join(str(int(b)) for b in recovered),"candidates":candidates[:8],"note":"Automatic reference-free detection. This is a heuristic estimate; framing or known reference information can provide stronger evidence."}
    return {"detected_type":"UNKNOWN","confidence":round(confidence,2),"margin":round(margin,4),"parameters":best["parameters"],"applied":False,"input_bits":int(n),"output_bits":0,"bitstream":"","candidates":candidates[:8],"note":"The best candidate was not sufficiently separated from alternatives, so no automatic deinterleaving was applied."}

def _recover(observed, candidate):
    k,p=candidate["type"],candidate["parameters"]
    if k=="BLOCK": return block_deinterleave(observed, rows=int(p["rows"]))
    if k=="CONVOLUTIONAL": return convolutional_deinterleave(observed, branches=int(p["branches"]), delay=int(p["delay"]))
    if k=="DIAGONAL": return diagonal_deinterleave(observed, rows=int(p["rows"]))
    if k=="PSEUDO-RANDOM": return pseudo_random_deinterleave(observed, seed=int(p["seed"]))
    raise ValueError(f"Unsupported candidate type: {k}")

def auto_deinterleave(bits): return detect_interleaver(bits)
