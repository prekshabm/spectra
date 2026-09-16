import numpy as np
from scipy.signal import medfilt

def preprocess(x):
    x=np.asarray(x,dtype=np.complex64)
    x=x[np.isfinite(x)]
    if len(x)<32: raise ValueError('Signal is too short')
    dc=np.mean(x); x=x-dc
    mag=np.abs(x); lim=np.percentile(mag,99.9)*4+1e-12
    clipped=int(np.sum(mag>lim))
    if clipped: x=np.where(mag>lim,x*(lim/mag),x)
    rms=np.sqrt(np.mean(np.abs(x)**2))+1e-12
    x=x/rms
    return x,{'dc_removed':float(abs(dc)),'normalized_rms':float(np.sqrt(np.mean(abs(x)**2))),'outliers_clipped':clipped}
