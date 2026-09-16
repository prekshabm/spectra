import io, wave
import numpy as np

def load_signal(name, raw, sample_rate, dtype='float32', iq_format='IQ'):
    if name.lower().endswith('.wav'):
        with wave.open(io.BytesIO(raw),'rb') as w:
            ch,nbytes,fs,nframes=w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes()
            frames=w.readframes(nframes)
        if nbytes==2: a=np.frombuffer(frames,np.int16).astype(np.float32)/32768
        elif nbytes==4: a=np.frombuffer(frames,np.int32).astype(np.float32)/2147483648
        else: raise ValueError('Only 16-bit or 32-bit WAV is supported')
        if ch>=2: sig=a.reshape(-1,ch)[:,0]+1j*a.reshape(-1,ch)[:,1]
        else:
            # Mono WAV is treated as real-valued baseband; analytic signal is formed by Hilbert in preprocessing.
            from scipy.signal import hilbert
            sig=hilbert(a)
        return sig.astype(np.complex64),float(fs),{'format':'WAV','channels':ch,'sample_rate':fs}
    dt={'float32':np.float32,'float64':np.float64,'int16':np.int16,'int8':np.int8}.get(dtype)
    if dt is None: raise ValueError('Unsupported dtype')
    a=np.frombuffer(raw,dtype=dt).astype(np.float32)
    if np.issubdtype(dt,np.integer):
        scale=np.iinfo(dt).max; a/=max(scale,1)
    if iq_format.upper() in ('IQ','I/Q'):
        if len(a)%2: a=a[:-1]
        sig=a[0::2]+1j*a[1::2]
    elif iq_format.upper()=='QI':
        if len(a)%2: a=a[:-1]
        sig=a[1::2]+1j*a[0::2]
    elif iq_format.upper()=='REAL':
        from scipy.signal import hilbert
        sig=hilbert(a)
    else: raise ValueError('IQ format must be IQ, QI or REAL')
    return sig.astype(np.complex64),float(sample_rate),{'format':'RAW IQ','dtype':dtype,'iq_format':iq_format,'sample_rate':sample_rate}
