import numpy as np
from backend.dsp.analysis import analyze_signal
from backend.preprocessing.pipeline import preprocess

def test_dsp_smoke():
    fs=1e6;t=np.arange(4096)/fs;x=np.exp(1j*2*np.pi*100e3*t)
    y,_=preprocess(x); r=analyze_signal(y,fs)
    assert r['samples']==4096 and np.isfinite(r['snr_db'])

def test_modulation_classifier_smoke():
    from pathlib import Path
    from backend.ml.classifier import ModulationClassifier, _signal
    clf = ModulationClassifier(Path('models/modulation_rf.joblib'))
    rng = np.random.default_rng(7)
    for mod in ['BPSK', 'QPSK', '2-FSK', '4-FSK', '16-QAM']:
        x = _signal(mod, 1e6, n_symbols=160, sps=16, snr=15, rng=rng)
        r = clf.predict({}, x, 1e6)
        assert r['detected'] in ['BPSK','QPSK','2-FSK','4-FSK','16-QAM']
        assert len(r['candidates']) == 5
