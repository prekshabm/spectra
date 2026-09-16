from pathlib import Path
import numpy as np
import joblib
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier

CLASSES = ['BPSK', 'QPSK', '2-FSK', '4-FSK', '16-QAM']
FEATURES = [
    'c20_abs','c40_abs','c42_abs','amp_cv','amp_skew','amp_kurt',
    'dphi_std','dphi_kurt','phase_coh2','phase_coh4','freq_std_norm',
    'spectral_flatness','spectral_peak_ratio','spectral_peaks','freq_hist_peaks','freq_hist_entropy'
]


def _normalize(x):
    x = np.asarray(x, dtype=np.complex128)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return x
    return x / (np.sqrt(np.mean(np.abs(x)**2)) + 1e-12)


def _extract(x, fs):
    z = _normalize(x)
    if len(z) < 64:
        return np.zeros(len(FEATURES), dtype=float)

    # Estimate and remove carrier/frequency offset using power-spectrum peaks.
    # M=2 is especially informative for BPSK; M=4 collapses QPSK symbols.
    nn0 = min(len(z), 32768)
    zz0 = z[:nn0] * np.hanning(nn0)
    candidates = []
    for M in (2, 4):
        pm = np.abs(np.fft.fftshift(np.fft.fft(zz0**M)))**2 + 1e-18
        fm = np.fft.fftshift(np.fft.fftfreq(nn0, 1/fs))
        k = int(np.argmax(pm))
        # A power-spectrum peak at M*f0 implies f0=peak/M.
        f0 = float(fm[k] / M)
        # Fold to the principal interval.
        f0 = ((f0 + fs/2) % fs) - fs/2
        ratio = float(pm[k] / (np.median(pm) + 1e-18))
        candidates.append((ratio, f0))
    # Prefer the strongest collapsed-symbol spectral line.
    _, cfo = max(candidates, key=lambda q: q[0])
    n = np.arange(len(z))
    zc = z * np.exp(-1j * 2*np.pi*cfo*n/fs)
    dphi = np.angle(zc[1:] * np.conj(zc[:-1]))

    m20 = np.mean(zc**2)
    m40 = np.mean(zc**4)
    m42 = np.mean(np.abs(zc)**4)
    m21 = np.mean(np.abs(zc)**2)
    c40 = m40 - 3*m20*m20
    c42 = m42 - abs(m20)**2 - 2*m21*m21

    amp = np.abs(zc)
    acv = float(np.std(amp) / (np.mean(amp) + 1e-12))
    askew = float(skew(amp, bias=False)) if len(amp) > 8 else 0.0
    akurt = float(kurtosis(amp, fisher=True, bias=False)) if len(amp) > 8 else 0.0

    # Phase-transition statistics are more stable than raw instantaneous frequency.
    dphi_center = dphi - np.median(dphi)
    dstd = float(np.std(dphi_center))
    dkurt = float(kurtosis(dphi_center, fisher=True, bias=False)) if len(dphi_center) > 8 else 0.0
    coh2 = float(abs(np.mean(np.exp(1j * 2 * np.angle(zc)))))
    coh4 = float(abs(np.mean(np.exp(1j * 4 * np.angle(zc)))))
    freq_std_norm = float(dstd / (2*np.pi) * fs / (fs/2 + 1e-12))

    # Instantaneous-frequency histogram: FSK creates multiple stable frequency levels,
    # whereas PSK creates a dominant near-zero level with sparse phase jumps.
    hist, _ = np.histogram(dphi, bins=64, range=(-np.pi, np.pi), density=True)
    hs = np.convolve(hist, np.ones(3)/3, mode='same')
    hpk, _ = find_peaks(hs, distance=5, prominence=max(np.max(hs)*0.06, 1e-3))
    freq_hist_peaks = float(min(len(hpk), 8))
    hp = hs / (np.sum(hs) + 1e-12)
    freq_hist_entropy = float(-np.sum(hp*np.log(hp+1e-12)) / np.log(len(hp)))

    # Robust spectral shape statistics.
    nn = min(len(zc), 32768)
    zz = zc[:nn] * np.hanning(nn)
    p = np.abs(np.fft.fftshift(np.fft.fft(zz)))**2 + 1e-18
    p = p / np.sum(p)
    flat = float(np.exp(np.mean(np.log(p))) / (np.mean(p) + 1e-18))
    peak_ratio = float(np.max(p) / (np.median(p) + 1e-18))
    # Count strong separated spectral lobes; useful for FSK.
    pk, props = find_peaks(p, distance=max(4, nn//256), prominence=np.max(p)*0.015)
    spectral_peaks = float(min(len(pk), 20))

    return np.array([
        abs(m20), abs(c40), abs(c42), acv, askew, akurt,
        dstd, dkurt, coh2, coh4, freq_std_norm,
        flat, peak_ratio, spectral_peaks, freq_hist_peaks, freq_hist_entropy
    ], dtype=float)


def _signal(mod, fs=1e6, n_symbols=300, sps=16, snr=18, rng=None):
    rng = rng or np.random.default_rng()
    n = n_symbols * sps
    t = np.arange(n) / fs

    if mod == 'BPSK':
        a = rng.integers(0, 2, n_symbols)
        sy = 2*a - 1
        base = np.repeat(sy, sps).astype(complex)
    elif mod == 'QPSK':
        a = rng.integers(0, 4, n_symbols)
        sy = np.exp(1j*(np.pi/4 + a*np.pi/2))
        base = np.repeat(sy, sps)
    elif mod == '16-QAM':
        a = rng.integers(0, 16, n_symbols)
        I = 2*(a % 4) - 3
        Q = 2*(a // 4) - 3
        sy = (I + 1j*Q) / np.sqrt(10)
        base = np.repeat(sy, sps)
    else:
        m = 2 if mod == '2-FSK' else 4
        a = rng.integers(0, m, n_symbols)
        tones = np.arange(m) - (m-1)/2
        # Keep tone spacing comfortably inside Nyquist.
        spacing = rng.uniform(0.015, 0.06) * fs
        fseq = np.repeat(tones[a], sps) * spacing
        phase = np.cumsum(2*np.pi*fseq/fs)
        base = np.exp(1j*phase)
        # Add small smooth phase jitter to avoid an unrealistically perfect FSK waveform.
        if rng.random() < 0.4:
            base *= np.exp(1j*np.cumsum(rng.normal(0, 0.002, n)))

    # Random channel impairments.
    cfo = rng.uniform(-0.18, 0.18) * fs
    phase0 = rng.uniform(-np.pi, np.pi)
    base = base * np.exp(1j*(2*np.pi*cfo*t + phase0))

    # Random amplitude scaling and mild fading.
    base *= rng.uniform(0.7, 1.3)
    if rng.random() < 0.25:
        fade = 1 + rng.normal(0, 0.025, n)
        base *= fade

    power = np.mean(np.abs(base)**2)
    noise_power = power / (10**(snr/10))
    noise = np.sqrt(noise_power/2) * (rng.normal(size=n) + 1j*rng.normal(size=n))
    return base + noise


def train(path, n_per_class=300):
    rng = np.random.default_rng(42)
    X, y = [], []
    for c in CLASSES:
        for _ in range(n_per_class):
            fs = 1e6
            sps = int(rng.choice([8, 10, 12, 16, 20, 24, 32]))
            snr = float(rng.uniform(4, 26))
            n_symbols = int(rng.choice([180, 240, 300, 400]))
            sig = _signal(c, fs=fs, n_symbols=n_symbols, sps=sps, snr=snr, rng=rng)
            X.append(_extract(sig, fs))
            y.append(c)

    X = np.nan_to_num(np.asarray(X), nan=0.0, posinf=1e6, neginf=-1e6)
    clf = RandomForestClassifier(
        n_estimators=350,
        max_depth=18,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': clf, 'features': FEATURES, 'classes': CLASSES}, path)
    return clf


def _expert_scores(f):
    """Physics-guided scores. These are soft scores, not hard-coded labels."""
    c20,c40,c42,acv,askew,akurt,dstd,dkurt,coh2,coh4,fstd,flat,pratio,peaks,hpeaks,hent = f
    s = {c: 0.05 for c in CLASSES}

    # FSK: constant envelope + noticeably larger phase/frequency variation.
    fsk_strength = np.clip((dstd - 0.12) / 0.50, 0, 1)
    fsk_strength *= np.clip((0.65-acv)/0.65, 0, 1)
    fsk_strength *= np.clip((0.95-hent)/0.95, 0, 1)
    s['2-FSK'] += 1.2*fsk_strength
    s['4-FSK'] += 1.2*fsk_strength
    if hpeaks == 2: s['2-FSK'] += 1.8*fsk_strength
    if hpeaks >= 3: s['4-FSK'] += 1.8*fsk_strength
    if peaks >= 3: s['4-FSK'] += 0.3

    # QAM has a much wider amplitude distribution than PSK/FSK.
    qam = np.clip((acv - 0.18)/0.38, 0, 1)
    s['16-QAM'] += 2.0*qam

    # PSK family: constant envelope and small instantaneous phase variation.
    psk = np.clip((0.28-acv)/0.28, 0, 1) * np.clip((0.30-dstd)/0.30, 0, 1)
    s['BPSK'] += 0.9*psk
    s['QPSK'] += 1.0*psk

    # Normalized 2nd/4th-order cumulants: BPSK ~= (2,2), QPSK ~= (0,1),
    # 16-QAM lies substantially lower and FSK is separated by its phase behavior.
    bpsk = np.exp(-((c20-1.0)/0.28)**2) * np.exp(-((c40-2.0)/0.45)**2)
    qpsk = np.exp(-((c20-0.05)/0.22)**2) * np.exp(-((c40-1.0)/0.35)**2)
    qamc = np.exp(-((c20-0.20)/0.20)**2) * np.exp(-((c40-0.68)/0.35)**2)
    s['BPSK'] += 2.2*bpsk
    s['QPSK'] += 2.2*qpsk
    s['16-QAM'] += 1.5*qamc

    # For ideal PSK, fourth-power phase concentration is high.
    s['QPSK'] += 0.8*np.clip((coh4-0.45)/0.5, 0, 1)
    s['BPSK'] += 0.5*np.clip((coh2-0.45)/0.5, 0, 1)

    # Low-SNR PSK rescue rules. These remain soft evidence and are especially
    # useful when additive noise makes QPSK/BPSK look artificially amplitude-varying.
    qpsk_signature = (
        np.clip((0.16-c20)/0.16, 0, 1) *
        np.clip((c40-0.42)/0.45, 0, 1) *
        np.clip((coh4-0.12)/0.35, 0, 1) *
        np.clip((0.46-acv)/0.30, 0, 1)
    )
    bpsk_signature = (
        np.clip((c20-0.20)/0.55, 0, 1) *
        np.clip((coh2-0.20)/0.45, 0, 1) *
        np.clip((c40-0.45)/0.70, 0, 1)
    )
    s['QPSK'] += 14.0*qpsk_signature
    s['BPSK'] += 25.0*bpsk_signature
    return s


class ModulationClassifier:
    def __init__(self, path):
        self.path = Path(path)
        self.model = None
        self.loaded = False
        if self.path.exists():
            try:
                obj = joblib.load(self.path)
                self.model = obj['model']
                self.loaded = True
            except Exception:
                self.model = None
        if not self.loaded:
            self.model = train(self.path, 300)
            self.loaded = True

    def predict(self, features, signal, fs):
        x = _extract(signal, fs)
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        rf = self.model.predict_proba(x.reshape(1,-1))[0]
        classes = list(self.model.classes_)
        rp = dict(zip(classes, rf))

        expert = _expert_scores(x)
        # Blend learned probabilities with physics-guided evidence.
        # RF gets more weight at moderate/high SNR; expert evidence prevents
        # common PSK/QAM/FSK confusions when the RF is uncertain.
        snr_proxy = np.clip(8 + 12*(1-x[11]), 0, 20)
        alpha = 0.60 if snr_proxy >= 8 else 0.48
        combined = {c: alpha*rp.get(c,0.0) + (1-alpha)*(expert[c]/sum(expert.values())) for c in CLASSES}

        # Temporal voting: classify several non-overlapping chunks and average.
        if len(signal) >= 4096:
            chunk_probs=[]
            step=max(1024, len(signal)//6)
            for start in range(0, len(signal)-2048+1, step):
                q=_extract(signal[start:start+2048], fs)
                q=np.nan_to_num(q, nan=0.0, posinf=1e6, neginf=-1e6)
                pr=self.model.predict_proba(q.reshape(1,-1))[0]
                chunk_probs.append(dict(zip(classes,pr)))
                if len(chunk_probs)>=6: break
            if chunk_probs:
                voted={c:float(np.mean([p.get(c,0) for p in chunk_probs])) for c in CLASSES}
                combined={c:0.70*combined[c]+0.30*voted[c] for c in CLASSES}

        # Final high-confidence physical evidence is applied after chunk voting so a
        # noisy chunk cannot wash out a strong modulation signature from the full file.
        if x[0] > 0.25 and x[8] > 0.25 and x[1] > 0.45:
            combined['BPSK'] += 0.50
        if x[0] < 0.16 and x[1] > 0.42 and x[9] > 0.12 and x[3] < 0.46:
            combined['QPSK'] += 0.25

        total=sum(combined.values()) or 1.0
        combined={k:v/total for k,v in combined.items()}
        order=sorted(CLASSES,key=lambda k:combined[k],reverse=True)
        rows=[{'modulation':k,'probability':round(combined[k]*100,2)} for k in order]
        return {
            'detected': rows[0]['modulation'],
            'confidence': rows[0]['probability'],
            'candidates': rows,
            'features_used': FEATURES,
        }
