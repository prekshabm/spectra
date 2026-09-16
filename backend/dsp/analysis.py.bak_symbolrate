import numpy as np
from scipy.signal import welch, spectrogram, find_peaks, hilbert


def _cumulants(x):
    z = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)
    m20 = np.mean(z ** 2)
    m21 = np.mean(np.abs(z) ** 2)
    m40 = np.mean(z ** 4)
    m42 = np.mean(np.abs(z) ** 4)
    c40 = m40 - 3 * m20 * m20
    c42 = m42 - abs(m20) ** 2 - 2 * m21 * m21
    return float(abs(c40)), float(abs(c42))


def _robust_noise_floor(psd):
    """Estimate PSD noise density robustly in linear units.

    Uses iterative upper-tail rejection in dB. This prevents strong signal
    bins from biasing the noise estimate while retaining broadband-noise bins.
    """
    p = np.asarray(psd, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size < 32:
        return float(np.median(p)) if p.size else 1e-15

    db = 10.0 * np.log10(np.maximum(p, 1e-30))
    keep = np.ones(db.size, dtype=bool)
    for _ in range(4):
        vals = db[keep]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) + 1e-9
        sigma = 1.4826 * mad
        # 3-sigma rejection, but never less than a 4 dB separation.
        threshold = med + max(4.0, 3.0 * sigma)
        new_keep = db <= threshold
        if new_keep.sum() < max(16, db.size // 10):
            break
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    # Median is robust to residual signal bins.
    return float(10.0 ** (np.median(db[keep]) / 10.0))


def _signal_mask(psd, noise_floor, df):
    """Find occupied spectral regions using a robust noise-relative threshold."""
    p = np.asarray(psd, dtype=float)
    # 6 dB above the estimated floor is the validated-signal threshold.
    threshold = noise_floor * 10.0 ** (6.0 / 10.0)
    mask = p >= threshold

    # Remove isolated one-bin noise excursions.
    if mask.size >= 5:
        core = mask.copy()
        for k in range(1, mask.size - 1):
            if mask[k] and not (mask[k - 1] or mask[k + 1]):
                core[k] = False
        mask = core

    # Close tiny gaps inside a signal region.
    max_gap = max(1, int(round(0.002 * len(mask))))
    if max_gap < len(mask):
        i = 0
        while i < len(mask):
            if mask[i]:
                i += 1
                continue
            j = i
            while j < len(mask) and not mask[j]:
                j += 1
            if i > 0 and j < len(mask) and (j - i) <= max_gap:
                mask[i:j] = True
            i = j

    # Keep contiguous regions with meaningful width; tolerate narrow tones.
    regions = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        if j - i >= 2:
            regions.append((i, j - 1))
        i = j

    return mask, regions


def _quadratic_peak(f, p_db, k):
    if k <= 0 or k >= len(p_db) - 1:
        return float(f[k])
    a, b, c = float(p_db[k - 1]), float(p_db[k]), float(p_db[k + 1])
    den = a - 2.0 * b + c
    if abs(den) < 1e-12:
        return float(f[k])
    delta = np.clip(0.5 * (a - c) / den, -0.5, 0.5)
    df = f[1] - f[0] if len(f) > 1 else 0.0
    return float(f[k] + delta * df)


def _estimate_symbol_rate(x, fs):
    """Return a conservative symbol-rate candidate from envelope periodicity."""
    n = len(x)
    if n < 256:
        return None
    amp = np.abs(x)
    amp = amp - np.mean(amp)
    # Downsample only for autocorrelation speed, while retaining periodicity.
    max_points = 50000
    step = max(1, n // max_points)
    a = amp[::step]
    if len(a) < 128:
        return None
    ac = np.correlate(a, a, mode='full')[len(a) - 1:]
    if ac[0] <= 1e-12:
        return None
    ac[:2] = 0
    # Candidate lags corresponding to 2..200 samples/symbol.
    max_lag = min(len(ac) - 1, max(4, int(fs / max(1000.0, fs / 2.0))))
    search = ac[1:min(len(ac), max_lag + 1)]
    if len(search) < 4:
        return None
    peaks, props = find_peaks(search, distance=2, prominence=max(ac[0] * 0.01, 1e-12))
    if len(peaks) == 0:
        return None
    # Prefer the first strong periodic peak; return as a candidate, not a claim.
    best = peaks[np.argmax(search[peaks])]
    lag = float(best + 1) * step
    if lag <= 0:
        return None
    rate = fs / lag
    if not (1e3 <= rate <= fs / 2.0):
        return None
    return float(rate)


def analyze_signal(x, fs):
    x = np.asarray(x, dtype=np.complex64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 32:
        raise ValueError('Signal is too short for DSP analysis')
    fs = float(fs)

    # Raw-domain measurements. Do not normalize here; these values should
    # describe the actual uploaded signal.
    dc = np.mean(x)
    x0 = x - dc
    rms = float(np.sqrt(np.mean(np.abs(x0) ** 2)))
    peak_amp = float(np.max(np.abs(x0)))
    signal_power = float(np.mean(np.abs(x0) ** 2))

    # Welch PSD: substantially more stable than a single FFT for noise/SNR.
    nperseg = min(4096, max(256, 2 ** int(np.floor(np.log2(min(n, 4096))))))
    if nperseg > n:
        nperseg = n
    noverlap = nperseg // 2
    f, p = welch(
        x0,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        scaling='density',
    )
    order = np.argsort(f)
    f = f[order]
    p = np.maximum(p[order], 1e-30)
    p_db = 10.0 * np.log10(p)
    df = float(np.median(np.diff(f))) if len(f) > 1 else fs / nperseg

    # Robust noise floor and validated signal regions.
    noise_density = _robust_noise_floor(p)
    mask, regions = _signal_mask(p, noise_density, df)

    # If the signal occupies almost everything, use the lower spectral tail
    # for noise and report the SNR as an estimate rather than forcing a region.
    if mask.mean() > 0.75:
        q = np.percentile(p, 20)
        noise_density = min(noise_density, float(q))
        mask = p >= noise_density * 10.0 ** (6.0 / 10.0)
        _, regions = _signal_mask(p, noise_density, df)

    # Global peak.
    peak_idx = int(np.argmax(p))
    global_peak = _quadratic_peak(f, p_db, peak_idx)

    # Dominant validated region and its current-window peak.
    if regions:
        region_scores = []
        for lo, hi in regions:
            score = float(np.sum(np.maximum(p[lo:hi + 1] - noise_density, 0.0)))
            region_scores.append((score, lo, hi))
        _, dlo, dhi = max(region_scores, key=lambda q: q[0])
    else:
        dlo = max(0, peak_idx - 1)
        dhi = min(len(p) - 1, peak_idx + 1)

    local_idx = dlo + int(np.argmax(p[dlo:dhi + 1]))
    current_window_peak = _quadratic_peak(f, p_db, local_idx)

    # -10 dB bandwidth around the dominant peak.
    peak_power = float(p[peak_idx])
    threshold10 = peak_power * 0.1
    lo10 = peak_idx
    hi10 = peak_idx
    while lo10 > 0 and p[lo10 - 1] >= threshold10:
        lo10 -= 1
    while hi10 < len(p) - 1 and p[hi10 + 1] >= threshold10:
        hi10 += 1
    bw10 = float(max(0.0, f[hi10] - f[lo10]))

    # 99% validated signal-power bandwidth. Integrate only above the noise floor.
    excess = np.maximum(p - noise_density, 0.0) * mask
    total_excess = float(np.sum(excess))
    if total_excess > 0:
        cdf = np.cumsum(excess) / total_excess
        obw_lo_idx = int(np.searchsorted(cdf, 0.005))
        obw_hi_idx = int(np.searchsorted(cdf, 0.995))
        obw_lo_idx = min(max(obw_lo_idx, 0), len(f) - 1)
        obw_hi_idx = min(max(obw_hi_idx, obw_lo_idx), len(f) - 1)
        obw99 = float(max(0.0, f[obw_hi_idx] - f[obw_lo_idx]))
    else:
        obw_lo_idx = obw_hi_idx = peak_idx
        obw99 = 0.0

    # Noise power is estimated over the full analyzed bandwidth outside the
    # validated signal region. This matches the conventional total-power SNR
    # used by labelled synthetic files such as *_10dB, rather than reporting
    # only an in-band SNR.
    occupied_bins = int(np.count_nonzero(mask))
    total_bins = max(len(p), 1)
    noise_bins = max(total_bins - occupied_bins, 1)
    noise_power_total = float(noise_density * noise_bins * max(df, 1.0))
    psd_total_power = float(np.sum(p) * max(df, 1.0))
    signal_power_spectral = max(psd_total_power - noise_power_total, 1e-30)
    snr_db = float(10.0 * np.log10(signal_power_spectral / max(noise_power_total, 1e-30)))

    # Also retain in-band SNR as a diagnostic. It is normally higher than total
    # SNR because the same noise density is integrated over a narrower band.
    inband_noise_power = float(noise_density * max(occupied_bins, 1) * max(df, 1.0))
    inband_signal_power = float(np.sum(np.maximum(p[mask] - noise_density, 0.0)) * max(df, 1.0))
    inband_snr_db = float(10.0 * np.log10(max(inband_signal_power, 1e-30) / max(inband_noise_power, 1e-30)))

    noise_rms = float(np.sqrt(max(noise_power_total, 1e-30)))
    noise_floor_db = float(10.0 * np.log10(max(noise_density, 1e-30)))
    peak_to_floor_db = float(10.0 * np.log10(max(peak_power, 1e-30) / max(noise_density, 1e-30)))

    # Estimate a center/frequency-offset candidate from the validated band.
    if regions:
        weighted = np.maximum(p[mask] - noise_density, 0.0)
        center_frequency = float(np.sum(f[mask] * weighted) / (np.sum(weighted) + 1e-30))
    else:
        center_frequency = global_peak
    frequency_offset = center_frequency

    # Tracking across STFT windows for the current-window peak parameter.
    # Use spectrogram only for visualization/diagnostic tracking.
    sf, st, S = spectrogram(
        x0,
        fs=fs,
        nperseg=min(1024, n),
        noverlap=min(768, max(0, min(1024, n) - 1)),
        return_onesided=False,
        mode='psd',
    )
    S = np.fft.fftshift(S, axes=0)
    sf = np.fft.fftshift(sf)
    track = []
    valid_count = 0
    strongest_frame_peak = current_window_peak
    strongest_frame_power = -np.inf
    if S.size:
        for j in range(S.shape[1]):
            row = np.maximum(S[:, j], 1e-30)
            k = int(np.argmax(row))
            row_noise = _robust_noise_floor(row)
            prom = 10.0 * np.log10(row[k] / max(row_noise, 1e-30))
            if prom >= 6.0:
                track.append(float(_quadratic_peak(sf, 10.0*np.log10(row), k)))
                valid_count += 1
                if row[k] > strongest_frame_power:
                    strongest_frame_power = float(row[k])
                    strongest_frame_peak = float(_quadratic_peak(sf, 10.0*np.log10(row), k))
    if track:
        track_min = float(np.min(track))
        track_max = float(np.max(track))
        avg_track = float(np.mean(track))
    else:
        track_min = track_max = avg_track = current_window_peak

    c40, c42 = _cumulants(x0)
    amp = np.abs(x0)
    amp_mean = float(np.mean(amp))
    amp_std = float(np.std(amp))
    amp_cv = amp_std / (amp_mean + 1e-12)
    phase = np.unwrap(np.angle(x0))
    dphi = np.angle(x0[1:] * np.conj(x0[:-1]))
    freq_inst = np.diff(phase) * fs / (2.0 * np.pi)
    freq_std = float(np.std(freq_inst)) if len(freq_inst) else 0.0
    sym_rate = _estimate_symbol_rate(x0, fs)

    # Visualization payloads are downsampled.
    keep = min(n, 20000)
    idx = np.linspace(0, n - 1, keep).astype(int)
    z = x0[idx]
    if S.shape[1] > 180:
        step_t = max(1, S.shape[1] // 180)
        S = S[:, ::step_t]
        st = st[::step_t]
    if S.shape[0] > 300:
        step_f = max(1, S.shape[0] // 300)
        S = S[::step_f, :]
        sf = sf[::step_f]

    parameters = {
        'global_spectral_peak_hz': global_peak,
        'current_window_peak_hz': strongest_frame_peak,
        'occupied_bandwidth_minus10db_hz': bw10,
        'validated_occupied_bandwidth_99pct_hz': obw99,
        'occupied_band_start_hz': float(f[obw_lo_idx]),
        'occupied_band_end_hz': float(f[obw_hi_idx]),
        'rms_amplitude': rms,
        'dc_offset_magnitude': float(abs(dc)),
        'dc_offset_i': float(dc.real),
        'dc_offset_q': float(dc.imag),
        'peak_amplitude': peak_amp,
        'signal_power': signal_power,
        'noise_floor_db_per_hz': noise_floor_db,
        'noise_rms_equivalent': noise_rms,
        'spectral_snr_db': snr_db,
        'inband_snr_db': inband_snr_db,
        'noise_power_total': noise_power_total,
        'peak_to_noise_floor_db': peak_to_floor_db,
        'analysis_windows': int(S.shape[1]),
        'dominant_tracking_min_hz': track_min,
        'dominant_tracking_max_hz': track_max,
        'tracking_coverage_percent': float(100.0 * valid_count / max(S.shape[1], 1)),
        'center_frequency_hz': center_frequency,
        'frequency_offset_hz': frequency_offset,
        'symbol_rate_candidate_sym_s': sym_rate,
        'psd_resolution_hz': abs(df),
        'validated_signal_regions': len(regions),
    }

    return {
        'samples': n,
        'sample_rate': fs,
        'duration': n / fs,
        'dominant_frequency': global_peak,
        'current_window_peak': strongest_frame_peak,
        'bandwidth': bw10,
        'occupied_bandwidth': obw99,
        'snr_db': snr_db,
        'signal_power': signal_power,
        'noise_floor_db': noise_floor_db,
        'noise_power': noise_power_total,
        'inband_snr_db': inband_snr_db,
        'frequency_offset': frequency_offset,
        'symbol_rate': sym_rate,
        'analysis_windows': int(S.shape[1]),
        'parameters': parameters,
        # Keep the classifier contract unchanged. These are internal ML inputs,
        # not a UI "Extracted Features" block.
        'features': {
            'c40': c40,
            'c42': c42,
            'amp_cv': amp_cv,
            'amp_mean': amp_mean,
            'amp_std': amp_std,
            'inst_freq_std': freq_std,
            'bandwidth': bw10,
            'symbol_rate': sym_rate,
        },
        'spectrum': {
            'frequency': f.tolist(),
            'power_db': p_db.tolist(),
        },
        'constellation': {
            'i': z.real.tolist(),
            'q': z.imag.tolist(),
        },
        'waterfall': {
            'frequency': sf.tolist(),
            'time': st.tolist(),
            'power_db': (10.0 * np.log10(np.maximum(S, 1e-30))).T.tolist(),
        },
    }
