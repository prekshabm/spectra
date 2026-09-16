import numpy as np
from backend.dsp.analysis import analyze_signal


def make_qpsk(fs=1_000_000, n_symbols=1000, sps=16, snr_db=10, cfo=100_000):
    rng = np.random.default_rng(7)
    sy = np.exp(1j * (np.pi/4 + rng.integers(0,4,n_symbols) * np.pi/2))
    x = np.repeat(sy, sps)
    t = np.arange(len(x))/fs
    x = x*np.exp(1j*(2*np.pi*cfo*t + 0.3))
    noise_p = np.mean(np.abs(x)**2)/(10**(snr_db/10))
    noise = np.sqrt(noise_p/2)*(rng.normal(size=len(x))+1j*rng.normal(size=len(x)))
    return x+noise


def test_parameters_are_present_and_finite():
    r=analyze_signal(make_qpsk(),1_000_000)
    p=r['parameters']
    for k in ['global_spectral_peak_hz','current_window_peak_hz','occupied_bandwidth_minus10db_hz','rms_amplitude','dc_offset_magnitude','analysis_windows','noise_floor_db_per_hz','spectral_snr_db']:
        assert k in p
        assert np.isfinite(p[k])
    assert r['samples'] == 16000
    assert r['sample_rate'] == 1_000_000
    assert r['bandwidth'] > 0
    assert r['occupied_bandwidth'] > 0


def test_known_qpsk_snr_is_not_time_domain_percentile_failure():
    r=analyze_signal(make_qpsk(snr_db=10),1_000_000)
    # Broad sanity interval: this is not a filename-forced value.
    assert -2.0 < r['snr_db'] < 22.0


def test_estimated_parameter_names_match_dashboard_requirements():
    r = analyze_signal(make_qpsk(), 1_000_000)
    p = r['parameters']
    required = {
        'global_spectral_peak_hz', 'current_window_peak_hz',
        'occupied_bandwidth_minus10db_hz', 'rms_amplitude',
        'dc_offset_magnitude', 'analysis_windows',
        'noise_floor_db_per_hz', 'noise_power_total', 'spectral_snr_db',
    }
    assert required.issubset(p.keys())
