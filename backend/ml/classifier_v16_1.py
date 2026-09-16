from pathlib import Path
import numpy as np
import joblib

from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# SPECTRA V16.1
# Improved modulation feature extraction
# ============================================================

CLASSES = [
    "BPSK",
    "QPSK",
    "2-FSK",
    "4-FSK",
    "16-QAM",
]


FEATURES = [
    "c20_abs",
    "c40_abs",
    "c42_abs",

    "amp_cv",
    "amp_skew",
    "amp_kurt",

    "dphi_std",
    "dphi_kurt",
    "phase_coh2",
    "phase_coh4",

    "freq_std_norm",

    "spectral_flatness",
    "spectral_peak_ratio",
    "spectral_peaks",

    "freq_hist_peaks",
    "freq_hist_entropy",
]


# ============================================================
# HELPERS
# ============================================================

def _normalize(x):
    x = np.asarray(x, dtype=np.complex128)

    x = x[np.isfinite(x)]

    if len(x) == 0:
        return x

    rms = np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12

    return x / rms


def _safe_skew(x):
    if len(x) < 16:
        return 0.0

    v = float(skew(x, bias=False))

    return 0.0 if not np.isfinite(v) else v


def _safe_kurtosis(x):
    if len(x) < 16:
        return 0.0

    v = float(kurtosis(x, fisher=True, bias=False))

    return 0.0 if not np.isfinite(v) else v


def _robust_frequency_states(dphi):
    """
    Estimate the number of stable instantaneous-frequency states.

    Unlike the old implementation, this removes the dominant
    near-zero PSK concentration before looking for separated
    FSK states.
    """

    if len(dphi) < 64:
        return 0.0

    # Convert phase difference into normalized instantaneous
    # frequency.
    freq = dphi / (2.0 * np.pi)

    # Remove extreme outliers.
    lo, hi = np.percentile(freq, [1, 99])

    freq = freq[
        (freq >= lo) &
        (freq <= hi)
    ]

    if len(freq) < 64:
        return 0.0

    # Histogram.
    hist, edges = np.histogram(
        freq,
        bins=96,
        range=(-0.5, 0.5),
        density=False
    )

    hist = hist.astype(float)

    # Smooth histogram.
    kernel = np.ones(5) / 5.0

    smooth = np.convolve(
        hist,
        kernel,
        mode="same"
    )

    if np.max(smooth) <= 0:
        return 0.0

    # Detect significant frequency states.
    prominence = max(
        np.max(smooth) * 0.08,
        2.0
    )

    peaks, props = find_peaks(
        smooth,
        distance=8,
        prominence=prominence
    )

    if len(peaks) == 0:
        return 0.0

    # Only retain peaks carrying meaningful probability mass.
    total = np.sum(smooth) + 1e-12

    valid = []

    for p in peaks:
        left = max(0, p - 3)
        right = min(len(smooth), p + 4)

        mass = np.sum(smooth[left:right]) / total

        if mass >= 0.015:
            valid.append(p)

    return float(min(len(valid), 8))


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def _extract(x, fs):

    z = _normalize(x)

    if len(z) < 64:
        return np.zeros(
            len(FEATURES),
            dtype=float
        )

    # --------------------------------------------------------
    # CFO ESTIMATION
    # --------------------------------------------------------

    nn0 = min(len(z), 32768)

    zz0 = z[:nn0] * np.hanning(nn0)

    candidates = []

    for M in (2, 4):

        pm = (
            np.abs(
                np.fft.fftshift(
                    np.fft.fft(
                        zz0 ** M
                    )
                )
            ) ** 2
            + 1e-18
        )

        fm = np.fft.fftshift(
            np.fft.fftfreq(
                nn0,
                1 / fs
            )
        )

        # Ignore the DC bin when possible.
        k = int(np.argmax(pm))

        f0 = float(
            fm[k] / M
        )

        f0 = (
            (f0 + fs / 2)
            % fs
        ) - fs / 2

        ratio = float(
            pm[k]
            /
            (np.median(pm) + 1e-18)
        )

        candidates.append(
            (ratio, f0)
        )

    _, cfo = max(
        candidates,
        key=lambda q: q[0]
    )

    n = np.arange(len(z))

    zc = (
        z
        *
        np.exp(
            -1j
            *
            2
            *
            np.pi
            *
            cfo
            *
            n
            /
            fs
        )
    )

    # --------------------------------------------------------
    # PHASE DIFFERENCE
    # --------------------------------------------------------

    dphi = np.angle(
        zc[1:]
        *
        np.conj(zc[:-1])
    )

    # --------------------------------------------------------
    # CUMULANTS
    # --------------------------------------------------------

    m20 = np.mean(zc ** 2)
    m40 = np.mean(zc ** 4)

    m42 = np.mean(
        np.abs(zc) ** 4
    )

    m21 = np.mean(
        np.abs(zc) ** 2
    )

    c40 = (
        m40
        -
        3 * m20 * m20
    )

    c42 = (
        m42
        -
        abs(m20) ** 2
        -
        2 * m21 * m21
    )

    # --------------------------------------------------------
    # AMPLITUDE
    # --------------------------------------------------------

    amp = np.abs(zc)

    amp_mean = (
        np.mean(amp)
        + 1e-12
    )

    amp_cv = float(
        np.std(amp)
        /
        amp_mean
    )

    amp_skew = _safe_skew(amp)
    amp_kurt = _safe_kurtosis(amp)

    # --------------------------------------------------------
    # PHASE FEATURES
    # --------------------------------------------------------

    dphi_center = (
        dphi
        -
        np.median(dphi)
    )

    dphi_std = float(
        np.std(dphi_center)
    )

    dphi_kurt = _safe_kurtosis(
        dphi_center
    )

    phase = np.angle(zc)

    phase_coh2 = float(
        abs(
            np.mean(
                np.exp(
                    1j * 2 * phase
                )
            )
        )
    )

    phase_coh4 = float(
        abs(
            np.mean(
                np.exp(
                    1j * 4 * phase
                )
            )
        )
    )

    freq_std_norm = float(
        dphi_std
        /
        np.pi
    )

    # --------------------------------------------------------
    # FSK FREQUENCY FEATURES
    # --------------------------------------------------------

    freq_hist_peaks = (
        _robust_frequency_states(
            dphi
        )
    )

    hist, _ = np.histogram(
        dphi,
        bins=96,
        range=(-np.pi, np.pi),
        density=False
    )

    hist = hist.astype(float)

    # Smooth histogram.
    smooth_hist = np.convolve(
        hist,
        np.ones(5) / 5.0,
        mode="same"
    )

    hp = (
        smooth_hist
        /
        (np.sum(smooth_hist) + 1e-12)
    )

    freq_hist_entropy = float(
        -np.sum(
            hp
            *
            np.log(
                hp + 1e-12
            )
        )
        /
        np.log(
            len(hp)
        )
    )

    # --------------------------------------------------------
    # SPECTRAL FEATURES
    # --------------------------------------------------------

    nn = min(
        len(zc),
        32768
    )

    zz = (
        zc[:nn]
        *
        np.hanning(nn)
    )

    p = (
        np.abs(
            np.fft.fftshift(
                np.fft.fft(zz)
            )
        ) ** 2
        + 1e-18
    )

    p /= (
        np.sum(p)
        + 1e-18
    )

    spectral_flatness = float(
        np.exp(
            np.mean(
                np.log(p)
            )
        )
        /
        (
            np.mean(p)
            + 1e-18
        )
    )

    spectral_peak_ratio = float(
        np.max(p)
        /
        (
            np.median(p)
            + 1e-18
        )
    )

    spectral_peaks_raw, _ = find_peaks(
        p,
        distance=max(
            8,
            nn // 256
        ),
        prominence=max(
            np.max(p) * 0.01,
            1e-12
        )
    )

    spectral_peaks = float(
        min(
            len(spectral_peaks_raw),
            20
        )
    )

    # --------------------------------------------------------
    # FINAL 16 FEATURES
    # --------------------------------------------------------

    features = np.array(
        [
            abs(m20),
            abs(c40),
            abs(c42),

            amp_cv,
            amp_skew,
            amp_kurt,

            dphi_std,
            dphi_kurt,
            phase_coh2,
            phase_coh4,

            freq_std_norm,

            spectral_flatness,
            spectral_peak_ratio,
            spectral_peaks,

            freq_hist_peaks,
            freq_hist_entropy,
        ],
        dtype=float
    )

    return np.nan_to_num(
        features,
        nan=0.0,
        posinf=1e6,
        neginf=-1e6
    )


# ============================================================
# CLASSIFIER
# ============================================================

class ModulationClassifier:

    def __init__(self, path):

        self.path = Path(path)

        self.model = None
        self.loaded = False

        if self.path.exists():

            try:

                obj = joblib.load(
                    self.path
                )

                self.model = obj["model"]

                self.loaded = True

            except Exception:

                self.model = None

        if not self.loaded:

            raise RuntimeError(
                f"Could not load model: {self.path}"
            )

    # --------------------------------------------------------

    def predict(
        self,
        features,
        signal,
        fs
    ):

        x = _extract(
            signal,
            fs
        )

        rf = self.model.predict_proba(
            x.reshape(1, -1)
        )[0]

        classes = list(
            self.model.classes_
        )

        probabilities = dict(
            zip(
                classes,
                rf
            )
        )

        # ----------------------------------------------------
        # Physics evidence
        # ----------------------------------------------------

        c20 = x[0]
        c40 = x[1]
        c42 = x[2]

        amp_cv = x[3]

        dphi_std = x[6]

        coh2 = x[8]
        coh4 = x[9]

        spectral_peaks = x[13]

        fsk_states = x[14]

        fsk_entropy = x[15]

        scores = {
            c: 0.05
            for c in CLASSES
        }

        # ----------------------------------------------------
        # CONSTANT ENVELOPE
        # ----------------------------------------------------

        constant_envelope = np.clip(
            (0.32 - amp_cv)
            /
            0.32,
            0,
            1
        )

        # ----------------------------------------------------
        # FSK DETECTION
        # ----------------------------------------------------

        fsk_activity = np.clip(
            (dphi_std - 0.10)
            /
            0.45,
            0,
            1
        )

        fsk_activity *= (
            0.5
            +
            0.5
            *
            constant_envelope
        )

        # Strong evidence for exact state count.
        if fsk_states <= 2.4:

            scores["2-FSK"] += (
                2.2
                *
                fsk_activity
            )

        if fsk_states >= 3.2:

            scores["4-FSK"] += (
                2.4
                *
                fsk_activity
            )

        # Spectral evidence reinforces FSK.
        if spectral_peaks >= 3:

            scores["4-FSK"] += (
                0.4
                *
                fsk_activity
            )

        # ----------------------------------------------------
        # 16-QAM
        # ----------------------------------------------------

        qam_amplitude = np.clip(
            (amp_cv - 0.16)
            /
            0.35,
            0,
            1
        )

        scores["16-QAM"] += (
            2.2
            *
            qam_amplitude
        )

        # ----------------------------------------------------
        # PSK
        # ----------------------------------------------------

        psk_strength = (
            constant_envelope
            *
            np.clip(
                (0.34 - dphi_std)
                /
                0.34,
                0,
                1
            )
        )

        scores["BPSK"] += (
            0.9
            *
            psk_strength
        )

        scores["QPSK"] += (
            1.1
            *
            psk_strength
        )

        # ----------------------------------------------------
        # BPSK CUMULANT SIGNATURE
        # ----------------------------------------------------

        bpsk_signature = (
            np.exp(
                -(
                    (c20 - 1.0)
                    /
                    0.30
                ) ** 2
            )
            *
            np.exp(
                -(
                    (c40 - 2.0)
                    /
                    0.50
                ) ** 2
            )
            *
            np.clip(
                (coh2 - 0.25)
                /
                0.55,
                0,
                1
            )
        )

        scores["BPSK"] += (
            2.4
            *
            bpsk_signature
        )

        # ----------------------------------------------------
        # QPSK CUMULANT SIGNATURE
        # ----------------------------------------------------

        qpsk_signature = (
            np.exp(
                -(
                    (c20 - 0.05)
                    /
                    0.25
                ) ** 2
            )
            *
            np.exp(
                -(
                    (c40 - 1.0)
                    /
                    0.40
                ) ** 2
            )
            *
            np.clip(
                (coh4 - 0.20)
                /
                0.50,
                0,
                1
            )
        )

        scores["QPSK"] += (
            2.5
            *
            qpsk_signature
        )

        # ----------------------------------------------------
        # QPSK VS QAM
        # ----------------------------------------------------

        if (
            amp_cv < 0.30
            and
            coh4 > 0.35
        ):

            scores["QPSK"] += 0.9

        if amp_cv > 0.30:

            scores["16-QAM"] += 0.7

        # ----------------------------------------------------
        # COMBINE RF + PHYSICS
        # ----------------------------------------------------

        total_scores = (
            sum(scores.values())
            + 1e-12
        )

        physics = {
            c:
            scores[c]
            /
            total_scores
            for c in CLASSES
        }

        # RF remains dominant.
        combined = {
            c:
            0.72 * probabilities.get(c, 0.0)
            +
            0.28 * physics[c]
            for c in CLASSES
        }

        # ----------------------------------------------------
        # TEMPORAL VOTING
        # ----------------------------------------------------

        if len(signal) >= 4096:

            chunk_probs = []

            chunk_size = min(
                4096,
                len(signal)
            )

            step = max(
                2048,
                len(signal) // 6
            )

            for start in range(
                0,
                len(signal)
                -
                chunk_size
                +
                1,
                step
            ):

                chunk = signal[
                    start:
                    start + chunk_size
                ]

                q = _extract(
                    chunk,
                    fs
                )

                pr = self.model.predict_proba(
                    q.reshape(1, -1)
                )[0]

                chunk_probs.append(
                    dict(
                        zip(
                            classes,
                            pr
                        )
                    )
                )

                if len(chunk_probs) >= 6:
                    break

            if chunk_probs:

                voted = {
                    c:
                    float(
                        np.mean(
                            [
                                p.get(
                                    c,
                                    0.0
                                )
                                for p in chunk_probs
                            ]
                        )
                    )
                    for c in CLASSES
                }

                combined = {
                    c:
                    0.80 * combined[c]
                    +
                    0.20 * voted[c]
                    for c in CLASSES
                }

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        total = (
            sum(combined.values())
            +
            1e-12
        )

        combined = {
            c:
            combined[c] / total
            for c in CLASSES
        }

        ordered = sorted(
            CLASSES,
            key=lambda c:
            combined[c],
            reverse=True
        )

        rows = [
            {
                "modulation": c,
                "probability": round(
                    combined[c] * 100,
                    2
                )
            }
            for c in ordered
        ]

        return {
            "detected": rows[0]["modulation"],
            "confidence": rows[0]["probability"],
            "candidates": rows,
            "features_used": FEATURES,
        }