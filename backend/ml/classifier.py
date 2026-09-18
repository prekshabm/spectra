from pathlib import Path
import numpy as np
import joblib

from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier

from backend.ml.fsk_specialist import extract_fsk_features


# ============================================================
# MODULATION CLASSES
# ============================================================

CLASSES = [
    "BPSK",
    "QPSK",
    "2-FSK",
    "4-FSK",
    "16-QAM",
]


# ============================================================
# MAIN FEATURE SET
# ============================================================

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
# NORMALIZATION
# ============================================================

def _normalize(x):
    x = np.asarray(
        x,
        dtype=np.complex128
    )

    x = x[np.isfinite(x)]

    if len(x) == 0:
        return x

    power = np.sqrt(
        np.mean(
            np.abs(x) ** 2
        )
    )

    return x / (power + 1e-12)


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
    # Estimate carrier frequency offset
    # --------------------------------------------------------

    nn0 = min(
        len(z),
        32768
    )

    zz0 = (
        z[:nn0]
        * np.hanning(nn0)
    )

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

        k = int(
            np.argmax(pm)
        )

        f0 = float(
            fm[k] / M
        )

        f0 = (
            (f0 + fs / 2)
            % fs
        ) - fs / 2

        ratio = float(
            pm[k]
            / (
                np.median(pm)
                + 1e-18
            )
        )

        candidates.append(
            (
                ratio,
                f0
            )
        )

    _, cfo = max(
        candidates,
        key=lambda q: q[0]
    )

    n = np.arange(
        len(z)
    )

    zc = (
        z
        * np.exp(
            -1j
            * 2
            * np.pi
            * cfo
            * n
            / fs
        )
    )

    # --------------------------------------------------------
    # Instantaneous phase difference
    # --------------------------------------------------------

    dphi = np.angle(
        zc[1:]
        * np.conj(
            zc[:-1]
        )
    )

    # --------------------------------------------------------
    # Higher-order moments
    # --------------------------------------------------------

    m20 = np.mean(
        zc ** 2
    )

    m40 = np.mean(
        zc ** 4
    )

    m42 = np.mean(
        np.abs(zc) ** 4
    )

    m21 = np.mean(
        np.abs(zc) ** 2
    )

    c40 = (
        m40
        - 3 * m20 * m20
    )

    c42 = (
        m42
        - abs(m20) ** 2
        - 2 * m21 * m21
    )

    # --------------------------------------------------------
    # Amplitude features
    # --------------------------------------------------------

    amp = np.abs(zc)

    acv = float(
        np.std(amp)
        / (
            np.mean(amp)
            + 1e-12
        )
    )

    if len(amp) > 8:

        askew = float(
            skew(
                amp,
                bias=False
            )
        )

        akurt = float(
            kurtosis(
                amp,
                fisher=True,
                bias=False
            )
        )

    else:

        askew = 0.0
        akurt = 0.0

    # --------------------------------------------------------
    # Phase statistics
    # --------------------------------------------------------

    dphi_center = (
        dphi
        - np.median(dphi)
    )

    dstd = float(
        np.std(
            dphi_center
        )
    )

    if len(dphi_center) > 8:

        dkurt = float(
            kurtosis(
                dphi_center,
                fisher=True,
                bias=False
            )
        )

    else:

        dkurt = 0.0

    phase = np.angle(zc)

    coh2 = float(
        abs(
            np.mean(
                np.exp(
                    1j
                    * 2
                    * phase
                )
            )
        )
    )

    coh4 = float(
        abs(
            np.mean(
                np.exp(
                    1j
                    * 4
                    * phase
                )
            )
        )
    )

    # --------------------------------------------------------
    # Frequency spread
    # --------------------------------------------------------

    freq_std_norm = float(
        dstd
        / (2 * np.pi)
        * fs
        / (
            fs / 2
            + 1e-12
        )
    )

    # --------------------------------------------------------
    # Histogram of instantaneous frequency
    # --------------------------------------------------------

    hist, _ = np.histogram(
        dphi,
        bins=64,
        range=(-np.pi, np.pi),
        density=True
    )

    hs = np.convolve(
        hist,
        np.ones(3) / 3,
        mode="same"
    )

    if len(hs) > 0:

        prominence = max(
            np.max(hs) * 0.06,
            1e-3
        )

    else:

        prominence = 1e-3

    hpk, _ = find_peaks(
        hs,
        distance=5,
        prominence=prominence
    )

    freq_hist_peaks = float(
        min(
            len(hpk),
            8
        )
    )

    hp = (
        hs
        / (
            np.sum(hs)
            + 1e-12
        )
    )

    freq_hist_entropy = float(
        -np.sum(
            hp
            * np.log(
                hp
                + 1e-12
            )
        )
        / np.log(
            len(hp)
            if len(hp) > 1
            else 2
        )
    )

    # --------------------------------------------------------
    # Spectrum
    # --------------------------------------------------------

    nn = min(
        len(zc),
        32768
    )

    zz = (
        zc[:nn]
        * np.hanning(nn)
    )

    p = (
        np.abs(
            np.fft.fftshift(
                np.fft.fft(
                    zz
                )
            )
        ) ** 2
        + 1e-18
    )

    p = (
        p
        / np.sum(p)
    )

    spectral_flatness = float(
        np.exp(
            np.mean(
                np.log(p)
            )
        )
        / (
            np.mean(p)
            + 1e-18
        )
    )

    spectral_peak_ratio = float(
        np.max(p)
        / (
            np.median(p)
            + 1e-18
        )
    )

    pk, _ = find_peaks(
        p,
        distance=max(
            4,
            nn // 256
        ),
        prominence=np.max(p) * 0.015
    )

    spectral_peaks = float(
        min(
            len(pk),
            20
        )
    )

    # --------------------------------------------------------
    # Return feature vector
    # --------------------------------------------------------

    return np.array(
        [
            abs(m20),
            abs(c40),
            abs(c42),
            acv,
            askew,
            akurt,
            dstd,
            dkurt,
            coh2,
            coh4,
            freq_std_norm,
            spectral_flatness,
            spectral_peak_ratio,
            spectral_peaks,
            freq_hist_peaks,
            freq_hist_entropy,
        ],
        dtype=float
    )


# ============================================================
# SYNTHETIC TRAINING SIGNAL
# ============================================================

def _signal(
    mod,
    fs=1e6,
    n_symbols=300,
    sps=16,
    snr=18,
    rng=None
):

    rng = (
        rng
        if rng is not None
        else np.random.default_rng()
    )

    n = (
        n_symbols
        * sps
    )

    t = (
        np.arange(n)
        / fs
    )

    # ========================================================
    # BPSK
    # ========================================================

    if mod == "BPSK":

        symbols = rng.integers(
            0,
            2,
            n_symbols
        )

        sy = (
            2 * symbols
            - 1
        )

        base = np.repeat(
            sy,
            sps
        ).astype(
            complex
        )

    # ========================================================
    # QPSK
    # ========================================================

    elif mod == "QPSK":

        symbols = rng.integers(
            0,
            4,
            n_symbols
        )

        sy = np.exp(
            1j
            * (
                np.pi / 4
                + symbols
                * np.pi / 2
            )
        )

        base = np.repeat(
            sy,
            sps
        )

    # ========================================================
    # 16-QAM
    # ========================================================

    elif mod == "16-QAM":

        symbols = rng.integers(
            0,
            16,
            n_symbols
        )

        I = (
            2 * (symbols % 4)
            - 3
        )

        Q = (
            2 * (symbols // 4)
            - 3
        )

        sy = (
            I
            + 1j * Q
        ) / np.sqrt(10)

        base = np.repeat(
            sy,
            sps
        )

    # ========================================================
    # FSK
    # ========================================================

    else:

        m = (
            2
            if mod == "2-FSK"
            else 4
        )

        symbols = rng.integers(
            0,
            m,
            n_symbols
        )

        tones = (
            np.arange(m)
            - (m - 1) / 2
        )

        spacing = (
            rng.uniform(
                0.015,
                0.06
            )
            * fs
        )

        fseq = (
            np.repeat(
                tones[symbols],
                sps
            )
            * spacing
        )

        phase = np.cumsum(
            2
            * np.pi
            * fseq
            / fs
        )

        base = np.exp(
            1j
            * phase
        )

        if rng.random() < 0.4:

            base *= np.exp(
                1j
                * np.cumsum(
                    rng.normal(
                        0,
                        0.002,
                        n
                    )
                )
            )

    # ========================================================
    # Random carrier offset
    # ========================================================

    cfo = (
        rng.uniform(
            -0.18,
            0.18
        )
        * fs
    )

    phase0 = rng.uniform(
        -np.pi,
        np.pi
    )

    base *= np.exp(
        1j
        * (
            2
            * np.pi
            * cfo
            * t
            + phase0
        )
    )

    # ========================================================
    # Amplitude scaling
    # ========================================================

    base *= rng.uniform(
        0.7,
        1.3
    )

    # ========================================================
    # Mild amplitude fading
    # ========================================================

    if rng.random() < 0.25:

        fade = (
            1
            + rng.normal(
                0,
                0.025,
                n
            )
        )

        base *= fade

    # ========================================================
    # AWGN
    # ========================================================

    power = np.mean(
        np.abs(base) ** 2
    )

    noise_power = (
        power
        / (
            10
            ** (
                snr / 10
            )
        )
    )

    noise = (
        np.sqrt(
            noise_power / 2
        )
        * (
            rng.normal(
                size=n
            )
            + 1j
            * rng.normal(
                size=n
            )
        )
    )

    return base + noise


# ============================================================
# TRAIN MAIN RANDOM FOREST
# ============================================================

def train(
    path,
    n_per_class=300
):

    rng = np.random.default_rng(
        42
    )

    X = []
    y = []

    for modulation in CLASSES:

        for _ in range(
            n_per_class
        ):

            fs = 1e6

            sps = int(
                rng.choice(
                    [
                        8,
                        10,
                        12,
                        16,
                        20,
                        24,
                        32
                    ]
                )
            )

            snr = float(
                rng.uniform(
                    4,
                    26
                )
            )

            n_symbols = int(
                rng.choice(
                    [
                        180,
                        240,
                        300,
                        400
                    ]
                )
            )

            sig = _signal(
                modulation,
                fs=fs,
                n_symbols=n_symbols,
                sps=sps,
                snr=snr,
                rng=rng
            )

            feats = _extract(
                sig,
                fs
            )

            X.append(
                feats
            )

            y.append(
                modulation
            )

    X = np.nan_to_num(
        np.asarray(X),
        nan=0.0,
        posinf=1e6,
        neginf=-1e6
    )

    clf = RandomForestClassifier(
        n_estimators=350,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    clf.fit(
        X,
        y
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": clf,
            "features": FEATURES,
            "classes": CLASSES
        },
        path
    )

    return clf


# ============================================================
# PHYSICS / SIGNAL EXPERT
# ============================================================

def _expert_scores(f):

    (
        c20,
        c40,
        c42,
        acv,
        askew,
        akurt,
        dstd,
        dkurt,
        coh2,
        coh4,
        fstd,
        flat,
        pratio,
        peaks,
        hpeaks,
        hent
    ) = f

    scores = {
        c: 0.05
        for c in CLASSES
    }

    # ========================================================
    # FSK evidence
    # ========================================================

    fsk_strength = np.clip(
        (
            dstd
            - 0.12
        )
        / 0.50,
        0,
        1
    )

    fsk_strength *= np.clip(
        (
            0.65
            - acv
        )
        / 0.65,
        0,
        1
    )

    fsk_strength *= np.clip(
        (
            0.95
            - hent
        )
        / 0.95,
        0,
        1
    )

    scores["2-FSK"] += (
        1.2
        * fsk_strength
    )

    scores["4-FSK"] += (
        1.2
        * fsk_strength
    )

    if hpeaks == 2:

        scores["2-FSK"] += (
            1.8
            * fsk_strength
        )

    if hpeaks >= 3:

        scores["4-FSK"] += (
            1.8
            * fsk_strength
        )

    if peaks >= 3:

        scores["4-FSK"] += 0.3

    # ========================================================
    # QAM evidence
    # ========================================================

    qam = np.clip(
        (
            acv
            - 0.18
        )
        / 0.38,
        0,
        1
    )

    scores["16-QAM"] += (
        2.0
        * qam
    )

    # ========================================================
    # PSK evidence
    # ========================================================

    psk = (
        np.clip(
            (
                0.28
                - acv
            )
            / 0.28,
            0,
            1
        )
        *
        np.clip(
            (
                0.30
                - dstd
            )
            / 0.30,
            0,
            1
        )
    )

    scores["BPSK"] += (
        0.9
        * psk
    )

    scores["QPSK"] += (
        1.0
        * psk
    )

    # ========================================================
    # Cumulant evidence
    # ========================================================

    bpsk = (
        np.exp(
            -(
                (
                    c20
                    - 1.0
                )
                / 0.28
            ) ** 2
        )
        *
        np.exp(
            -(
                (
                    c40
                    - 2.0
                )
                / 0.45
            ) ** 2
        )
    )

    qpsk = (
        np.exp(
            -(
                (
                    c20
                    - 0.05
                )
                / 0.22
            ) ** 2
        )
        *
        np.exp(
            -(
                (
                    c40
                    - 1.0
                )
                / 0.35
            ) ** 2
        )
    )

    qamc = (
        np.exp(
            -(
                (
                    c20
                    - 0.20
                )
                / 0.20
            ) ** 2
        )
        *
        np.exp(
            -(
                (
                    c40
                    - 0.68
                )
                / 0.35
            ) ** 2
        )
    )

    scores["BPSK"] += (
        2.2
        * bpsk
    )

    scores["QPSK"] += (
        2.2
        * qpsk
    )

    scores["16-QAM"] += (
        1.5
        * qamc
    )

    # ========================================================
    # Phase concentration
    # ========================================================

    scores["QPSK"] += (
        0.8
        * np.clip(
            (
                coh4
                - 0.45
            )
            / 0.5,
            0,
            1
        )
    )

    scores["BPSK"] += (
        0.5
        * np.clip(
            (
                coh2
                - 0.45
            )
            / 0.5,
            0,
            1
        )
    )

    # ========================================================
    # QPSK low-SNR signature
    # ========================================================

    qpsk_signature = (
        np.clip(
            (
                0.16
                - c20
            )
            / 0.16,
            0,
            1
        )
        *
        np.clip(
            (
                c40
                - 0.42
            )
            / 0.45,
            0,
            1
        )
        *
        np.clip(
            (
                coh4
                - 0.12
            )
            / 0.35,
            0,
            1
        )
        *
        np.clip(
            (
                0.46
                - acv
            )
            / 0.30,
            0,
            1
        )
    )

    bpsk_signature = (
        np.clip(
            (
                c20
                - 0.20
            )
            / 0.55,
            0,
            1
        )
        *
        np.clip(
            (
                coh2
                - 0.20
            )
            / 0.45,
            0,
            1
        )
        *
        np.clip(
            (
                c40
                - 0.45
            )
            / 0.70,
            0,
            1
        )
    )

    scores["QPSK"] += (
        14.0
        * qpsk_signature
    )

    scores["BPSK"] += (
        25.0
        * bpsk_signature
    )

    return scores


# ============================================================
# MODULATION CLASSIFIER
# ============================================================

class ModulationClassifier:

    def __init__(
        self,
        path
    ):

        self.path = Path(
            path
        )

        # V18 is the currently selected real-data model.
        # For V18 we trust the trained Random Forest directly.
        self.v18_primary = (
            self.path.name
            == "modulation_rf_v18_all_training.joblib"
        )

        self.model = None
        self.loaded = False

        # ====================================================
        # SPECIALISTS
        # ====================================================
        #
        # They are only loaded for the legacy path.
        # V18 does not use them during prediction.
        # This keeps V18 classification lighter/faster.
        # ====================================================

        self.qpsk_qam_specialist = None
        self.fsk_specialist = None

        if not self.v18_primary:

            # ------------------------------------------------
            # QPSK / 16-QAM specialist
            # ------------------------------------------------

            specialist_path = (
                self.path.parent
                / "specialist_qpsk_qam.joblib"
            )

            if specialist_path.exists():

                try:

                    specialist_obj = joblib.load(
                        specialist_path
                    )

                    self.qpsk_qam_specialist = (
                        specialist_obj["model"]
                    )

                except Exception:

                    self.qpsk_qam_specialist = None

            # ------------------------------------------------
            # FSK specialist
            # ------------------------------------------------

            fsk_specialist_path = (
                self.path.parent
                / "specialist_fsk_real.joblib"
            )

            if fsk_specialist_path.exists():

                try:

                    fsk_obj = joblib.load(
                        fsk_specialist_path
                    )

                    self.fsk_specialist = (
                        fsk_obj["model"]
                    )

                except Exception:

                    self.fsk_specialist = None

        # ====================================================
        # MAIN MODEL
        # ====================================================

        if self.path.exists():

            try:

                obj = joblib.load(
                    self.path
                )

                self.model = (
                    obj["model"]
                )

                self.loaded = True

            except Exception:

                self.model = None
                self.loaded = False

        # ====================================================
        # TRAIN IF MODEL DOES NOT EXIST
        # ====================================================

        if not self.loaded:

            self.model = train(
                self.path,
                300
            )

            self.loaded = True

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        features,
        signal,
        fs
    ):

        # ----------------------------------------------------
        # Main feature extraction
        # ----------------------------------------------------

        x = _extract(
            signal,
            fs
        )

        x = np.nan_to_num(
            x,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6
        )

        # ----------------------------------------------------
        # Random Forest probabilities
        # ----------------------------------------------------

        rf = self.model.predict_proba(
            x.reshape(
                1,
                -1
            )
        )[0]

        classes = list(
            self.model.classes_
        )

        rf_scores = dict(
            zip(
                classes,
                rf
            )
        )

        # ====================================================
        # V18 PRIMARY PATH
        # ====================================================
        #
        # The V18 model was independently tested on:
        #
        # validation_50 : 50/50
        # validation    : 25/25
        #
        # So for V18 we use the trained Random Forest
        # probabilities directly.
        #
        # This prevents the older expert/voting/specialist
        # layers from changing an already validated model
        # prediction.
        #
        # It also reduces classification computation.
        # ====================================================

        if self.v18_primary:

            order = sorted(
                CLASSES,
                key=lambda modulation:
                    rf_scores.get(
                        modulation,
                        0.0
                    ),
                reverse=True
            )

            rows = [
                {
                    "modulation": modulation,
                    "probability":
                        round(
                            rf_scores.get(
                                modulation,
                                0.0
                            )
                            * 100,
                            2
                        )
                }
                for modulation in order
            ]

            return {
                "detected":
                    rows[0]["modulation"],

                "confidence":
                    rows[0]["probability"],

                "candidates":
                    rows,

                "features_used":
                    FEATURES,
            }

        # ====================================================
        # LEGACY MODEL PATH
        # ====================================================
        #
        # The following preserves the previous behaviour for
        # models other than V18.
        # ====================================================

        # ----------------------------------------------------
        # Physics expert probabilities
        # ----------------------------------------------------

        expert = _expert_scores(
            x
        )

        expert_total = (
            sum(
                expert.values()
            )
            or 1.0
        )

        # ----------------------------------------------------
        # RF + expert blend
        # ----------------------------------------------------

        snr_proxy = np.clip(
            8
            + 12
            * (
                1
                - x[11]
            ),
            0,
            20
        )

        alpha = (
            0.60
            if snr_proxy >= 8
            else 0.48
        )

        combined = {
            c:
                alpha
                * rf_scores.get(
                    c,
                    0.0
                )
                +
                (
                    1
                    - alpha
                )
                * (
                    expert[c]
                    / expert_total
                )
            for c in CLASSES
        }

        # ====================================================
        # TEMPORAL VOTING
        # ====================================================

        if len(signal) >= 4096:

            chunk_probs = []

            step = max(
                1024,
                len(signal) // 6
            )

            for start in range(
                0,
                len(signal) - 2048 + 1,
                step
            ):

                q = _extract(
                    signal[
                        start:
                        start + 2048
                    ],
                    fs
                )

                q = np.nan_to_num(
                    q,
                    nan=0.0,
                    posinf=1e6,
                    neginf=-1e6
                )

                pr = (
                    self.model
                    .predict_proba(
                        q.reshape(
                            1,
                            -1
                        )
                    )[0]
                )

                chunk_probs.append(
                    dict(
                        zip(
                            classes,
                            pr
                        )
                    )
                )

                if len(
                    chunk_probs
                ) >= 6:

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
                        0.70
                        * combined[c]
                        +
                        0.30
                        * voted[c]
                    for c in CLASSES
                }

        # ====================================================
        # BPSK RULE
        # ====================================================

        if (
            x[0] > 0.25
            and x[8] > 0.25
            and x[1] > 0.45
        ):

            combined["BPSK"] += 0.50

        # ====================================================
        # QPSK RULE
        # ====================================================

        if (
            x[0] < 0.16
            and x[1] > 0.42
            and x[9] > 0.12
            and x[3] < 0.46
        ):

            combined["QPSK"] += 0.25

        # ====================================================
        # QPSK / 16-QAM SPECIALIST
        # ====================================================

        top_general = max(
            combined,
            key=combined.get
        )

        if (
            self.qpsk_qam_specialist
            is not None
            and top_general
            in (
                "QPSK",
                "16-QAM"
            )
        ):

            specialist_prob = (
                self.qpsk_qam_specialist
                .predict_proba(
                    x.reshape(
                        1,
                        -1
                    )
                )[0]
            )

            specialist_classes = list(
                self.qpsk_qam_specialist
                .classes_
            )

            specialist_scores = dict(
                zip(
                    specialist_classes,
                    specialist_prob
                )
            )

            specialist_prediction = max(
                specialist_scores,
                key=specialist_scores.get
            )

            if specialist_prediction in (
                "QPSK",
                "16-QAM"
            ):

                pair_total = (
                    combined.get(
                        "QPSK",
                        0.0
                    )
                    +
                    combined.get(
                        "16-QAM",
                        0.0
                    )
                )

                combined["QPSK"] = (
                    pair_total
                    * specialist_scores.get(
                        "QPSK",
                        0.0
                    )
                )

                combined["16-QAM"] = (
                    pair_total
                    * specialist_scores.get(
                        "16-QAM",
                        0.0
                    )
                )

        # ====================================================
        # FSK SPECIALIST
        # ====================================================

        if self.fsk_specialist is not None:

            try:

                fsk_x = extract_fsk_features(
                    signal,
                    fs
                )

                fsk_x = np.nan_to_num(
                    fsk_x,
                    nan=0.0,
                    posinf=1e6,
                    neginf=-1e6
                )

                fsk_prob = (
                    self.fsk_specialist
                    .predict_proba(
                        fsk_x.reshape(
                            1,
                            -1
                        )
                    )[0]
                )

                fsk_classes = list(
                    self.fsk_specialist
                    .classes_
                )

                fsk_scores = dict(
                    zip(
                        fsk_classes,
                        fsk_prob
                    )
                )

                fsk_2 = float(
                    fsk_scores.get(
                        "2-FSK",
                        0.0
                    )
                )

                fsk_4 = float(
                    fsk_scores.get(
                        "4-FSK",
                        0.0
                    )
                )

                fsk_best = max(
                    fsk_2,
                    fsk_4
                )

                fsk_label = (
                    "4-FSK"
                    if fsk_4 >= fsk_2
                    else "2-FSK"
                )

                # ------------------------------------------------
                # Main-model FSK evidence
                # ------------------------------------------------

                fsk_evidence = (
                    np.clip(
                        (
                            x[6]
                            - 0.12
                        )
                        / 0.45,
                        0.0,
                        1.0
                    )
                    +
                    np.clip(
                        (
                            x[14]
                            - 1.3
                        )
                        / 3.0,
                        0.0,
                        1.0
                    )
                    +
                    np.clip(
                        (
                            0.98
                            - x[15]
                        )
                        / 0.55,
                        0.0,
                        1.0
                    )
                ) / 3.0

                # ------------------------------------------------
                # Current decision
                # ------------------------------------------------

                top_general = max(
                    combined,
                    key=combined.get
                )

                # =================================================
                # CASE A: General classifier already says FSK
                # =================================================

                if top_general in (
                    "2-FSK",
                    "4-FSK"
                ):

                    pair_total = (
                        combined.get(
                            "2-FSK",
                            0.0
                        )
                        +
                        combined.get(
                            "4-FSK",
                            0.0
                        )
                    )

                    combined["2-FSK"] = (
                        pair_total
                        * fsk_2
                    )

                    combined["4-FSK"] = (
                        pair_total
                        * fsk_4
                    )

                # =================================================
                # CASE B: General classifier says 16-QAM
                # =================================================

                elif top_general == "16-QAM":

                    qam_score = combined.get(
                        "16-QAM",
                        0.0
                    )

                    if (
                        fsk_best >= 0.82
                        and fsk_evidence >= 0.42
                    ):

                        rescue_strength = np.clip(
                            (
                                fsk_best
                                - 0.82
                            )
                            / 0.18,
                            0.0,
                            1.0
                        )

                        evidence_strength = np.clip(
                            (
                                fsk_evidence
                                - 0.42
                            )
                            / 0.58,
                            0.0,
                            1.0
                        )

                        rescue_strength *= (
                            evidence_strength
                        )

                        transfer = (
                            qam_score
                            * 0.55
                            * rescue_strength
                        )

                        combined["16-QAM"] = (
                            qam_score
                            - transfer
                        )

                        combined[fsk_label] += (
                            transfer
                        )

            except Exception:

                # Specialist must never crash
                # the main classifier.
                pass

        # ====================================================
        # NORMALIZE FINAL SCORES
        # ====================================================

        total = (
            sum(
                combined.values()
            )
            or 1.0
        )

        combined = {
            key:
                value / total
            for key, value
            in combined.items()
        }

        # ====================================================
        # SORT RESULTS
        # ====================================================

        order = sorted(
            CLASSES,
            key=lambda key:
                combined[key],
            reverse=True
        )

        rows = [
            {
                "modulation": modulation,
                "probability":
                    round(
                        combined[
                            modulation
                        ]
                        * 100,
                        2
                    )
            }
            for modulation in order
        ]

        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        return {
            "detected":
                rows[0]["modulation"],

            "confidence":
                rows[0]["probability"],

            "candidates":
                rows,

            "features_used":
                FEATURES,
        }