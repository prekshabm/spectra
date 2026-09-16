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

    return x / (
        np.sqrt(
            np.mean(
                np.abs(x) ** 2
            )
        )
        + 1e-12
    )


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
    # Carrier / frequency offset estimation
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

    dphi = np.angle(
        zc[1:]
        * np.conj(
            zc[:-1]
        )
    )

    # --------------------------------------------------------
    # Higher-order statistics
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

    askew = (
        float(
            skew(
                amp,
                bias=False
            )
        )
        if len(amp) > 8
        else 0.0
    )

    akurt = (
        float(
            kurtosis(
                amp,
                fisher=True,
                bias=False
            )
        )
        if len(amp) > 8
        else 0.0
    )

    # --------------------------------------------------------
    # Phase statistics
    # --------------------------------------------------------

    dphi_center = (
        dphi
        - np.median(dphi)
    )

    dstd = float(
        np.std(dphi_center)
    )

    dkurt = (
        float(
            kurtosis(
                dphi_center,
                fisher=True,
                bias=False
            )
        )
        if len(dphi_center) > 8
        else 0.0
    )

    coh2 = float(
        abs(
            np.mean(
                np.exp(
                    1j
                    * 2
                    * np.angle(zc)
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
                    * np.angle(zc)
                )
            )
        )
    )

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
    # Frequency histogram
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

    hpk, _ = find_peaks(
        hs,
        distance=5,
        prominence=max(
            np.max(hs) * 0.06,
            1e-3
        )
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

    flat = float(
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

    peak_ratio = float(
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
            flat,
            peak_ratio,
            spectral_peaks,
            freq_hist_peaks,
            freq_hist_entropy,
        ],
        dtype=float
    )


# ============================================================
# SYNTHETIC SIGNAL GENERATOR
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
        or np.random.default_rng()
    )

    n = (
        n_symbols
        * sps
    )

    t = (
        np.arange(n)
        / fs
    )

    # --------------------------------------------------------
    # BPSK
    # --------------------------------------------------------

    if mod == "BPSK":

        a = rng.integers(
            0,
            2,
            n_symbols
        )

        sy = (
            2 * a
            - 1
        )

        base = np.repeat(
            sy,
            sps
        ).astype(complex)

    # --------------------------------------------------------
    # QPSK
    # --------------------------------------------------------

    elif mod == "QPSK":

        a = rng.integers(
            0,
            4,
            n_symbols
        )

        sy = np.exp(
            1j
            * (
                np.pi / 4
                + a
                * np.pi / 2
            )
        )

        base = np.repeat(
            sy,
            sps
        )

    # --------------------------------------------------------
    # 16-QAM
    # --------------------------------------------------------

    elif mod == "16-QAM":

        a = rng.integers(
            0,
            16,
            n_symbols
        )

        I = (
            2 * (a % 4)
            - 3
        )

        Q = (
            2 * (a // 4)
            - 3
        )

        sy = (
            I + 1j * Q
        ) / np.sqrt(10)

        base = np.repeat(
            sy,
            sps
        )

    # --------------------------------------------------------
    # FSK
    # --------------------------------------------------------

    else:

        m = (
            2
            if mod == "2-FSK"
            else 4
        )

        a = rng.integers(
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
                tones[a],
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

    # --------------------------------------------------------
    # Carrier offset
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Amplitude scaling / fading
    # --------------------------------------------------------

    base *= rng.uniform(
        0.7,
        1.3
    )

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

    # --------------------------------------------------------
    # AWGN
    # --------------------------------------------------------

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
# MAIN RF TRAINING
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

    for c in CLASSES:

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
                c,
                fs=fs,
                n_symbols=n_symbols,
                sps=sps,
                snr=snr,
                rng=rng
            )

            X.append(
                _extract(
                    sig,
                    fs
                )
            )

            y.append(c)

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
# PHYSICS EXPERT
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

    s = {
        c: 0.05
        for c in CLASSES
    }

    # --------------------------------------------------------
    # FSK
    # --------------------------------------------------------

    fsk_strength = np.clip(
        (dstd - 0.12)
        / 0.50,
        0,
        1
    )

    fsk_strength *= np.clip(
        (0.65 - acv)
        / 0.65,
        0,
        1
    )

    fsk_strength *= np.clip(
        (0.95 - hent)
        / 0.95,
        0,
        1
    )

    s["2-FSK"] += (
        1.2
        * fsk_strength
    )

    s["4-FSK"] += (
        1.2
        * fsk_strength
    )

    if hpeaks == 2:

        s["2-FSK"] += (
            1.8
            * fsk_strength
        )

    if hpeaks >= 3:

        s["4-FSK"] += (
            1.8
            * fsk_strength
        )

    if peaks >= 3:

        s["4-FSK"] += 0.3

    # --------------------------------------------------------
    # QAM
    # --------------------------------------------------------

    qam = np.clip(
        (acv - 0.18)
        / 0.38,
        0,
        1
    )

    s["16-QAM"] += (
        2.0
        * qam
    )

    # --------------------------------------------------------
    # PSK
    # --------------------------------------------------------

    psk = (
        np.clip(
            (0.28 - acv)
            / 0.28,
            0,
            1
        )
        *
        np.clip(
            (0.30 - dstd)
            / 0.30,
            0,
            1
        )
    )

    s["BPSK"] += (
        0.9
        * psk
    )

    s["QPSK"] += (
        1.0
        * psk
    )

    # --------------------------------------------------------
    # Cumulants
    # --------------------------------------------------------

    bpsk = (
        np.exp(
            -(
                (c20 - 1.0)
                / 0.28
            ) ** 2
        )
        *
        np.exp(
            -(
                (c40 - 2.0)
                / 0.45
            ) ** 2
        )
    )

    qpsk = (
        np.exp(
            -(
                (c20 - 0.05)
                / 0.22
            ) ** 2
        )
        *
        np.exp(
            -(
                (c40 - 1.0)
                / 0.35
            ) ** 2
        )
    )

    qamc = (
        np.exp(
            -(
                (c20 - 0.20)
                / 0.20
            ) ** 2
        )
        *
        np.exp(
            -(
                (c40 - 0.68)
                / 0.35
            ) ** 2
        )
    )

    s["BPSK"] += (
        2.2
        * bpsk
    )

    s["QPSK"] += (
        2.2
        * qpsk
    )

    s["16-QAM"] += (
        1.5
        * qamc
    )

    # --------------------------------------------------------
    # Phase concentration
    # --------------------------------------------------------

    s["QPSK"] += (
        0.8
        * np.clip(
            (coh4 - 0.45)
            / 0.5,
            0,
            1
        )
    )

    s["BPSK"] += (
        0.5
        * np.clip(
            (coh2 - 0.45)
            / 0.5,
            0,
            1
        )
    )

    # --------------------------------------------------------
    # Low-SNR QPSK rescue
    # --------------------------------------------------------

    qpsk_signature = (
        np.clip(
            (0.16 - c20)
            / 0.16,
            0,
            1
        )
        *
        np.clip(
            (c40 - 0.42)
            / 0.45,
            0,
            1
        )
        *
        np.clip(
            (coh4 - 0.12)
            / 0.35,
            0,
            1
        )
        *
        np.clip(
            (0.46 - acv)
            / 0.30,
            0,
            1
        )
    )

    bpsk_signature = (
        np.clip(
            (c20 - 0.20)
            / 0.55,
            0,
            1
        )
        *
        np.clip(
            (coh2 - 0.20)
            / 0.45,
            0,
            1
        )
        *
        np.clip(
            (c40 - 0.45)
            / 0.70,
            0,
            1
        )
    )

    s["QPSK"] += (
        14.0
        * qpsk_signature
    )

    s["BPSK"] += (
        25.0
        * bpsk_signature
    )

    return s


# ============================================================
# CLASSIFIER
# ============================================================

class ModulationClassifier:

    def __init__(
        self,
        path
    ):

        self.path = Path(
            path
        )

        self.model = None
        self.loaded = False

        # ====================================================
        # V17 QPSK / 16-QAM SPECIALIST
        # ====================================================

        self.qpsk_qam_specialist = None

        specialist_path = (
            self.path.parent
            / "specialist_qpsk_qam.joblib"
        )

        if specialist_path.exists():

            try:

                specialist_obj = (
                    joblib.load(
                        specialist_path
                    )
                )

                self.qpsk_qam_specialist = (
                    specialist_obj["model"]
                )

            except Exception:

                self.qpsk_qam_specialist = None

        # ====================================================
        # V17.1 FSK SPECIALIST
        # ====================================================

        # ====================================================
    # V17.7 REAL-DATA FSK SPECIALIST
    # ====================================================

    self.fsk_specialist = None

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
        # LOAD MAIN MODEL
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

        # ====================================================
        # FALLBACK
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
        # Feature extraction
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
        # Random Forest
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

        rp = dict(
            zip(
                classes,
                rf
            )
        )

        # ----------------------------------------------------
        # Physics expert
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
        # RF / expert blend
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
                * rp.get(
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

        # ----------------------------------------------------
        # Temporal voting
        # ----------------------------------------------------

        if len(signal) >= 4096:

            chunk_probs = []

            step = max(
                1024,
                len(signal) // 6
            )

            for start in range(
                0,
                len(signal)
                - 2048
                + 1,
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
                                        0
                                    )
                                    for p
                                    in chunk_probs
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

        # ----------------------------------------------------
        # Existing BPSK evidence
        # ----------------------------------------------------

        if (
            x[0] > 0.25
            and x[8] > 0.25
            and x[1] > 0.45
        ):

            combined["BPSK"] += 0.50

        # ----------------------------------------------------
        # Existing QPSK evidence
        # ----------------------------------------------------

        if (
            x[0] < 0.16
            and x[1] > 0.42
            and x[9] > 0.12
            and x[3] < 0.46
        ):

            combined["QPSK"] += 0.25

        # ====================================================
        # V17 QPSK / 16-QAM SPECIALIST
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
        # V17.1 2-FSK / 4-FSK SPECIALIST
        # ====================================================

        top_general = max(
            combined,
            key=combined.get
        )

        if (
            self.fsk_specialist
            is not None
            and top_general
            in (
                "2-FSK",
                "4-FSK"
            )
        ):

            fsk_prob = (
                self.fsk_specialist
                .predict_proba(
                    x.reshape(
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

            fsk_prediction = max(
                fsk_scores,
                key=fsk_scores.get
            )

            if fsk_prediction in (
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
                    * fsk_scores.get(
                        "2-FSK",
                        0.0
                    )
                )

                combined["4-FSK"] = (
                    pair_total
                    * fsk_scores.get(
                        "4-FSK",
                        0.0
                    )
                )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

          top_general = max(
        combined,
        key=combined.get
    )

    if (
     self.fsk_specialist is not None
        and top_general in (
            "2-FSK",
            "4-FSK"
     )
    ):

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
            .   predict_proba(
                    fsk_x.reshape(1, -1)
                )[0]
            )

            fsk_classes = list(
                self.fsk_specialist.classes_
            )

            fsk_scores = dict(
                zip(
                    fsk_classes,
                    fsk_prob
                )
            )

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
                * fsk_scores.get(
                    "2-FSK",
                    0.0
                )
            )

            combined["4-FSK"] = (
             pair_total
                * fsk_scores.get(
                    "4-FSK",
                    0.0
                )
            )

        except Exception:

            pass