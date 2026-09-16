from pathlib import Path
import sys
import csv

import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORT PROJECT LOADER
# ============================================================

from backend.signal_io.loader import load_signal


# ============================================================
# PATHS
# ============================================================

DATA_DIR = ROOT / "data"
VALIDATION_DIR = ROOT / "validation_50"
MODEL_PATH = (
    ROOT
    / "models"
    / "specialist_fsk_v2.joblib"
)


# ============================================================
# SETTINGS
# ============================================================

FS = 1_000_000
DTYPE = "float32"
IQ_FORMAT = "IQ"

FSK_CLASSES = [
    "2-FSK",
    "4-FSK",
]


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(name):

    s = (
        str(name)
        .strip()
        .upper()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )

    if s in {
        "2FSK",
    }:
        return "2-FSK"

    if s in {
        "4FSK",
    }:
        return "4-FSK"

    return None


# ============================================================
# SIGNAL LOADER
# ============================================================

def load_project_signal(path):

    with path.open("rb") as f:

        raw = f.read()

    signal, fs, meta = load_signal(
        path.name,
        raw,
        FS,
        DTYPE,
        IQ_FORMAT,
    )

    signal = np.asarray(
        signal,
        dtype=np.complex128
    )

    fs = float(fs)

    return signal, fs


# ============================================================
# DISCOVER TRAINING SIGNALS
#
# We search DATA recursively and automatically detect
# 2-FSK / 4-FSK from folder or filename names.
#
# Validation_50 is deliberately excluded.
# ============================================================

def discover_training_files():

    found = []

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            f"DATA directory not found:\n{DATA_DIR}"
        )

    for path in DATA_DIR.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".iq",
            ".wav",
        }:
            continue

        text = (
            str(path)
            .upper()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

        if "2FSK" in text:

            label = "2-FSK"

        elif "4FSK" in text:

            label = "4-FSK"

        else:

            continue

        found.append(
            (
                path,
                label
            )
        )

    # remove duplicates
    unique = {}

    for path, label in found:

        unique[str(path.resolve())] = (
            path,
            label
        )

    found = list(
        unique.values()
    )

    return sorted(
        found,
        key=lambda item: str(item[0]).lower()
    )


# ============================================================
# FREQUENCY FEATURE EXTRACTION
# ============================================================

def extract_v2_features(
    signal,
    fs
):

    x = np.asarray(
        signal,
        dtype=np.complex128
    )

    # --------------------------------------------------------
    # Remove invalid samples
    # --------------------------------------------------------

    x = x[
        np.isfinite(x)
    ]

    if len(x) < 256:

        return np.zeros(
            24,
            dtype=float
        )

    # --------------------------------------------------------
    # Normalize amplitude
    # --------------------------------------------------------

    power = np.sqrt(
        np.mean(
            np.abs(x) ** 2
        )
    )

    x = (
        x
        / (
            power
            + 1e-12
        )
    )

    # --------------------------------------------------------
    # Limit computational cost
    # --------------------------------------------------------

    if len(x) > 65536:

        x = x[:65536]

    # --------------------------------------------------------
    # Instantaneous frequency
    # --------------------------------------------------------

    phase = np.unwrap(
        np.angle(x)
    )

    inst_freq = (
        np.diff(phase)
        * fs
        / (
            2
            * np.pi
        )
    )

    inst_freq = inst_freq[
        np.isfinite(inst_freq)
    ]

    if len(inst_freq) < 128:

        return np.zeros(
            24,
            dtype=float
        )

    # --------------------------------------------------------
    # Robust trimming
    #
    # Removes extreme noise/outliers without assuming
    # exact frequency locations.
    # --------------------------------------------------------

    lo, hi = np.percentile(
        inst_freq,
        [
            2,
            98
        ]
    )

    f = inst_freq[
        (inst_freq >= lo)
        &
        (inst_freq <= hi)
    ]

    if len(f) < 128:

        f = inst_freq

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    mean_f = float(
        np.mean(f)
    )

    std_f = float(
        np.std(f)
    )

    q05 = float(
        np.percentile(
            f,
            5
        )
    )

    q10 = float(
        np.percentile(
            f,
            10
        )
    )

    q25 = float(
        np.percentile(
            f,
            25
        )
    )

    q50 = float(
        np.percentile(
            f,
            50
        )
    )

    q75 = float(
        np.percentile(
            f,
            75
        )
    )

    q90 = float(
        np.percentile(
            f,
            90
        )
    )

    q95 = float(
        np.percentile(
            f,
            95
        )
    )

    # --------------------------------------------------------
    # Histogram
    # --------------------------------------------------------

    hist, edges = np.histogram(
        f,
        bins=96,
        density=False
    )

    hist = hist.astype(
        float
    )

    hist_sum = (
        np.sum(hist)
        + 1e-12
    )

    prob = (
        hist
        / hist_sum
    )

    entropy = float(
        -np.sum(
            prob
            * np.log(
                prob
                + 1e-12
            )
        )
        / np.log(
            len(prob)
        )
    )

    # --------------------------------------------------------
    # Histogram peak detection
    # --------------------------------------------------------

    smooth = np.convolve(
        hist,
        np.ones(5) / 5.0,
        mode="same"
    )

    peak_threshold = (
        np.max(smooth)
        * 0.10
    )

    peaks = []

    for i in range(
        2,
        len(smooth) - 2
    ):

        if (
            smooth[i]
            > smooth[i - 1]
            and smooth[i]
            >= smooth[i + 1]
            and smooth[i]
            >= peak_threshold
        ):

            # non-maximum suppression
            if not peaks:

                peaks.append(i)

            else:

                if (
                    i
                    - peaks[-1]
                    >= 4
                ):

                    peaks.append(i)

                elif (
                    smooth[i]
                    > smooth[peaks[-1]]
                ):

                    peaks[-1] = i

    histogram_peak_count = float(
        min(
            len(peaks),
            8
        )
    )

    # --------------------------------------------------------
    # KMEANS K=2
    # --------------------------------------------------------

    sample_count = min(
        len(f),
        12000
    )

    sample_idx = np.linspace(
        0,
        len(f) - 1,
        sample_count
    ).astype(int)

    fsample = f[
        sample_idx
    ].reshape(
        -1,
        1
    )

    try:

        km2 = KMeans(
            n_clusters=2,
            n_init=8,
            random_state=42
        )

        lab2 = km2.fit_predict(
            fsample
        )

        centers2 = np.sort(
            km2.cluster_centers_.ravel()
        )

        gap2 = float(
            abs(
                centers2[1]
                - centers2[0]
            )
        )

        counts2 = np.bincount(
            lab2,
            minlength=2
        ).astype(float)

        props2 = (
            counts2
            / (
                np.sum(counts2)
                + 1e-12
            )
        )

        balance2 = float(
            np.min(
                props2
            )
        )

    except Exception:

        gap2 = 0.0
        balance2 = 0.0

    # --------------------------------------------------------
    # KMEANS K=3
    # --------------------------------------------------------

    try:

        km3 = KMeans(
            n_clusters=3,
            n_init=8,
            random_state=42
        )

        lab3 = km3.fit_predict(
            fsample
        )

        centers3 = np.sort(
            km3.cluster_centers_.ravel()
        )

        gaps3 = np.diff(
            centers3
        )

        mean_gap3 = float(
            np.mean(
                gaps3
            )
        )

        gap_cv3 = float(
            np.std(
                gaps3
            )
            / (
                np.mean(
                    gaps3
                )
                + 1e-12
            )
        )

        counts3 = np.bincount(
            lab3,
            minlength=3
        ).astype(float)

        props3 = (
            counts3
            / (
                np.sum(counts3)
                + 1e-12
            )
        )

        min_prop3 = float(
            np.min(
                props3
            )
        )

    except Exception:

        mean_gap3 = 0.0
        gap_cv3 = 0.0
        min_prop3 = 0.0

    # --------------------------------------------------------
    # KMEANS K=4
    # --------------------------------------------------------

    try:

        km4 = KMeans(
            n_clusters=4,
            n_init=8,
            random_state=42
        )

        lab4 = km4.fit_predict(
            fsample
        )

        centers4 = np.sort(
            km4.cluster_centers_.ravel()
        )

        gaps4 = np.diff(
            centers4
        )

        mean_gap4 = float(
            np.mean(
                gaps4
            )
        )

        gap_std4 = float(
            np.std(
                gaps4
            )
        )

        gap_cv4 = float(
            gap_std4
            / (
                mean_gap4
                + 1e-12
            )
        )

        # Relative spread of the outer levels
        spread4 = float(
            centers4[-1]
            - centers4[0]
        )

        # How well populated are all four clusters?
        counts4 = np.bincount(
            lab4,
            minlength=4
        ).astype(float)

        props4 = (
            counts4
            / (
                np.sum(
                    counts4
                )
                + 1e-12
            )
        )

        min_prop4 = float(
            np.min(
                props4
            )
        )

        second_min_prop4 = float(
            np.partition(
                props4,
                1
            )[1]
        )

    except Exception:

        mean_gap4 = 0.0
        gap_std4 = 0.0
        gap_cv4 = 0.0
        spread4 = 0.0
        min_prop4 = 0.0
        second_min_prop4 = 0.0

    # --------------------------------------------------------
    # Occupancy around four candidate levels
    #
    # This helps when noise makes one level weak but the
    # underlying distribution is still multi-level.
    # --------------------------------------------------------

    try:

        centers = centers4

        d = np.abs(
            fsample
            - centers.reshape(
                1,
                -1
            )
        )

        nearest = np.argmin(
            d,
            axis=1
        )

        nearest_dist = np.min(
            d,
            axis=1
        )

        scale = (
            mean_gap4
            + 1e-12
        )

        confidence4 = (
            1.0
            - np.clip(
                nearest_dist
                / scale,
                0.0,
                1.0
            )
        )

        mean_cluster_confidence = float(
            np.mean(
                confidence4
            )
        )

        # Outer versus inner population.
        # Cluster indices 0 and 3 are outer.
        outer_fraction = float(
            np.mean(
                (
                    nearest == 0
                )
                |
                (
                    nearest == 3
                )
            )
        )

        middle_fraction = float(
            np.mean(
                (
                    nearest == 1
                )
                |
                (
                    nearest == 2
                )
            )
        )

    except Exception:

        mean_cluster_confidence = 0.0
        outer_fraction = 0.0
        middle_fraction = 0.0

    # --------------------------------------------------------
    # Frequency-transition behavior
    # --------------------------------------------------------

    # Smooth instantaneous frequency so tiny noise jumps
    # don't dominate the transition count.
    kernel = np.ones(
         nine := 9
    ) / nine

    fsmooth = np.convolve(
        inst_freq,
        kernel,
        mode="same"
    )

    local_diff = np.abs(
        np.diff(
            fsmooth
        )
    )

    transition_threshold = max(
        float(
            np.percentile(
                local_diff,
                85
            )
        ),
        0.15
        * std_f
    )

    transition_count = int(
        np.sum(
            local_diff
            > transition_threshold
        )
    )

    transition_rate = float(
        transition_count
        / (
            len(local_diff)
            + 1e-12
        )
    )

    # --------------------------------------------------------
    # Return robust 24-feature vector
    # --------------------------------------------------------

    features = np.array(
        [
            mean_f,
            std_f,

            q05,
            q10,
            q25,
            q50,
            q75,
            q90,
            q95,

            entropy,
            histogram_peak_count,

            gap2,
            balance2,

            mean_gap3,
            gap_cv3,
            min_prop3,

            mean_gap4,
            gap_std4,
            gap_cv4,
            spread4,
            min_prop4,
            second_min_prop4,

            mean_cluster_confidence,
            transition_rate,
        ],
        dtype=float
    )

    return np.nan_to_num(
        features,
        nan=0.0,
        posinf=1e12,
        neginf=-1e12
    )


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(
    files
):

    X = []
    y = []

    print()
    print(
        "Extracting training features..."
    )

    for index, (
        path,
        label
    ) in enumerate(
        files,
        start=1
    ):

        try:

            signal, fs = (
                load_project_signal(
                    path
                )
            )

            features = (
                extract_v2_features(
                    signal,
                    fs
                )
            )

            X.append(
                features
            )

            y.append(
                label
            )

            print(
                f"[{index:03d}/"
                f"{len(files):03d}] "
                f"{label:<6} "
                f"{path.name}"
            )

        except Exception as e:

            print(
                f"[SKIP] "
                f"{path.name}: {e}"
            )

    if not X:

        raise RuntimeError(
            "No usable training signals."
        )

    X = np.asarray(
        X,
        dtype=float
    )

    y = np.asarray(
        y
    )

    return X, y


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X,
    y
):

    print()
    print(
        "=" * 70
    )

    print(
        "TRAINING FSK V2 MODEL"
    )

    print(
        "=" * 70
    )

    print()
    print(
        f"Training samples : {len(X)}"
    )

    print(
        f"Feature count    : {X.shape[1]}"
    )

    print()

    for label in FSK_CLASSES:

        count = int(
            np.sum(
                y == label
            )
        )

        print(
            f"{label:<8}: {count}"
        )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=20,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X,
        y
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "features": [
                "mean_f",
                "std_f",
                "q05",
                "q10",
                "q25",
                "q50",
                "q75",
                "q90",
                "q95",
                "entropy",
                "histogram_peak_count",
                "gap2",
                "balance2",
                "mean_gap3",
                "gap_cv3",
                "min_prop3",
                "mean_gap4",
                "gap_std4",
                "gap_cv4",
                "spread4",
                "min_prop4",
                "second_min_prop4",
                "mean_cluster_confidence",
                "transition_rate",
            ],
            "classes": FSK_CLASSES,
        },
        MODEL_PATH
    )

    print()
    print(
        f"Saved model to:"
    )

    print(
        MODEL_PATH
    )

    return model


# ============================================================
# VALIDATION DISCOVERY
# ============================================================

def discover_validation_fsk():

    files = []

    if not VALIDATION_DIR.exists():

        raise FileNotFoundError(
            f"Validation directory not found:\n"
            f"{VALIDATION_DIR}"
        )

    for class_dir in sorted(
        VALIDATION_DIR.iterdir()
    ):

        if not class_dir.is_dir():
            continue

        label = normalize_label(
            class_dir.name
        )

        if label not in FSK_CLASSES:
            continue

        for path in sorted(
            class_dir.rglob("*")
        ):

            if not path.is_file():
                continue

            if path.suffix.lower() not in {
                ".iq",
                ".wav",
            }:
                continue

            files.append(
                (
                    path,
                    label
                )
            )

    return sorted(
        files,
        key=lambda item: (
            item[1],
            item[0].name.lower()
        )
    )


# ============================================================
# VALIDATE V2
# ============================================================

def validate_model(
    model,
    files
):

    print()
    print("=" * 110)
    print(
        "FSK V2 VALIDATION"
    )
    print("=" * 110)

    print()

    correct = 0
    total = 0

    per_class = {
        "2-FSK": {
            "correct": 0,
            "total": 0,
        },
        "4-FSK": {
            "correct": 0,
            "total": 0,
        },
    }

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print(
        f"{'FILE':<40}"
        f"{'TRUE':<9}"
        f"{'PRED':<9}"
        f"{'CONF':>8}"
        f"  STATUS"
    )

    print(
        "-" * 110
    )

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    for path, true_label in files:

        try:

            signal, fs = (
                load_project_signal(
                    path
                )
            )

            features = (
                extract_v2_features(
                    signal,
                    fs
                )
            )

            prob = (
                model
                .predict_proba(
                    features.reshape(
                        1,
                        -1
                    )
                )[0]
            )

            classes = list(
                model.classes_
            )

            scores = dict(
                zip(
                    classes,
                    prob
                )
            )

            prediction = max(
                scores,
                key=scores.get
            )

            confidence = (
                scores[prediction]
                * 100.0
            )

            total += 1

            per_class[
                true_label
            ]["total"] += 1

            if prediction == true_label:

                correct += 1

                per_class[
                    true_label
                ]["correct"] += 1

                status = "CORRECT"

            else:

                status = "WRONG"

            print(
                f"{path.name:<40}"
                f"{true_label:<9}"
                f"{prediction:<9}"
                f"{confidence:>7.2f}%"
                f"  {status}"
            )

        except Exception as e:

            print(
                f"{path.name:<40}"
                f"{true_label:<9}"
                f"ERROR: {e}"
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    accuracy = (
        correct
        / total
        * 100.0
        if total
        else 0.0
    )

    print()
    print(
        "=" * 70
    )

    print(
        "RESULT"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"FSK V2 accuracy : "
        f"{correct}/{total} "
        f"= {accuracy:.2f}%"
    )

    print()

    for label in FSK_CLASSES:

        c = per_class[
            label
        ]["correct"]

        n = per_class[
            label
        ]["total"]

        pct = (
            c
            / n
            * 100.0
            if n
            else 0.0
        )

        print(
            f"{label:<8}: "
            f"{c}/{n} "
            f"= {pct:.2f}%"
        )

    print()
    print(
        "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 70
    )

    print(
        "SPECTRA V17.8 - REAL-DATA FSK V2"
    )

    print(
        "=" * 70
    )

    print()
    print(
        f"Training directory:"
    )

    print(
        DATA_DIR
    )

    print()
    print(
        f"Validation directory:"
    )

    print(
        VALIDATION_DIR
    )

    # --------------------------------------------------------
    # Training files
    # --------------------------------------------------------

    training_files = (
        discover_training_files()
    )

    print()
    print(
        f"Found "
        f"{len(training_files)} "
        f"FSK training files."
    )

    two_fsk = sum(
        1
        for _, label
        in training_files
        if label == "2-FSK"
    )

    four_fsk = sum(
        1
        for _, label
        in training_files
        if label == "4-FSK"
    )

    print(
        f"2-FSK training files : "
        f"{two_fsk}"
    )

    print(
        f"4-FSK training files : "
        f"{four_fsk}"
    )

    if two_fsk == 0 or four_fsk == 0:

        print()
        print(
            "ERROR: Could not find both "
            "2-FSK and 4-FSK training data."
        )

        print()
        print(
            "Check your data folder structure."
        )

        return

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    X, y = build_dataset(
        training_files
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X,
        y
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_files = (
        discover_validation_fsk()
    )

    print()
    print(
        f"Found "
        f"{len(validation_files)} "
        f"FSK validation files."
    )

    if len(validation_files) != 20:

        print()
        print(
            "[WARNING] Expected 20 FSK "
            "validation files."
        )

    validate_model(
        model,
        validation_files
    )


if __name__ == "__main__":
    main()