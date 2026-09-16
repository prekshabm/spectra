import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.signal_io.loader import load_signal


SAMPLE_RATE = 1_000_000
DTYPE = "float32"
IQ_FORMAT = "IQ"

VALIDATION_DIR = ROOT / "validation_50"


def load_iq(path):

    with path.open("rb") as f:
        raw = f.read()

    signal, fs, meta = load_signal(
        path.name,
        raw,
        SAMPLE_RATE,
        DTYPE,
        IQ_FORMAT,
    )

    return np.asarray(signal), float(fs)


def label_from_path(path):

    name = path.name.upper()

    if "2FSK" in name:
        return "2-FSK"

    if "4FSK" in name:
        return "4-FSK"

    return None


def find_training_files():

    files = []

    # Search common project data locations.
    roots = [
        ROOT / "data",
        ROOT / "train",
        ROOT / "training",
        ROOT / "data_50",
    ]

    for base in roots:

        if not base.exists():
            continue

        for path in base.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() != ".iq":
                continue

            label = label_from_path(path)

            if label is not None:
                files.append((path, label))

    # Remove duplicates.
    unique = {}
    for path, label in files:
        unique[str(path.resolve())] = (path, label)

    return list(unique.values())


def find_validation_files():

    files = []

    for path in VALIDATION_DIR.rglob("*.iq"):

        label = label_from_path(path)

        if label is not None:
            files.append((path, label))

    return files


def instantaneous_frequency(iq, fs):

    phase = np.unwrap(np.angle(iq))

    freq = (
        np.diff(phase)
        * fs
        / (2.0 * np.pi)
    )

    return freq


def frequency_features(iq, fs):

    freq = instantaneous_frequency(iq, fs)

    # Remove extreme outliers.
    lo, hi = np.percentile(freq, [2, 98])

    freq = freq[
        (freq >= lo) &
        (freq <= hi)
    ]

    # Downsample.
    freq = freq[::5]

    if len(freq) < 100:
        return np.zeros(18, dtype=float)

    q = np.percentile(
        freq,
        [5, 10, 25, 50, 75, 90, 95]
    )

    mean = np.mean(freq)
    std = np.std(freq)

    # Histogram features.
    hist, edges = np.histogram(
        freq,
        bins=40
    )

    p = hist.astype(float)

    if p.sum() > 0:
        p /= p.sum()

    nz = p[p > 0]

    entropy = (
        -np.sum(nz * np.log2(nz))
        if len(nz)
        else 0.0
    )

    peak_bins = 0

    if len(hist) >= 3:

        for i in range(1, len(hist) - 1):

            if (
                hist[i] > hist[i - 1]
                and hist[i] >= hist[i + 1]
                and hist[i] > 0.10 * np.max(hist)
            ):
                peak_bins += 1

    # KMeans-like frequency structure using
    # simple 2 and 4 cluster fits.
    from sklearn.cluster import KMeans

    X = freq.reshape(-1, 1)

    km2 = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=5
    )

    lab2 = km2.fit_predict(X)

    centers2 = np.sort(
        km2.cluster_centers_.ravel()
    )

    sep2 = (
        centers2[1] - centers2[0]
    )

    km4 = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=5
    )

    lab4 = km4.fit_predict(X)

    centers4 = np.sort(
        km4.cluster_centers_.ravel()
    )

    gaps4 = np.diff(centers4)

    mean_gap4 = np.mean(gaps4)

    gap_cv4 = (
        np.std(gaps4) / mean_gap4
        if mean_gap4 > 0
        else 0.0
    )

    # Silhouette-like separation using
    # within/between spread.
    def simple_cluster_score(labels, centers):

        total = 0.0

        for idx, center in enumerate(centers):

            pts = freq[labels == idx]

            if len(pts) == 0:
                continue

            total += (
                len(pts)
                * np.std(pts)
            )

        return total / len(freq)

    spread2 = simple_cluster_score(
        lab2,
        centers2
    )

    spread4 = simple_cluster_score(
        lab4,
        centers4
    )

    # Occupancy of 4 frequency states.
    counts4 = np.bincount(
        np.argsort(
            np.argsort(
                km4.cluster_centers_.ravel()
            )
        )[lab4],
        minlength=4
    )

    props4 = (
        counts4
        / max(np.sum(counts4), 1)
    )

    return np.array([
        mean,
        std,
        q[0],
        q[1],
        q[2],
        q[3],
        q[4],
        q[5],
        q[6],
        entropy,
        peak_bins,
        sep2,
        mean_gap4,
        gap_cv4,
        spread2,
        spread4,
        props4[0],
        props4[-1],
    ], dtype=float)


def build_dataset(files):

    X = []
    y = []

    for i, (path, label) in enumerate(files, 1):

        print(
            f"Extracting {i}/{len(files)}: "
            f"{path.name}"
        )

        try:

            iq, fs = load_iq(path)

            features = frequency_features(
                iq,
                fs
            )

            X.append(features)
            y.append(label)

        except Exception as e:

            print(
                f"  ERROR: {e}"
            )

    return np.asarray(X), np.asarray(y)


def main():

    print("=" * 90)
    print(
        "SPECTRA V17.7 - REAL-DATA FSK SPECIALIST"
    )
    print("=" * 90)

    train_files = find_training_files()
    val_files = find_validation_files()

    print(
        f"\nTraining FSK files found   : "
        f"{len(train_files)}"
    )

    print(
        f"Validation FSK files       : "
        f"{len(val_files)}"
    )

    train_2 = sum(
        label == "2-FSK"
        for _, label in train_files
    )

    train_4 = sum(
        label == "4-FSK"
        for _, label in train_files
    )

    print(
        f"Training 2-FSK             : {train_2}"
    )

    print(
        f"Training 4-FSK             : {train_4}"
    )

    if train_2 < 5 or train_4 < 5:

        print(
            "\nERROR: Not enough real FSK training files "
            "were found."
        )

        print(
            "Check the data folder structure before "
            "changing anything else."
        )

        return

    print("\nBuilding training features...")

    X_train, y_train = build_dataset(
        train_files
    )

    print(
        "\nTraining Random Forest..."
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    
    model.fit(
    X_train,
    y_train
    )

    import joblib

    MODEL_PATH = ROOT / "models" / "specialist_fsk_real.joblib"

    joblib.dump(
       {
        "model": model,
        "feature_count": X_train.shape[1],
        "classes": ["2-FSK", "4-FSK"],
    },
         MODEL_PATH
)

    print(
          f"\nSaved specialist model to:\n{MODEL_PATH}"
)


    print(
        "\nBuilding validation features..."
    )

    X_val, y_val = build_dataset(
        val_files
    )

    predictions = model.predict(
        X_val
    )

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    print()
    print("=" * 90)
    print("V17.7 RESULT")
    print("=" * 90)

    print(
        f"Accuracy: "
        f"{np.sum(predictions == y_val)}"
        f"/{len(y_val)}"
        f" = {accuracy * 100:.1f}%"
    )

    print()
    print("PREDICTIONS")
    print("-" * 90)

    for (path, true_label), prediction in zip(
        val_files,
        predictions
    ):

        print(
            f"{path.name:35s} "
            f"True={true_label:6s} "
            f"Pred={prediction}"
        )

    print()
    print("CONFUSION MATRIX")
    print("-" * 90)

    print(
        confusion_matrix(
            y_val,
            predictions,
            labels=["2-FSK", "4-FSK"]
        )
    )


if __name__ == "__main__":
    main()