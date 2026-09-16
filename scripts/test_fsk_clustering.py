import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.signal_io.loader import load_signal


VALIDATION_DIR = ROOT / "validation_50"

DEFAULT_SAMPLE_RATE = 1_000_000
DEFAULT_DTYPE = "float32"
DEFAULT_IQ_FORMAT = "IQ"


LABEL_MAP = {
    "2FSK": "2-FSK",
    "2-FSK": "2-FSK",
    "4FSK": "4-FSK",
    "4-FSK": "4-FSK",
}


def normalize_label(label):

    key = (
        str(label)
        .strip()
        .upper()
        .replace("_", "")
        .replace(" ", "")
    )

    return LABEL_MAP.get(key, str(label))


def discover_fsk_files():

    files = []

    for class_dir in sorted(VALIDATION_DIR.iterdir()):

        if not class_dir.is_dir():
            continue

        label = normalize_label(class_dir.name)

        if label not in {"2-FSK", "4-FSK"}:
            continue

        for path in sorted(class_dir.rglob("*")):

            if (
                path.is_file()
                and path.suffix.lower() in {".iq", ".wav"}
            ):
                files.append((path, label))

    return files


def load_v16_signal(path):

    with path.open("rb") as f:
        raw = f.read()

    signal, fs, meta = load_signal(
        path.name,
        raw,
        DEFAULT_SAMPLE_RATE,
        DEFAULT_DTYPE,
        DEFAULT_IQ_FORMAT,
    )

    return np.asarray(signal), float(fs)


def instantaneous_frequency(iq, fs):

    phase = np.unwrap(
        np.angle(iq)
    )

    freq = (
        np.diff(phase)
        * fs
        / (2 * np.pi)
    )

    return freq


def prepare_frequency(freq):

    lo, hi = np.percentile(
        freq,
        [5, 95]
    )

    freq = freq[
        (freq >= lo)
        & (freq <= hi)
    ]

    return freq


def smooth_labels(labels, window=21):

    if len(labels) < window:
        return labels

    half = window // 2

    output = labels.copy()

    for i in range(
        half,
        len(labels) - half
    ):

        section = labels[
            i - half :
            i + half + 1
        ]

        values, counts = np.unique(
            section,
            return_counts=True
        )

        output[i] = values[
            np.argmax(counts)
        ]

    return output


def get_cluster_data(freq, k):

    data = freq[::5]

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        data.reshape(-1, 1)
    )

    centers = np.sort(
        model.cluster_centers_.ravel()
    )

    # Remap labels according to sorted centers
    order = np.argsort(
        model.cluster_centers_.ravel()
    )

    mapping = {
        old: new
        for new, old in enumerate(order)
    }

    labels = np.array(
        [mapping[x] for x in labels]
    )

    return data, labels, centers


def calculate_runs(labels):

    if len(labels) == 0:
        return []

    runs = []

    current = labels[0]
    length = 1

    for value in labels[1:]:

        if value == current:
            length += 1
        else:
            runs.append(
                (current, length)
            )

            current = value
            length = 1

    runs.append(
        (current, length)
    )

    return runs


def analyze(freq):

    data, labels, centers = (
        get_cluster_data(freq, 4)
    )

    original_labels = labels.copy()

    labels = smooth_labels(
        labels,
        window=21
    )

    runs_before = calculate_runs(
        original_labels
    )

    runs_after = calculate_runs(
        labels
    )

    transitions_before = (
        max(len(runs_before) - 1, 0)
    )

    transitions_after = (
        max(len(runs_after) - 1, 0)
    )

    mean_run_before = (
        np.mean(
            [length for _, length in runs_before]
        )
        if runs_before
        else 0
    )

    mean_run_after = (
        np.mean(
            [length for _, length in runs_after]
        )
        if runs_after
        else 0
    )

    # Occupancy of the four detected levels
    counts = np.bincount(
        labels,
        minlength=4
    )

    proportions = (
        counts / max(counts.sum(), 1)
    )

    active_levels = int(
        np.sum(proportions > 0.08)
    )

    return {
        "centers": centers,
        "active_levels": active_levels,
        "transitions_before": transitions_before,
        "transitions_after": transitions_after,
        "mean_run_before": mean_run_before,
        "mean_run_after": mean_run_after,
    }


def predict(result):

    active = result["active_levels"]
    run_after = result["mean_run_after"]

    # A genuine 4-FSK signal should retain
    # four meaningful states after smoothing.
    if active >= 4 and run_after >= 8:
        return "4-FSK"

    # Otherwise favor 2-FSK.
    return "2-FSK"


def main():

    print("=" * 90)
    print(
        "SPECTRA V17.6 - FSK FREQUENCY STATE PERSISTENCE"
    )
    print("=" * 90)

    files = discover_fsk_files()

    print(
        f"\nFound {len(files)} FSK validation files.\n"
    )

    correct = 0
    total = 0

    for path, true_label in files:

        print("-" * 90)
        print(path.name)
        print(
            f"True label          : {true_label}"
        )

        try:

            iq, fs = load_v16_signal(path)

            freq = instantaneous_frequency(
                iq,
                fs
            )

            freq = prepare_frequency(
                freq
            )

            result = analyze(freq)

            predicted = predict(
                result
            )

            total += 1

            if predicted == true_label:
                correct += 1
                status = "CORRECT"
            else:
                status = "WRONG"

            print(
                "4-level centers     :",
                np.round(
                    result["centers"],
                    1
                )
            )

            print(
                f"Active levels       : "
                f"{result['active_levels']}"
            )

            print(
                f"Transitions raw     : "
                f"{result['transitions_before']}"
            )

            print(
                f"Transitions smooth  : "
                f"{result['transitions_after']}"
            )

            print(
                f"Mean run raw        : "
                f"{result['mean_run_before']:.2f}"
            )

            print(
                f"Mean run smooth     : "
                f"{result['mean_run_after']:.2f}"
            )

            print(
                f"Predicted           : {predicted}"
            )

            print(
                f"Result              : {status}"
            )

        except Exception as e:

            print(
                f"ERROR               : {e}"
            )

    print()
    print("=" * 90)
    print("FINAL RESULT")
    print("=" * 90)

    if total:

        accuracy = (
            100.0
            * correct
            / total
        )

        print(
            f"Accuracy            : "
            f"{correct}/{total} "
            f"= {accuracy:.1f}%"
        )

    else:

        print(
            "No files were analyzed."
        )


if __name__ == "__main__":
    main()