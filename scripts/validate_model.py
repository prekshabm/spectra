from pathlib import Path
import sys
import csv
import traceback

import numpy as np

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# SPECTRA V16 IMPORTS
# ============================================================

from backend.signal_io.loader import load_signal
from backend.preprocessing.pipeline import preprocess
from backend.dsp.analysis import analyze_signal
from backend.ml.classifier import ModulationClassifier, CLASSES


# ============================================================
# PATHS
# ============================================================

VALIDATION_DIR = ROOT / "validation_50"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# V16 DEFAULT SETTINGS
# ============================================================

DEFAULT_SAMPLE_RATE = 1_000_000
DEFAULT_DTYPE = "float32"
DEFAULT_IQ_FORMAT = "IQ"


# ============================================================
# LABEL NORMALIZATION
# ============================================================

LABEL_MAP = {
    "BPSK": "BPSK",
    "QPSK": "QPSK",
    "2FSK": "2-FSK",
    "2-FSK": "2-FSK",
    "4FSK": "4-FSK",
    "4-FSK": "4-FSK",
    "16QAM": "16-QAM",
    "16-QAM": "16-QAM",
}


def normalize_label(label):

    key = (
        str(label)
        .strip()
        .upper()
        .replace("_", "")
        .replace(" ", "")
    )

    return LABEL_MAP.get(
        key,
        str(label)
    )


# ============================================================
# DISCOVER VALIDATION FILES
# ============================================================

def discover_files():

    files = []

    if not VALIDATION_DIR.exists():

        raise FileNotFoundError(
            f"Validation directory does not exist:\n"
            f"{VALIDATION_DIR}"
        )

    for class_dir in sorted(
        VALIDATION_DIR.iterdir()
    ):

        if not class_dir.is_dir():
            continue

        true_label = normalize_label(
            class_dir.name
        )

        if true_label not in CLASSES:

            print(
                f"[WARNING] Ignoring unknown "
                f"class folder: {class_dir.name}"
            )

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
                    true_label
                )
            )

    return files


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(value):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


# ============================================================
# LOAD SIGNAL USING THE SAME V16 LOADER
# ============================================================

def load_v16_signal(path):

    with path.open(
        "rb"
    ) as f:

        raw = f.read()

    signal, fs, meta = load_signal(
        path.name,
        raw,
        DEFAULT_SAMPLE_RATE,
        DEFAULT_DTYPE,
        DEFAULT_IQ_FORMAT,
    )

    return (
        np.asarray(signal),
        float(fs),
        meta,
    )


# ============================================================
# EXTRACT PREDICTION
# ============================================================

def get_prediction(
    classifier,
    features,
    signal,
    fs
):

    result = classifier.predict(
        features,
        signal,
        fs
    )

    if not isinstance(
        result,
        dict
    ):

        return (
            str(result),
            np.nan,
            {}
        )

    detected = result.get(
        "detected",
        ""
    )

    confidence = safe_float(
        result.get(
            "confidence",
            np.nan
        )
    )

    candidates = result.get(
        "candidates",
        []
    )

    probabilities = {}

    if isinstance(
        candidates,
        list
    ):

        for item in candidates:

            if not isinstance(
                item,
                dict
            ):
                continue

            label = item.get(
                "modulation"
            )

            probability = safe_float(
                item.get(
                    "probability",
                    np.nan
                )
            )

            if label is not None:

                probabilities[
                    normalize_label(label)
                ] = probability

    return (
        normalize_label(detected),
        confidence,
        probabilities
    )


# ============================================================
# VALIDATE ONE FILE
# ============================================================

def validate_file(
    classifier,
    path,
    true_label
):

    print()
    print(
        "-" * 60
    )

    print(
        f"Testing: {path.name}"
    )

    print(
        f"True label: {true_label}"
    )

    try:

        # ----------------------------------------------------
        # STEP 1: LOAD
        # ----------------------------------------------------

        signal, fs, meta = (
            load_v16_signal(path)
        )

        print(
            f"Loaded samples: "
            f"{len(signal):,}"
        )

        print(
            f"Sample rate: "
            f"{fs:,.0f} Hz"
        )

        # ----------------------------------------------------
        # STEP 2: PREPROCESS
        # ----------------------------------------------------

        try:

            processed = preprocess(
                signal,
                fs
            )

        except TypeError:

            # Some v16 versions may expose
            # preprocessing with a different
            # argument contract. If so, use
            # the raw complex signal exactly
            # like main.py does.

            processed = signal

        if isinstance(
            processed,
            tuple
        ):

            analysis_signal_input = (
                processed[0]
            )

        else:

            analysis_signal_input = (
                processed
            )

        analysis_signal_input = np.asarray(
            analysis_signal_input
        )

        # ----------------------------------------------------
        # STEP 3: DSP ANALYSIS
        # ----------------------------------------------------

        analysis = analyze_signal(
            signal,
            fs
        )

        # ----------------------------------------------------
        # STEP 4: GET FEATURES
        # ----------------------------------------------------

        features = analysis.get(
            "features",
            {}
        )

        if not features:

            raise RuntimeError(
                "analyze_signal() returned "
                "no feature dictionary."
            )

        # ----------------------------------------------------
        # STEP 5: CLASSIFICATION
        # ----------------------------------------------------

        (
            predicted,
            confidence,
            probabilities
        ) = get_prediction(
            classifier,
            features,
            signal,
            fs
        )

        # ----------------------------------------------------
        # STEP 6: MEASURE PHYSICAL PARAMETERS
        # ----------------------------------------------------

        snr = safe_float(
            analysis.get(
                "snr_db",
                np.nan
            )
        )

        inband_snr = safe_float(
            analysis.get(
                "inband_snr_db",
                np.nan
            )
        )

        frequency_offset = safe_float(
            analysis.get(
                "frequency_offset",
                np.nan
            )
        )

        symbol_rate = safe_float(
            analysis.get(
                "symbol_rate",
                np.nan
            )
        )

        occupied_bandwidth = safe_float(
            analysis.get(
                "occupied_bandwidth",
                np.nan
            )
        )

        signal_power = safe_float(
            analysis.get(
                "signal_power",
                np.nan
            )
        )

        noise_power = safe_float(
            analysis.get(
                "noise_power",
                np.nan
            )
        )

        noise_floor_db = safe_float(
            analysis.get(
                "noise_floor_db",
                np.nan
            )
        )

        # ----------------------------------------------------
        # STEP 7: RESULT
        # ----------------------------------------------------

        correct = (
            predicted == true_label
        )

        print(
            f"Predicted: {predicted}"
        )

        print(
            f"Confidence: "
            f"{confidence:.2f}%"
            if np.isfinite(confidence)
            else "Confidence: N/A"
        )

        print(
            f"SNR: "
            f"{snr:.2f} dB"
            if np.isfinite(snr)
            else "SNR: N/A"
        )

        print(
            f"Frequency offset: "
            f"{frequency_offset:.2f} Hz"
            if np.isfinite(frequency_offset)
            else "Frequency offset: N/A"
        )

        print(
            "Result: "
            + (
                "CORRECT"
                if correct
                else "WRONG"
            )
        )

        row = {
            "filename":
                str(path.relative_to(ROOT)),

            "true_label":
                true_label,

            "predicted_label":
                predicted,

            "correct":
                int(correct),

            "confidence_percent":
                confidence,

            "snr_db":
                snr,

            "inband_snr_db":
                inband_snr,

            "frequency_offset_hz":
                frequency_offset,

            "symbol_rate":
                symbol_rate,

            "occupied_bandwidth_hz":
                occupied_bandwidth,

            "signal_power":
                signal_power,

            "noise_power":
                noise_power,

            "noise_floor_db":
                noise_floor_db,
        }

        # Add probability for every class.

        for label in CLASSES:

            row[
                f"prob_{label}"
            ] = probabilities.get(
                label,
                np.nan
            )

        return row

    except Exception as exc:

        print(
            f"[ERROR] {path.name}"
        )

        print(
            str(exc)
        )

        traceback.print_exc()

        return {
            "filename":
                str(path.relative_to(ROOT)),

            "true_label":
                true_label,

            "predicted_label":
                "",

            "correct":
                0,

            "confidence_percent":
                np.nan,

            "snr_db":
                np.nan,

            "inband_snr_db":
                np.nan,

            "frequency_offset_hz":
                np.nan,

            "symbol_rate":
                np.nan,

            "occupied_bandwidth_hz":
                np.nan,

            "signal_power":
                np.nan,

            "noise_power":
                np.nan,

            "noise_floor_db":
                np.nan,

            **{
                f"prob_{label}":
                    np.nan
                for label in CLASSES
            },

            "error":
                str(exc)
        }


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    filename,
    rows
):

    output = (
        RESULTS_DIR /
        filename
    )

    if not rows:
        return

    fieldnames = []

    for row in rows:

        for key in row:

            if key not in fieldnames:

                fieldnames.append(
                    key
                )

    with output.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)

    print(
        f"\nSaved: {output}"
    )


# ============================================================
# ACCURACY
# ============================================================

def calculate_accuracy(rows):

    print()
    print(
        "=" * 60
    )

    print(
        "SPECTRA v16 MODULATION VALIDATION"
    )

    print(
        "=" * 60
    )

    total = len(rows)

    successful = [
        r
        for r in rows
        if r.get("predicted_label")
    ]

    correct = sum(
        int(r["correct"])
        for r in successful
    )

    overall = (
        100.0 * correct /
        max(len(successful), 1)
    )

    print(
        f"\nFiles tested : {total}"
    )

    print(
        f"Classified   : "
        f"{len(successful)}"
    )

    print(
        f"Correct      : "
        f"{correct}"
    )

    print(
        f"Overall      : "
        f"{overall:.2f}%"
    )

    print()
    print(
        "PER-CLASS ACCURACY"
    )

    print(
        "-" * 60
    )

    class_rows = []

    for label in CLASSES:

        class_items = [
            r
            for r in successful
            if r["true_label"] == label
        ]

        n = len(
            class_items
        )

        c = sum(
            int(r["correct"])
            for r in class_items
        )

        accuracy = (
            100.0 * c / n
            if n
            else np.nan
        )

        class_rows.append(
            {
                "class": label,
                "files": n,
                "correct": c,
                "accuracy_percent":
                    accuracy,
            }
        )

        if n:

            print(
                f"{label:8s} "
                f"{c:4d}/{n:<4d} "
                f"{accuracy:7.2f}%"
            )

        else:

            print(
                f"{label:8s} "
                "NO FILES"
            )

    save_csv(
        "accuracy_by_class.csv",
        class_rows
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def calculate_confusion_matrix(
    rows
):

    matrix = {
        true_label: {
            predicted_label: 0
            for predicted_label in CLASSES
        }
        for true_label in CLASSES
    }

    for row in rows:

        true_label = row[
            "true_label"
        ]

        predicted_label = row[
            "predicted_label"
        ]

        if (
            true_label in matrix
            and
            predicted_label in CLASSES
        ):

            matrix[
                true_label
            ][
                predicted_label
            ] += 1

    print()
    print(
        "CONFUSION MATRIX"
    )

    print(
        "-" * 60
    )

    header = (
        "True\\Pred".ljust(12)
    )

    for label in CLASSES:

        header += (
            f"{label:>10s}"
        )

    print(header)

    csv_rows = []

    for true_label in CLASSES:

        line = true_label.ljust(
            12
        )

        csv_row = {
            "true_label":
                true_label
        }

        for predicted_label in CLASSES:

            value = matrix[
                true_label
            ][
                predicted_label
            ]

            line += (
                f"{value:>10d}"
            )

            csv_row[
                predicted_label
            ] = value

        print(line)

        csv_rows.append(
            csv_row
        )

    save_csv(
        "confusion_matrix.csv",
        csv_rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "SPECTRA v16 validation starting..."
    )

    print(
        f"Validation directory:"
        f"\n{VALIDATION_DIR}"
    )

    files = discover_files()

    if not files:

        print()
        print(
            "No validation files found."
        )

        sys.exit(1)

    print()
    print(
        f"Found {len(files)} "
        f"validation files."
    )

    # --------------------------------------------------------
    # Load the SAME model used by v16
    # --------------------------------------------------------

    model_path = (
        ROOT /
        "models" /
        "modulation_rf_v16_data50_new.joblib"
    )

    classifier = (
        ModulationClassifier(
            str(model_path)
        )
    )

    print(
        "Classifier loaded."
    )

    rows = []

    for path, true_label in files:

        row = validate_file(
            classifier,
            path,
            true_label
        )

        rows.append(
            row
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    save_csv(
        "validation_results.csv",
        rows
    )

    calculate_accuracy(
        rows
    )

    calculate_confusion_matrix(
        rows
    )

    print()
    print(
        "=" * 60
    )

    print(
        "VALIDATION COMPLETE"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    main()