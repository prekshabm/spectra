from pathlib import Path
import sys
import numpy as np

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORTS
# ============================================================

from backend.signal_io.loader import load_signal
from backend.ml.classifier import ModulationClassifier, CLASSES
from backend.ml.fsk_specialist import extract_fsk_features


# ============================================================
# PATHS
# ============================================================

VALIDATION_DIR = (
    ROOT / "validation_50"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "modulation_rf_v16_data50_new.joblib"
)


# ============================================================
# DEFAULT V16 SETTINGS
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
# SAME STRUCTURE AS REAL VALIDATION SCRIPT
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
# LOAD SIGNAL
# SAME LOADING METHOD AS VALIDATION SCRIPT
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
        meta
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SPECTRA FSK SPECIALIST DIAGNOSTIC")
    print("=" * 70)

    print()
    print(
        "Validation directory:"
    )
    print(
        VALIDATION_DIR
    )

    print()
    print(
        "Model:"
    )
    print(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Discover
    # --------------------------------------------------------

    files = discover_files()

    print()

    print(
        f"Found {len(files)} validation files."
    )

    if not files:

        print()
        print(
            "ERROR: No validation files found."
        )
        print()

        return

    # --------------------------------------------------------
    # Load classifier
    # --------------------------------------------------------

    print()

    classifier = ModulationClassifier(
        str(MODEL_PATH)
    )

    print(
        "Classifier loaded."
    )

    # --------------------------------------------------------
    # Check FSK specialist
    # --------------------------------------------------------

    if classifier.fsk_specialist is None:

        print()
        print(
            "ERROR: specialist_fsk_real.joblib "
            "was not loaded."
        )
        print()

        return

    print(
        "Real-data FSK specialist loaded."
    )

    print()
    print("-" * 110)

    print(
        f"{'FILE':<38}"
        f"{'TRUE':<9}"
        f"{'MAIN':<9}"
        f"{'MAIN %':>8}"
        f"{'FSK':<9}"
        f"{'FSK %':>8}"
        f"  STATUS"
    )

    print("-" * 110)

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total = 0
    main_correct = 0
    specialist_correct = 0

    fsk_total = 0
    fsk_specialist_correct = 0

    # --------------------------------------------------------
    # Process every validation signal
    # --------------------------------------------------------

    for path, true_label in files:

        total += 1

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        try:

            signal, fs, meta = (
                load_v16_signal(path)
            )

        except Exception as e:

            print()

            print(
                f"ERROR loading "
                f"{path.name}: {e}"
            )

            continue

        # ----------------------------------------------------
        # Main classifier
        # ----------------------------------------------------

        try:

            result = classifier.predict(
                None,
                signal,
                fs
            )

            main_prediction = (
                result.get(
                    "detected",
                    ""
                )
            )

            main_confidence = float(
                result.get(
                    "confidence",
                    0.0
                )
            )

        except Exception as e:

            print()

            print(
                f"ERROR classifier on "
                f"{path.name}: {e}"
            )

            continue

        # ----------------------------------------------------
        # FSK specialist
        # ----------------------------------------------------

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
                classifier
                .fsk_specialist
                .predict_proba(
                    fsk_x.reshape(
                        1,
                        -1
                    )
                )[0]
            )

            fsk_classes = list(
                classifier
                .fsk_specialist
                .classes_
            )

            fsk_scores = dict(
                zip(
                    fsk_classes,
                    fsk_prob
                )
            )

            score_2 = float(
                fsk_scores.get(
                    "2-FSK",
                    0.0
                )
            )

            score_4 = float(
                fsk_scores.get(
                    "4-FSK",
                    0.0
                )
            )

            if score_4 >= score_2:

                specialist_prediction = (
                    "4-FSK"
                )

                specialist_confidence = (
                    score_4 * 100.0
                )

            else:

                specialist_prediction = (
                    "2-FSK"
                )

                specialist_confidence = (
                    score_2 * 100.0
                )

        except Exception as e:

            print()

            print(
                f"ERROR specialist on "
                f"{path.name}: {e}"
            )

            continue

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        if main_prediction == true_label:

            main_correct += 1

        if (
            specialist_prediction
            == true_label
        ):

            specialist_correct += 1

        # ----------------------------------------------------
        # FSK-only statistics
        # ----------------------------------------------------

        if true_label in (
            "2-FSK",
            "4-FSK"
        ):

            fsk_total += 1

            if (
                specialist_prediction
                == true_label
            ):

                fsk_specialist_correct += 1

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        status = ""

        if (
            true_label in (
                "2-FSK",
                "4-FSK"
            )
        ):

            if main_prediction != true_label:

                if (
                    specialist_prediction
                    == true_label
                ):

                    status = (
                        "<<< SPECIALIST RESCUE"
                    )

                else:

                    status = (
                        "<<< BOTH WRONG"
                    )

            elif (
                specialist_prediction
                != true_label
            ):

                status = (
                    "<<< SPECIALIST WRONG"
                )

            else:

                status = "OK"

        else:

            status = "-"

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"{path.name:<38}"
            f"{true_label:<9}"
            f"{main_prediction:<9}"
            f"{main_confidence:>7.2f}%"
            f"{specialist_prediction:<9}"
            f"{specialist_confidence:>7.2f}%"
            f"  {status}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("-" * 110)

    if total > 0:

        main_accuracy = (
            main_correct
            / total
            * 100.0
        )

        specialist_accuracy = (
            specialist_correct
            / total
            * 100.0
        )

    else:

        main_accuracy = 0.0
        specialist_accuracy = 0.0

    print()
    print(
        "DIAGNOSTIC SUMMARY"
    )

    print(
        f"Total files               : {total}"
    )

    print(
        f"Main classifier correct   : "
        f"{main_correct}/{total} "
        f"= {main_accuracy:.2f}%"
    )

    print(
        f"FSK specialist correct   : "
        f"{specialist_correct}/{total} "
        f"= {specialist_accuracy:.2f}%"
    )

    print()

    print(
        f"FSK files                 : "
        f"{fsk_total}"
    )

    if fsk_total > 0:

        fsk_accuracy = (
            fsk_specialist_correct
            / fsk_total
            * 100.0
        )

        print(
            f"FSK specialist accuracy  : "
            f"{fsk_specialist_correct}/"
            f"{fsk_total} "
            f"= {fsk_accuracy:.2f}%"
        )

    print()
    print("=" * 70)
    print(
        "DIAGNOSTIC COMPLETE"
    )
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()