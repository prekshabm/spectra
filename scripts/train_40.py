from pathlib import Path
import sys
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ml.classifier import _extract, FEATURES, CLASSES
from backend.signal_io.loader import load_signal


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = ROOT / "data"
VALIDATION_DIR = ROOT / "validation_50"

MODEL_PATH = ROOT / "models" / "modulation_rf_v16_data40.joblib"

SAMPLE_RATE = 1_000_000

CLASS_FOLDERS = {
    "bpsk": "BPSK",
    "qpsk": "QPSK",
    "2fsk": "2-FSK",
    "4fsk": "4-FSK",
    "16qam": "16-QAM",
}


# ============================================================
# LOAD IQ
# ============================================================

def load_iq(path):
    raw = path.read_bytes()

    signal, fs, meta = load_signal(
        path.name,
        raw,
        SAMPLE_RATE,
        dtype="float32",
        iq_format="IQ",
    )

    return signal, fs


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def get_features(signal, fs):
    f = _extract(signal, fs)

    f = np.asarray(f, dtype=float)

    f = np.nan_to_num(
        f,
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )

    return f


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("SPECTRA V16 — TRAINING ON 40 FILES PER CLASS")
print("=" * 70)

print()
print("Validation files will be EXCLUDED from training.")
print()

X = []
y = []

total_loaded = 0


for folder_name, class_name in CLASS_FOLDERS.items():

    source = DATA_DIR / folder_name
    validation_source = VALIDATION_DIR / class_name

    validation_names = {
        p.name for p in validation_source.glob("*.iq")
    }

    files = sorted(source.glob("*.iq"))

    training_files = [
        p for p in files
        if p.name not in validation_names
    ]

    print("-" * 70)
    print(class_name)

    print(f"Total files : {len(files)}")
    print(f"Validation  : {len(validation_names)}")
    print(f"Training    : {len(training_files)}")

    if len(training_files) != 40:
        raise RuntimeError(
            f"{class_name}: expected 40 training files, "
            f"but found {len(training_files)}"
        )

    for path in training_files:

        try:
            signal, fs = load_iq(path)

            if len(signal) < 64:
                print(f"[SKIP] Too short: {path.name}")
                continue

            feat = get_features(signal, fs)

            X.append(feat)
            y.append(class_name)

            total_loaded += 1

        except Exception as e:
            print(f"[ERROR] {path.name}: {e}")


# ============================================================
# DATASET CHECK
# ============================================================

print()
print("=" * 70)
print("TRAINING DATASET")
print("=" * 70)

for class_name in CLASSES:
    count = y.count(class_name)
    print(f"{class_name:<10} {count:>3}")

if total_loaded != 200:
    raise RuntimeError(
        f"Expected 200 training files but loaded {total_loaded}"
    )

X = np.asarray(X, dtype=float)
y = np.asarray(y)

print()
print(f"Total training files : {len(X)}")
print(f"Features             : {X.shape[1]}")


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print()
print("=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

clf = RandomForestClassifier(
    n_estimators=600,
    max_depth=20,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

clf.fit(X, y)

print()
print("Training complete.")


# ============================================================
# SAVE
# ============================================================

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": clf,
        "features": FEATURES,
        "classes": CLASSES,
        "training_source": "real IQ data",
        "training_files_per_class": 40,
        "validation_files_per_class": 10,
        "sample_rate": SAMPLE_RATE,
    },
    MODEL_PATH,
)

print()
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print()
print(MODEL_PATH)

print()
print("Original models remain untouched:")
print(ROOT / "models" / "modulation_rf.joblib")
print(ROOT / "models" / "modulation_rf_v16_data50.joblib")

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)