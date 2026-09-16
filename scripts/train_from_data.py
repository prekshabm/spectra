from pathlib import Path
import sys

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ============================================================
# PATH SETUP
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ml.classifier import _extract, FEATURES, CLASSES
from backend.signal_io.loader import load_signal


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = ROOT / "data"

# IMPORTANT:
# This is a NEW model. Your original V16 model is untouched.
MODEL_PATH = ROOT / "models" / "modulation_rf_v16_data50.joblib"

SAMPLE_RATE = 1_000_000

# Folder name -> V16 class name
CLASS_FOLDERS = {
    "bpsk": "BPSK",
    "qpsk": "QPSK",
    "2fsk": "2-FSK",
    "4fsk": "4-FSK",
    "16qam": "16-QAM",
}


# ============================================================
# LOAD ONE IQ FILE
# ============================================================

def load_iq_file(path):
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
# EXTRACT V16 FEATURES
# ============================================================

def extract_features(signal, fs):
    features = _extract(signal, fs)

    features = np.asarray(features, dtype=float)

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )

    return features


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("SPECTRA V16 - TRAINING FROM REAL IQ DATA")
print("=" * 70)

print()
print("Classes:")
for c in CLASSES:
    print(f"  {c}")

print()
print("Training data directory:")
print(DATA_DIR)

print()
print("Output model:")
print(MODEL_PATH)

print()
print("Loading real IQ files...")
print("-" * 70)


X = []
y = []

file_counts = {}

# ============================================================
# READ ALL FIVE CLASS DIRECTORIES
# ============================================================

for folder_name, class_name in CLASS_FOLDERS.items():

    folder = DATA_DIR / folder_name

    if not folder.exists():
        print(f"[ERROR] Missing folder: {folder}")
        continue

    files = sorted(folder.glob("*.iq"))

    file_counts[class_name] = len(files)

    print()
    print(f"{class_name}: {len(files)} files")

    for i, path in enumerate(files, start=1):

        try:
            signal, fs = load_iq_file(path)

            if len(signal) < 64:
                print(f"  [SKIP] {path.name} - signal too short")
                continue

            feat = extract_features(signal, fs)

            X.append(feat)
            y.append(class_name)

            print(
                f"  [{i:02d}/{len(files):02d}] "
                f"{path.name} "
                f"-> {len(signal):,} samples"
            )

        except Exception as e:
            print(f"  [ERROR] {path.name}: {e}")


# ============================================================
# CHECK DATASET
# ============================================================

print()
print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print()

for class_name in CLASSES:
    count = y.count(class_name)
    print(f"{class_name:<10} {count:>3} files loaded")

if len(X) == 0:
    raise RuntimeError("No training data was loaded.")

X = np.asarray(X, dtype=float)
y = np.asarray(y)

print()
print(f"Total samples:   {len(X)}")
print(f"Features/sample: {X.shape[1]}")

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print()
print("=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print()
print(f"Training examples : {len(X_train)}")
print(f"Testing examples  : {len(X_test)}")


# ============================================================
# RANDOM FOREST
# ============================================================

print()
print("=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

print()
print("Training...")
clf.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# HOLDOUT TEST
# ============================================================

print()
print("=" * 70)
print("HOLDOUT TEST RESULTS")
print("=" * 70)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print()
print(f"Holdout accuracy: {accuracy * 100:.2f}%")

print()
print("Classification report:")
print()

print(
    classification_report(
        y_test,
        y_pred,
        labels=CLASSES,
        zero_division=0,
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": clf,
        "features": FEATURES,
        "classes": CLASSES,
        "training_source": "real IQ data",
        "training_files_per_class": file_counts,
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
print("Original V16 model was NOT modified:")
print(ROOT / "models" / "modulation_rf.joblib")

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)