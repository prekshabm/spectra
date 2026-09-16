from pathlib import Path
import sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ml.classifier import _extract, FEATURES
from backend.signal_io.loader import load_signal


# ============================================================
# PATHS
# ============================================================

DATA_DIR = ROOT / "data"
VALIDATION_DIR = ROOT / "validation_50"

MODEL_PATH = (
    ROOT
    / "models"
    / "specialist_fsk.joblib"
)

SAMPLE_RATE = 1_000_000


# ============================================================
# REAL DATA FOLDERS
# ============================================================

CLASS_FOLDERS = {
    "2fsk": "2-FSK",
    "4fsk": "4-FSK",
}


# ============================================================
# LOAD IQ
# ============================================================

def load_iq(path):

    signal, fs, _ = load_signal(
        path.name,
        path.read_bytes(),
        SAMPLE_RATE,
        dtype="float32",
        iq_format="IQ",
    )

    return np.asarray(signal), float(fs)


# ============================================================
# TRAINING DATA
# ============================================================

X = []
y = []

print("=" * 70)
print("SPECTRA V17.1 - 2-FSK / 4-FSK SPECIALIST")
print("=" * 70)


for folder_name, class_name in CLASS_FOLDERS.items():

    source_dir = (
        DATA_DIR
        / folder_name
    )

    validation_dir = (
        VALIDATION_DIR
        / class_name
    )

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found:\n{source_dir}"
        )

    if not validation_dir.exists():
        raise FileNotFoundError(
            f"Validation directory not found:\n{validation_dir}"
        )

    validation_names = {
        p.name
        for p in validation_dir.glob("*.iq")
    }

    files = sorted(
        source_dir.glob("*.iq")
    )

    training_files = [
        p
        for p in files
        if p.name not in validation_names
    ]

    print()
    print(class_name)
    print("-" * 40)
    print(f"Total files      : {len(files)}")
    print(f"Validation files : {len(validation_names)}")
    print(f"Training files   : {len(training_files)}")

    if len(training_files) != 50:
        raise RuntimeError(
            f"{class_name}: expected exactly 50 "
            f"training files after excluding validation."
        )

    for path in training_files:

        try:

            signal, fs = load_iq(path)

            if len(signal) < 64:
                print(
                    f"Skipping short file: {path.name}"
                )
                continue

            features = _extract(
                signal,
                fs
            )

            features = np.nan_to_num(
                features,
                nan=0.0,
                posinf=1e6,
                neginf=-1e6,
            )

            X.append(features)
            y.append(class_name)

        except Exception as exc:

            print(
                f"Skipping {path.name}: {exc}"
            )


# ============================================================
# DATASET SUMMARY
# ============================================================

X = np.asarray(
    X,
    dtype=float
)

y = np.asarray(
    y
)

print()
print("=" * 70)
print("DATASET READY")
print("=" * 70)

print(f"Samples  : {len(X)}")
print(f"Features : {X.shape[1]}")

print()
print("Class distribution:")

for cls in CLASS_FOLDERS.values():

    print(
        f"  {cls}: "
        f"{np.sum(y == cls)}"
    )


# ============================================================
# TRAIN SPECIALIST
# ============================================================

print()
print("=" * 70)
print("TRAINING FSK SPECIALIST")
print("=" * 70)


model = RandomForestClassifier(
    n_estimators=350,
    max_depth=18,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X,
    y
)


# ============================================================
# SAVE
# ============================================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "classes": list(model.classes_),
        "training_source": (
            "100 real FSK training files "
            "(50 per class); "
            "validation files excluded"
        ),
    },
    MODEL_PATH
)


print()
print("=" * 70)
print("FSK SPECIALIST SAVED")
print("=" * 70)

print(
    MODEL_PATH
)