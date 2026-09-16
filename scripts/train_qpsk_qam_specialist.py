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


DATA_DIR = ROOT / "data"
VALIDATION_DIR = ROOT / "validation_50"

MODEL_PATH = ROOT / "models" / "specialist_qpsk_qam.joblib"

SAMPLE_RATE = 1_000_000

CLASS_FOLDERS = {
    "qpsk": "QPSK",
    "16qam": "16-QAM",
}


def load_iq(path):
    signal, fs, _ = load_signal(
        path.name,
        path.read_bytes(),
        SAMPLE_RATE,
        dtype="float32",
        iq_format="IQ",
    )

    return np.asarray(signal), float(fs)


X = []
y = []

print("=" * 70)
print("SPECTRA V17 - QPSK / 16-QAM SPECIALIST")
print("=" * 70)

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

    print()
    print(f"{class_name}")
    print(f"Total files : {len(files)}")
    print(f"Validation  : {len(validation_names)}")
    print(f"Training    : {len(training_files)}")

    if len(training_files) != 50:
        raise RuntimeError(
            f"{class_name}: expected 50 training files"
        )

    for path in training_files:

        signal, fs = load_iq(path)

        if len(signal) < 64:
            continue

        features = _extract(signal, fs)

        features = np.nan_to_num(
            features,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )

        X.append(features)
        y.append(class_name)


X = np.asarray(X, dtype=float)
y = np.asarray(y)

print()
print("=" * 70)
print("DATASET")
print("=" * 70)

print("Samples:", len(X))
print("Features:", X.shape[1])

print()
print("Training specialist...")


model = RandomForestClassifier(
    n_estimators=250,
    max_depth=18,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X, y)


MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "classes": list(model.classes_),
        "training_source": (
            "50 real training files/class; "
            "validation files excluded"
        ),
    },
    MODEL_PATH,
)


print()
print("=" * 70)
print("SPECIALIST SAVED")
print("=" * 70)

print(MODEL_PATH)