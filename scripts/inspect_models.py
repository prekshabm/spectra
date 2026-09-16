from pathlib import Path
import sys
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_PATHS = [
    ROOT / "models" / "modulation_rf.joblib",
    ROOT / "models" / "modulation_rf_v16_data50_new.joblib",
]

for path in MODEL_PATHS:
    print("\n" + "=" * 70)
    print(path.name)
    print("=" * 70)

    bundle = joblib.load(path)

    print("Type:", type(bundle))

    if isinstance(bundle, dict):
        print("Keys:")
        for key, value in bundle.items():
            print(f"  {key}: {type(value)}")
    else:
        print("Attributes:")
        print("classes_:", getattr(bundle, "classes_", "NOT FOUND"))