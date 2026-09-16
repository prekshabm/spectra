from pathlib import Path
import sys
import joblib

# Project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from backend.ml.classifier import _extract
from scripts.validate_model import load_v16_signal, discover_files, normalize_label


MODEL_PATHS = [
    ROOT / "models" / "modulation_rf.joblib",
    ROOT / "models" / "modulation_rf_v16_data50_new.joblib",
]


def predict_raw(bundle, signal, fs):
    """
    Run ONLY the Random Forest contained inside the model bundle.

    This deliberately does NOT use ModulationClassifier.predict(),
    because that adds the physics expert + temporal voting layers.
    """

    model = bundle["model"]
    features = bundle["features"]
    classes = bundle["classes"]

    x = _extract(signal, fs)

    # Make sure the feature order matches the model.
    x_dict = dict(zip(
        [
            "c20_abs",
            "c40_abs",
            "c42_abs",
            "amp_cv",
            "amp_skew",
            "amp_kurt",
            "dphi_std",
            "dphi_kurt",
            "phase_coh2",
            "phase_coh4",
            "freq_std_norm",
            "spectral_flatness",
            "spectral_peak_ratio",
            "spectral_peaks",
            "freq_hist_peaks",
            "freq_hist_entropy",
        ],
        x
    ))

    x_ordered = np.array(
        [[x_dict[f] for f in features]],
        dtype=float
    )

    probabilities = model.predict_proba(x_ordered)[0]
    best_index = int(np.argmax(probabilities))

    predicted = classes[best_index]

    return predicted, probabilities, classes


def main():
    print("=" * 70)
    print("SPECTRA MODEL COMPARISON")
    print("=" * 70)

    files = discover_files()

    if not files:
        print("\nERROR: No validation files found.")
        return

    print(f"\nValidation files found: {len(files)}")

    all_results = {}

    for model_path in MODEL_PATHS:

        print("\n" + "-" * 70)
        print(f"Testing: {model_path.name}")
        print("-" * 70)

        if not model_path.exists():
            print("ERROR: Model not found:")
            print(model_path)
            continue

        bundle = joblib.load(model_path)

        correct = 0
        total = 0

        per_class = {}
        confusion = {}

        for path, true_label in files:

            true_label = normalize_label(true_label)

            try:
                signal, fs, meta = load_v16_signal(path)

                predicted, probabilities, classes = predict_raw(
                    bundle,
                    signal,
                    fs
                )

                predicted = normalize_label(predicted)

                total += 1

                if predicted == true_label:
                    correct += 1

                if true_label not in per_class:
                    per_class[true_label] = [0, 0]

                per_class[true_label][1] += 1

                if predicted == true_label:
                    per_class[true_label][0] += 1

                if true_label not in confusion:
                    confusion[true_label] = {}

                confusion[true_label][predicted] = (
                    confusion[true_label].get(predicted, 0) + 1
                )

            except Exception as e:
                print(f"\nERROR processing {path.name}: {e}")

        accuracy = 100.0 * correct / total if total else 0.0

        all_results[model_path.name] = accuracy

        print(f"\nAccuracy: {correct}/{total} = {accuracy:.2f}%")

        print("\nPer-class accuracy:")

        for label in sorted(per_class):
            c, t = per_class[label]
            pct = 100.0 * c / t if t else 0.0

            print(
                f"  {label:8s}: "
                f"{c}/{t} = {pct:.2f}%"
            )

        print("\nConfusion matrix:")

        labels = sorted(
            set(confusion.keys()) |
            {
                p
                for row in confusion.values()
                for p in row.keys()
            }
        )

        print("True\\Pred", end="")

        for label in labels:
            print(f"{label:>10s}", end="")

        print()

        for true_label in labels:

            print(f"{true_label:9s}", end="")

            for predicted_label in labels:

                value = confusion.get(
                    true_label,
                    {}
                ).get(
                    predicted_label,
                    0
                )

                print(f"{value:10d}", end="")

            print()

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    for model_name, accuracy in all_results.items():
        print(
            f"{model_name:40s} "
            f"{accuracy:.2f}%"
        )

    if all_results:
        best_model = max(
            all_results,
            key=all_results.get
        )

        print("\nBest model:")
        print(best_model)
        print(
            f"Accuracy: "
            f"{all_results[best_model]:.2f}%"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()