from pathlib import Path
import sys
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ml.classifier import _extract, _expert_scores, ModulationClassifier


FEATURE_NAMES = [
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
]


TARGET_FILES = [
    "QPSK_33_SNR_00dB.iq",
    "QPSK_30_SNR_11dB.iq",
    "2FSK_1239072Hz_26.5dB.iq",
    "2FSK_29_SNR_09dB.iq",
]


def find_file(filename):
    matches = list((ROOT / "validation_50").rglob(filename))
    return matches[0] if matches else None


def load_iq_file(path):
    raw = path.read_bytes()
    a = np.frombuffer(raw, dtype=np.float32)

    if len(a) % 2 != 0:
        a = a[:-1]

    signal = a[0::2] + 1j * a[1::2]

    return signal.astype(np.complex64), 1_000_000.0


def inspect_file(path):
    print("\n" + "=" * 75)
    print(f"FILE: {path.name}")
    print("=" * 75)

    try:
        signal, fs = load_iq_file(path)

        features = _extract(signal, fs)

        print(f"Samples: {len(signal)}")
        print(f"Sample rate: {fs:.0f} Hz")

        # ---------------------------------------------------------
        # FEATURE VALUES
        # ---------------------------------------------------------
        print("\nFEATURES")
        print("-" * 75)

        for name, value in zip(FEATURE_NAMES, features):
            print(f"{name:22s} = {value: .8f}")

        # ---------------------------------------------------------
        # RANDOM FOREST
        # ---------------------------------------------------------
        model_path = ROOT / "models" / "modulation_rf.joblib"

        obj = joblib.load(model_path)
        model = obj["model"]

        x = np.nan_to_num(
            features,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )

        rf = model.predict_proba(x.reshape(1, -1))[0]
        classes = list(model.classes_)

        print("\nRANDOM FOREST PROBABILITIES")
        print("-" * 75)

        rf_dict = dict(zip(classes, rf))

        for cls, probability in sorted(
            rf_dict.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"{cls:10s} = {probability * 100:6.2f}%")

        # ---------------------------------------------------------
        # PHYSICS / EXPERT SCORES
        # ---------------------------------------------------------
        expert = _expert_scores(x)

        print("\nPHYSICS / EXPERT SCORES")
        print("-" * 75)

        for cls, score in sorted(
            expert.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"{cls:10s} = {score:.6f}")

        # ---------------------------------------------------------
        # FULL SPECTRA CLASSIFIER
        # ---------------------------------------------------------
        print("\nFULL SPECTRA CLASSIFIER RESULT")
        print("-" * 75)

        classifier = ModulationClassifier(model_path)

        result = classifier.predict(
            features,
            signal,
            fs,
        )

        print(f"FINAL DETECTION: {result['detected']}")
        print(f"FINAL CONFIDENCE: {result['confidence']:.2f}%")

        print("\nFINAL RANKING")
        print("-" * 75)

        for row in result["candidates"]:
            print(
                f"{row['modulation']:10s} = "
                f"{row['probability']:6.2f}%"
            )

    except Exception as exc:
        print(f"ERROR processing {path.name}: {exc}")


def main():
    for filename in TARGET_FILES:

        path = find_file(filename)

        if path is None:
            print(f"\nNOT FOUND: {filename}")
            continue

        inspect_file(path)


if __name__ == "__main__":
    main()