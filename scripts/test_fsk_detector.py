from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ml.fsk_detector import estimate_fsk_states
from scripts.validate_model import (
    discover_files,
    load_v16_signal,
)


print("=" * 75)
print("SPECTRA V17.2 - FSK STATE DETECTOR TEST")
print("=" * 75)

files = discover_files()

fsk_files = [
    item
    for item in files
    if item[1] in (
        "2-FSK",
        "4-FSK"
    )
]

correct = 0
total = 0

print(
    f"\nFSK validation files: "
    f"{len(fsk_files)}"
)

for path, true_label in fsk_files:

    print("\n" + "-" * 75)
    print(
        f"File: {path.name}"
    )

    signal, fs, _ = (
        load_v16_signal(path)
    )

    result = estimate_fsk_states(
        signal,
        fs
    )

    predicted = (
        "2-FSK"
        if result["state_count"] == 2
        else "4-FSK"
    )

    ok = (
        predicted
        == true_label
    )

    if ok:
        correct += 1

    total += 1

    print(
        f"True      : {true_label}"
    )

    print(
        f"Predicted : {predicted}"
    )

    print(
        f"Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        f"2-state   : "
        f"{result['score_2'] * 100:.2f}%"
    )

    print(
        f"4-state   : "
        f"{result['score_4'] * 100:.2f}%"
    )

    print(
        f"BIC(2)    : "
        f"{result['bic_2']:.2f}"
    )

    print(
        f"BIC(4)    : "
        f"{result['bic_4']:.2f}"
    )

    print(
        "Result    : "
        + (
            "CORRECT"
            if ok
            else "WRONG"
        )
    )


accuracy = (
    100.0
    * correct
    / max(total, 1)
)

print("\n" + "=" * 75)
print("V17.2 FSK DETECTOR RESULT")
print("=" * 75)

print(
    f"Correct : {correct}/{total}"
)

print(
    f"Accuracy: {accuracy:.2f}%"
)