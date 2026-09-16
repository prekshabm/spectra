import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from backend.analysis.concatenated_fec import (
    concatenated_encode,
    concatenated_decode,
    introduce_bit_errors,
    byte_errors,
)


print("=" * 70)
print("SPECTRA CONCATENATED FEC TEST")
print("=" * 70)


rng = np.random.default_rng(42)


# ============================================================
# TEST 1 — PERFECT CHANNEL
# ============================================================

print()
print("-" * 70)
print("TEST 1: PERFECT CHANNEL")
print("-" * 70)

original = rng.integers(
    0,
    256,
    50,
    dtype=np.uint8,
)

encoded = concatenated_encode(
    original,
    nsym=10,
)

decoded = concatenated_decode(
    encoded,
    nsym=10,
)

errors = byte_errors(
    original,
    decoded,
)

print("Original bytes :", len(original))
print("Encoded bits   :", len(encoded))
print("Decoded bytes  :", len(decoded))
print("Byte errors    :", errors)

print(
    "RESULT         :",
    "PASS" if errors == 0 else "FAIL"
)


# ============================================================
# TEST 2 — BIT ERRORS
# ============================================================

print()
print("-" * 70)
print("TEST 2: BIT ERRORS IN CHANNEL")
print("-" * 70)

original = rng.integers(
    0,
    256,
    50,
    dtype=np.uint8,
)

encoded = concatenated_encode(
    original,
    nsym=10,
)

# Add 50 random bit errors.
positions = rng.choice(
    len(encoded),
    size=50,
    replace=False,
)

corrupted = introduce_bit_errors(
    encoded,
    positions,
)

decoded = concatenated_decode(
    corrupted,
    nsym=10,
)

errors = byte_errors(
    original,
    decoded,
)

print("Original bytes :", len(original))
print("Encoded bits   :", len(encoded))
print("Injected errors:", len(positions))
print("Remaining errors:", errors)

print(
    "RESULT         :",
    "PASS" if errors == 0 else "FAIL"
)


# ============================================================
# TEST 3 — DIFFERENT MESSAGE LENGTHS
# ============================================================

print()
print("-" * 70)
print("TEST 3: MULTIPLE MESSAGE LENGTHS")
print("-" * 70)

all_pass = True

for length in [
    10,
    20,
    50,
    100,
]:

    original = rng.integers(
        0,
        256,
        length,
        dtype=np.uint8,
    )

    encoded = concatenated_encode(
        original,
        nsym=10,
    )

    decoded = concatenated_decode(
        encoded,
        nsym=10,
    )

    errors = byte_errors(
        original,
        decoded,
    )

    passed = errors == 0

    if not passed:
        all_pass = False

    print(
        f"Length={length:3d} | "
        f"Encoded bits={len(encoded):5d} | "
        f"Errors={errors:2d} | "
        f"{'PASS' if passed else 'FAIL'}"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print(
    "FINAL RESULT:",
    "PASS" if all_pass else "FAIL"
)
print("=" * 70)