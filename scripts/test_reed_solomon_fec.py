import sys
from pathlib import Path

import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from backend.analysis.reed_solomon_fec import (
    reed_solomon_encode,
    reed_solomon_decode,
    introduce_errors,
    byte_errors,
)


# ============================================================
# RANDOM GENERATOR
# ============================================================

rng = np.random.default_rng(42)


print("=" * 70)
print("SPECTRA REED-SOLOMON FEC TEST")
print("=" * 70)


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
    100,
    dtype=np.uint8,
)

encoded = reed_solomon_encode(
    original,
    nsym=10,
)

decoded = reed_solomon_decode(
    encoded,
    nsym=10,
)

errors = byte_errors(
    original,
    decoded,
)

print("Original bytes :", len(original))
print("Encoded bytes  :", len(encoded))
print("Decoded bytes  :", len(decoded))
print("Byte errors    :", errors)

print(
    "RESULT         :",
    "PASS" if errors == 0 else "FAIL"
)


# ============================================================
# TEST 2 — 5 BYTE ERRORS
# ============================================================

print()
print("-" * 70)
print("TEST 2: 5 BYTE ERRORS")
print("-" * 70)

original = rng.integers(
    0,
    256,
    100,
    dtype=np.uint8,
)

encoded = reed_solomon_encode(
    original,
    nsym=10,
)

corrupted = introduce_errors(
    encoded,
    positions=[
        3,
        17,
        31,
        58,
        91,
    ],
)

decoded = reed_solomon_decode(
    corrupted,
    nsym=10,
)

errors = byte_errors(
    original,
    decoded,
)

print("Original bytes :", len(original))
print("Encoded bytes  :", len(encoded))
print("Injected errors: 5")
print("Remaining errors:", errors)

print(
    "RESULT         :",
    "PASS" if errors == 0 else "FAIL"
)


# ============================================================
# TEST 3 — MAXIMUM CORRECTABLE ERRORS
# ============================================================

print()
print("-" * 70)
print("TEST 3: 5 BYTE ERRORS (MAXIMUM FOR nsym=10)")
print("-" * 70)

original = rng.integers(
    0,
    256,
    100,
    dtype=np.uint8,
)

encoded = reed_solomon_encode(
    original,
    nsym=10,
)

positions = np.array(
    [
        1,
        15,
        34,
        67,
        96,
    ]
)

corrupted = introduce_errors(
    encoded,
    positions,
)

decoded = reed_solomon_decode(
    corrupted,
    nsym=10,
)

errors = byte_errors(
    original,
    decoded,
)

print("Injected errors :", 5)
print("Remaining errors:", errors)

print(
    "RESULT          :",
    "PASS" if errors == 0 else "FAIL"
)


# ============================================================
# TEST 4 — MULTIPLE MESSAGE LENGTHS
# ============================================================

print()
print("-" * 70)
print("TEST 4: MULTIPLE MESSAGE LENGTHS")
print("-" * 70)

all_pass = True

for length in [
    10,
    20,
    50,
    100,
    200,
]:

    original = rng.integers(
        0,
        256,
        length,
        dtype=np.uint8,
    )

    encoded = reed_solomon_encode(
        original,
        nsym=10,
    )

    decoded = reed_solomon_decode(
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
        f"Encoded={len(encoded):3d} | "
        f"Errors={errors:2d} | "
        f"{'PASS' if passed else 'FAIL'}"
    )


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)
print(
    "FINAL RESULT:",
    "PASS" if all_pass else "FAIL"
)
print("=" * 70)