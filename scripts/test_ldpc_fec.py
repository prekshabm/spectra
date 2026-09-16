import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from backend.analysis.ldpc_fec import (
    ldpc_encode,
    ldpc_decode,
    ldpc_syndrome,
    introduce_bit_errors,
    bit_errors,
)


print("=" * 70)
print("SPECTRA LDPC FEC TEST")
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
    2,
    120,
    dtype=np.uint8,
)

encoded = ldpc_encode(
    original
)

syndrome = ldpc_syndrome(
    encoded
)

decoded = ldpc_decode(
    encoded
)

errors = bit_errors(
    original,
    decoded
)

print(
    "Original bits :",
    len(original)
)

print(
    "Encoded bits  :",
    len(encoded)
)

print(
    "Syndrome errors:",
    int(np.count_nonzero(syndrome))
)

print(
    "Decoded bits  :",
    len(decoded)
)

print(
    "Bit errors    :",
    errors
)

print(
    "RESULT        :",
    "PASS"
    if errors == 0
    else "FAIL"
)


# ============================================================
# TEST 2 — SINGLE BIT ERRORS
# ============================================================

print()
print("-" * 70)
print("TEST 2: SINGLE BIT ERRORS")
print("-" * 70)

original = rng.integers(
    0,
    2,
    120,
    dtype=np.uint8,
)

encoded = ldpc_encode(
    original
)

positions = [
    3,
    27,
    51,
    79,
    101,
]

corrupted = introduce_bit_errors(
    encoded,
    positions
)

decoded = ldpc_decode(
    corrupted
)

errors = bit_errors(
    original,
    decoded
)

print(
    "Injected errors:",
    len(positions)
)

print(
    "Remaining errors:",
    errors
)

print(
    "RESULT         :",
    "PASS"
    if errors == 0
    else "FAIL"
)


# ============================================================
# TEST 3 — MULTIPLE MESSAGE SIZES
# ============================================================

print()
print("-" * 70)
print("TEST 3: MULTIPLE MESSAGE SIZES")
print("-" * 70)

all_pass = True

for length in [
    12,
    24,
    48,
    96,
    120,
]:

    original = rng.integers(
        0,
        2,
        length,
        dtype=np.uint8,
    )

    encoded = ldpc_encode(
        original
    )

    decoded = ldpc_decode(
        encoded
    )

    errors = bit_errors(
        original,
        decoded
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
    "PASS"
    if all_pass
    else "FAIL"
)

print("=" * 70)