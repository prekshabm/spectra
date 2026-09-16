import numpy as np
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from backend.analysis.convolutional_fec import (
    convolutional_encode,
    viterbi_decode,
    bit_errors,
)


print("=" * 70)
print("SPECTRA CONVOLUTIONAL FEC + VITERBI TEST")
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
    256,
    dtype=np.uint8,
)

encoded = convolutional_encode(
    original,
    terminate=True,
)

decoded = viterbi_decode(
    encoded,
    terminated=True,
)

errors = bit_errors(
    original,
    decoded,
)

print(
    f"Original bits : {len(original)}"
)

print(
    f"Encoded bits  : {len(encoded)}"
)

print(
    f"Decoded bits  : {len(decoded)}"
)

print(
    f"Bit errors    : {errors}"
)

print(
    f"Result        : "
    f"{'PASS' if errors == 0 else 'FAIL'}"
)


# ============================================================
# TEST 2 — SINGLE RANDOM BIT ERRORS
# ============================================================

print()
print("-" * 70)
print("TEST 2: MODERATE BIT ERRORS")
print("-" * 70)


original = rng.integers(
    0,
    2,
    500,
    dtype=np.uint8,
)

encoded = convolutional_encode(
    original,
    terminate=True,
)

corrupted = encoded.copy()

# Introduce approximately 5% coded-bit errors.
error_count = max(
    1,
    int(0.05 * len(corrupted))
)

error_positions = rng.choice(
    len(corrupted),
    size=error_count,
    replace=False,
)

corrupted[
    error_positions
] ^= 1

decoded = viterbi_decode(
    corrupted,
    terminated=True,
)

errors = bit_errors(
    original,
    decoded,
)

print(
    f"Original bits      : {len(original)}"
)

print(
    f"Coded bits         : {len(encoded)}"
)

print(
    f"Injected errors    : {error_count}"
)

print(
    f"Remaining errors   : {errors}"
)

print(
    f"Result             : "
    f"{'PASS' if errors < error_count else 'FAIL'}"
)


# ============================================================
# TEST 3 — DIFFERENT MESSAGE LENGTHS
# ============================================================

print()
print("-" * 70)
print("TEST 3: MULTIPLE MESSAGE LENGTHS")
print("-" * 70)


for length in (
    16,
    32,
    64,
    128,
    256,
    512,
):

    original = rng.integers(
        0,
        2,
        length,
        dtype=np.uint8,
    )

    encoded = convolutional_encode(
        original,
        terminate=True,
    )

    decoded = viterbi_decode(
        encoded,
        terminated=True,
    )

    errors = bit_errors(
        original,
        decoded,
    )

    print(
        f"Length={length:3d} | "
        f"Encoded={len(encoded):3d} | "
        f"Errors={errors:2d} | "
        f"{'PASS' if errors == 0 else 'FAIL'}"
    )


print()
print("=" * 70)
print("CONVOLUTIONAL FEC TEST COMPLETE")
print("=" * 70)