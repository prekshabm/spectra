from backend.analysis.interleaving_detector import (
    detect_interleaver_from_reference
)

from backend.analysis.interleavers import (
    block_interleave,
    convolutional_interleave,
    diagonal_interleave,
    pseudo_random_interleave
)

import numpy as np


# ============================================================
# REFERENCE BITSTREAM
# ============================================================

# ============================================================
# LONG REFERENCE BITSTREAM
# ============================================================

rng = np.random.default_rng(12345)

reference_bits = rng.integers(
    0,
    2,
    size=10000,
    dtype=np.uint8
)


# ============================================================
# INTERLEAVERS
# ============================================================

tests = [

    (
        "BLOCK",
        lambda bits:
            block_interleave(
                bits,
                rows=8
            )
    ),

    (
        "CONVOLUTIONAL",
        lambda bits:
            convolutional_interleave(
                bits,
                branches=4,
                delay=3
            )
    ),

    (
        "DIAGONAL",
        lambda bits:
            diagonal_interleave(
                bits,
                rows=8
            )
    ),

    (
        "PSEUDO-RANDOM",
        lambda bits:
            pseudo_random_interleave(
                bits,
                seed=42
            )
    )
]


# ============================================================
# BIT ERROR FUNCTION
# ============================================================

def add_bit_errors(bits, error_rate, seed=42):

    rng = np.random.default_rng(seed)

    noisy = bits.copy()

    number_of_errors = int(
        len(noisy) * error_rate
    )

    if number_of_errors > 0:

        positions = rng.choice(
            len(noisy),
            size=number_of_errors,
            replace=False
        )

        noisy[positions] ^= 1

    return noisy


# ============================================================
# ERROR LEVELS
# ============================================================

error_rates = [
    0.00,
    0.01,
    0.02,
    0.05,
    0.10
]


# ============================================================
# RUN TEST
# ============================================================

for expected_type, interleave_function in tests:

    print("\n" + "=" * 70)
    print("EXPECTED:", expected_type)
    print("=" * 70)

    observed = interleave_function(
        reference_bits
    )

    for error_rate in error_rates:

        noisy_observed = add_bit_errors(
            observed,
            error_rate
        )

        result = detect_interleaver_from_reference(
            reference_bits,
            noisy_observed,
            rows=8,
            branches=4,
            delay=3,
            seed=42
        )

        detected = result.get(
            "type"
        )

        score = result.get(
            "score",
            0.0
        )

        confidence = result.get(
            "confidence",
            0.0
        )

        passed = (
            detected == expected_type
        )

        print(
            f"Errors: {error_rate * 100:5.1f}% | "
            f"Detected: {detected:25s} | "
            f"Match: {score:7.2f}% | "
            f"Confidence: {confidence:7.2f} | "
            f"PASS: {passed}"
        )