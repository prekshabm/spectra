from backend.analysis.interleaving_detector import (
    detect_interleaver_from_reference
)

from backend.analysis.interleavers import (
    block_interleave,
    convolutional_interleave,
    diagonal_interleave,
    pseudo_random_interleave
)


# ============================================================
# REFERENCE BITSTREAM
# ============================================================

reference_bits = [
    int(b)
    for b in (
        "001101100101110010011010110100101101001011001110"
        "110010100111010010110101001101100101101001011001"
        "101100110010110100101101001110010110100101101100"
    )
]


# ============================================================
# TEST CASES
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
# RUN TESTS
# ============================================================

for expected, interleave_function in tests:

    print("\n" + "=" * 60)
    print("EXPECTED:", expected)
    print("=" * 60)

    observed = interleave_function(
        reference_bits
    )

    result = detect_interleaver_from_reference(
        reference_bits,
        observed,
        rows=8,
        branches=4,
        delay=3,
        seed=42
    )

    print(
        "DETECTED:",
        result["type"]
    )

    print(
        "CONFIDENCE:",
        result["confidence"]
    )

    print(
        "SCORE:",
        result["score"]
    )

    print(
        "MARGIN:",
        result["margin"]
    )

    print(
        "REASON:",
        result["reason"]
    )

    print("\nCANDIDATES:")

    for candidate in result["candidates"]:

        print(
            f"  {candidate['type']}: "
            f"{candidate['match_percentage']:.2f}% "
            f"match"
        )

    print(
        "\nPASS:",
        result["type"] == expected
    )