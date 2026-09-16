import numpy as np

from backend.main import detect_interleaving_type

from backend.analysis.interleavers import (
    block_interleave,
    convolutional_interleave,
    diagonal_interleave,
    pseudo_random_interleave,
)


# ------------------------------------------------------------
# STRUCTURED ORIGINAL BITSTREAM
# ------------------------------------------------------------

sync = "10101010101010101010101010101010"

header = (
    "1100110011001100"
    "1111000011110000"
)

payload = (
    "0011011001011100"
    "1001101011010010"
    "1100101001110100"
    "1011010010110011"
)

original_string = (
    sync
    + header
    + payload * 20
)

original = np.array(
    [int(b) for b in original_string],
    dtype=np.uint8
)


# ------------------------------------------------------------
# INTERLEAVER TESTS
# ------------------------------------------------------------

tests = [
    (
        "BLOCK",
        lambda x: block_interleave(x, rows=8),
    ),

    (
        "CONVOLUTIONAL",
        lambda x: convolutional_interleave(
            x,
            branches=4,
            delay=3,
        ),
    ),

    (
        "DIAGONAL",
        lambda x: diagonal_interleave(
            x,
            rows=8,
        ),
    ),

    (
        "PSEUDO-RANDOM",
        lambda x: pseudo_random_interleave(
            x,
            seed=42,
        ),
    ),
]


# ------------------------------------------------------------
# RUN DETECTION
# ------------------------------------------------------------

print(
    "Original bitstream length:",
    len(original)
)


for expected, interleave in tests:

    print("\n" + "=" * 60)
    print("EXPECTED:", expected)
    print("=" * 60)

    interleaved = interleave(original)

    result = detect_interleaving_type(
        interleaved,
        rows=8,
        branches=4,
        delay=3,
        seed=42,
    )

    print(
        "DETECTED:",
        result.get("type")
    )

    print(
        "CONFIDENCE:",
        result.get("confidence")
    )

    print(
        "SCORE:",
        result.get("score")
    )

    print(
        "MARGIN:",
        result.get("margin")
    )

    print("\nCANDIDATES:")

    for candidate in result.get(
        "candidates",
        []
    ):

        print(
            f"  {candidate['type']}: "
            f"{candidate['score']}"
        )