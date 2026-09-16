from backend.main import detect_interleaving_type
from backend.analysis.interleavers import (
    block_interleave,
    convolutional_interleave,
    diagonal_interleave,
    pseudo_random_interleave
)


# ------------------------------------------------------------
# TEST BITSTREAM
# ------------------------------------------------------------

original_bits = [
    int(b)
    for b in (
        "001101100101110010011010110100101101001011001110"
        "110010100111010010110101001101100101101001011001"
        "101100110010110100101101001110010110100101101100"
    )
]


# ------------------------------------------------------------
# KNOWN INTERLEAVERS
# ------------------------------------------------------------

tests = [
    (
        "BLOCK",
        lambda bits: block_interleave(
            bits,
            rows=8
        )
    ),

    (
        "CONVOLUTIONAL",
        lambda bits: convolutional_interleave(
            bits,
            branches=4,
            delay=3
        )
    ),

    (
        "DIAGONAL",
        lambda bits: diagonal_interleave(
            bits,
            rows=8
        )
    ),

    (
        "PSEUDO-RANDOM",
        lambda bits: pseudo_random_interleave(
            bits,
            seed=42
        )
    )
]


# ------------------------------------------------------------
# RUN VALIDATION
# ------------------------------------------------------------

for expected_type, interleave_function in tests:

    print("\n" + "=" * 60)
    print("EXPECTED:", expected_type)
    print("=" * 60)

    try:

        # Apply known interleaver
        interleaved_bits = interleave_function(
            original_bits
        )

        print(
            "Input bits:",
            len(original_bits)
        )

        print(
            "Interleaved bits:",
            len(interleaved_bits)
        )

        # Run automatic detector
        detection = detect_interleaving_type(
            interleaved_bits,
            rows=8,
            branches=4,
            delay=3,
            seed=42
        )

        print(
            "DETECTED:",
            detection.get("type")
        )

        print(
            "CONFIDENCE:",
            detection.get("confidence")
        )

        print(
            "SCORE:",
            detection.get("score")
        )

        print(
            "MARGIN:",
            detection.get("margin")
        )

        print("\nCANDIDATES:")

        for candidate in detection.get(
            "candidates",
            []
        ):

            print(
                f"  {candidate['type']}: "
                f"{candidate['score']}"
            )

    except Exception as error:

        print(
            "ERROR:",
            type(error).__name__,
            error
        )