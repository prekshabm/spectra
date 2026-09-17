import numpy as np

from backend.analysis.interleavers import (
    block_interleave,
    block_deinterleave,
    convolutional_interleave,
    convolutional_deinterleave,
    diagonal_interleave,
    diagonal_deinterleave,
    pseudo_random_interleave,
    pseudo_random_deinterleave,
)


# ------------------------------------------------------------
# ORIGINAL TEST DATA
# ------------------------------------------------------------

original = np.array(
    [
        int(b)
        for b in (
            "001101100101110010011010110100101101001011001110"
            "110010100111010010110101001101100101101001011001"
            "101100110010110100101101001110010110100101101100"
        )
    ],
    dtype=np.uint8
)


# ------------------------------------------------------------
# ROUND-TRIP TEST
# ------------------------------------------------------------

tests = [

    (
        "BLOCK",
        lambda x: block_interleave(x, rows=8),
        lambda x: block_deinterleave(x, rows=8),
    ),

    (
        "CONVOLUTIONAL",
        lambda x: convolutional_interleave(
            x,
            branches=4,
            delay=3
        ),
        lambda x: convolutional_deinterleave(
            x,
            branches=4,
            delay=3
        ),
    ),

    (
        "DIAGONAL",
        lambda x: diagonal_interleave(x, rows=8),
        lambda x: diagonal_deinterleave(x, rows=8),
    ),

    (
        "PSEUDO-RANDOM",
        lambda x: pseudo_random_interleave(
            x,
            seed=42
        ),
        lambda x: pseudo_random_deinterleave(
            x,
            seed=42
        ),
    ),
]


for name, interleave, deinterleave in tests:

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    try:

        interleaved = interleave(original)

        recovered = deinterleave(interleaved)

        matches = np.sum(
            recovered == original
        )

        accuracy = (
            matches / len(original)
        ) * 100

        print(
            "Original length:",
            len(original)
        )

        print(
            "Interleaved length:",
            len(interleaved)
        )

        print(
            "Recovered length:",
            len(recovered)
        )

        print(
            "Matching bits:",
            matches,
            "/",
            len(original)
        )

        print(
            "Round-trip accuracy:",
            f"{accuracy:.2f}%"
        )

        print(
            "PASS:",
            np.array_equal(
                original,
                recovered
            )
        )

    except Exception as error:

        print(
            "ERROR:",
            type(error).__name__,
            error
        )