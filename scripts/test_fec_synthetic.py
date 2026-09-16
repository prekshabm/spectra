import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.analysis.fec import analyze_fec


# ============================================================
# SPECTRA - CONTROLLED FEC TEST
# ============================================================

N_DATA = 1024

RNG = np.random.default_rng(42)


# ============================================================
# DATA
# ============================================================

def make_data(n):
    """Generate deterministic random payload bits."""

    return RNG.integers(
        0,
        2,
        size=n,
        dtype=np.uint8
    )


# ============================================================
# NO FEC
# ============================================================

def no_fec(bits):

    return bits.copy()


# ============================================================
# PARITY
# ============================================================

def parity_encode(bits, block_size=7):
    """
    Encode 7 data bits followed by 1 parity bit.
    """

    usable = (
        len(bits) // block_size
    ) * block_size

    data = bits[:usable]

    output = []

    for start in range(
        0,
        len(data),
        block_size
    ):

        block = data[
            start:
            start + block_size
        ]

        parity = int(
            np.sum(block) % 2
        )

        output.extend(
            block.tolist()
        )

        output.append(
            parity
        )

    return np.asarray(
        output,
        dtype=np.uint8
    )


# ============================================================
# REPETITION CODE
# ============================================================

def repetition_encode(bits, repeat=3):
    """
    Repeat every bit three times.
    """

    return np.repeat(
        bits,
        repeat
    ).astype(
        np.uint8
    )


# ============================================================
# HAMMING (7,4)
# ============================================================

def hamming74_encode(bits):
    """
    Hamming (7,4) encoder.

    Four data bits become seven transmitted bits.

    Positions:

        1 2 3 4 5 6 7
        p p d p d d d
    """

    usable = (
        len(bits) // 4
    ) * 4

    data = bits[
        :usable
    ].reshape(
        -1,
        4
    )

    output = []

    for row in data:

        d1 = int(row[0])
        d2 = int(row[1])
        d3 = int(row[2])
        d4 = int(row[3])

        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p4 = d2 ^ d3 ^ d4

        codeword = [
            p1,
            p2,
            d1,
            p4,
            d2,
            d3,
            d4,
        ]

        output.extend(
            codeword
        )

    return np.asarray(
        output,
        dtype=np.uint8
    )


# ============================================================
# RANDOMIZED NO-FEC CONTROL
# ============================================================

def shuffled_reference(bits):
    """
    Randomly reorder the same payload.

    This provides an additional no-FEC control.
    """

    output = bits.copy()

    RNG.shuffle(
        output
    )

    return output


# ============================================================
# RESULT DISPLAY
# ============================================================

def print_result(
    name,
    bits,
    expected
):

    result = analyze_fec(
        bits
    )

    print()
    print("-" * 80)
    print(name)
    print("-" * 80)

    print(
        f"Expected:              "
        f"{expected}"
    )

    print(
        f"Bits analyzed:         "
        f"{result['bits_analyzed']}"
    )

    print(
        f"0 percentage:          "
        f"{result['zero_percentage']:.2f}%"
    )

    print(
        f"1 percentage:          "
        f"{result['one_percentage']:.2f}%"
    )

    print(
        f"Parity group size:     "
        f"{result['parity_group_size']}"
    )

    print(
        f"Parity evidence:       "
        f"{result['parity_evidence']:.4f}"
    )

    print(
        f"Best repeat:            "
        f"{result['best_repeat']}"
    )

    print(
        f"Repetition evidence:   "
        f"{result['repetition_evidence']:.4f}"
    )

    print(
        f"Best block size:       "
        f"{result['best_block_size']}"
    )

    print(
        f"Block correlation:     "
        f"{result['block_correlation']:.4f}"
    )

    print(
        f"FEC EVIDENCE:          "
        f"{result['fec_evidence']}"
    )

    print(
        f"CONFIDENCE:            "
        f"{result['confidence']:.2f}%"
    )

    if result["evidence_reasons"]:

        print(
            "Reasons:               "
            + "; ".join(
                result["evidence_reasons"]
            )
        )

    else:

        print(
            "Reasons:               NONE"
        )


# ============================================================
# RUN TESTS
# ============================================================

print("=" * 80)
print("SPECTRA - CONTROLLED FEC TEST")
print("=" * 80)


reference = make_data(
    N_DATA
)


tests = [

    (
        "NO FEC",
        no_fec(reference),
        "NONE"
    ),

    (
        "RANDOMIZED NO FEC",
        shuffled_reference(reference),
        "NONE"
    ),

    (
        "SINGLE PARITY",
        parity_encode(reference),
        "POSSIBLE / LIKELY"
    ),

    (
        "REPETITION x3",
        repetition_encode(
            reference,
            repeat=3
        ),
        "LIKELY"
    ),

    (
        "HAMMING (7,4)",
        hamming74_encode(reference),
        "POSSIBLE / LIKELY"
    ),
]


for name, bits, expected in tests:

    print_result(
        name,
        bits,
        expected
    )


print()
print("=" * 80)
print("CONTROLLED FEC TEST COMPLETE")
print("=" * 80)