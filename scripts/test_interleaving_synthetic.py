import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.analysis.interleaving import analyze_interleaving


# ============================================================
# SPECTRA — STRUCTURED INTERLEAVING TEST
# ============================================================

N_BITS = 2048


# ============================================================
# STRUCTURED REFERENCE
# ============================================================

def make_bits(n):
    """
    Create a structured but non-trivial reference bitstream.

    The repeating frame contains several different local
    patterns so that different interleavers produce
    measurable structural changes.
    """

    pattern = np.array(
        (
            [0] * 8
            + [1] * 8
            + [0, 1] * 8
            + [1, 1, 0, 0] * 4
            + [0, 0, 0, 1, 1, 1, 0, 1]
        ),
        dtype=np.uint8
    )

    repeats = int(
        np.ceil(
            n / len(pattern)
        )
    )

    return np.tile(
        pattern,
        repeats
    )[:n]


# ============================================================
# INTERLEAVERS
# ============================================================

def block_interleave(bits, block_size=64):
    """
    Block interleaver.

    Reverse the ordering of blocks rather than simply
    reversing each block.
    """

    n = len(bits)

    usable = (
        n // block_size
    ) * block_size

    if usable == 0:
        return bits.copy()

    blocks = bits[
        :usable
    ].reshape(
        -1,
        block_size
    )

    output = blocks[
        ::-1
    ].reshape(
        -1
    )

    if usable < n:
        output = np.concatenate(
            (
                output,
                bits[usable:]
            )
        )

    return output


def matrix_interleave(bits, rows=32):
    """
    Write row-wise and read column-wise.
    """

    usable = (
        len(bits)
        //
        rows
        *
        rows
    )

    x = bits[
        :usable
    ]

    cols = usable // rows

    matrix = x.reshape(
        rows,
        cols
    )

    return matrix.T.flatten()


def stride_interleave(bits, stride=17):
    """
    Deterministic stride permutation.
    """

    n = len(bits)

    order = (
        np.arange(n)
        *
        stride
    ) % n

    return bits[
        order
    ]


# ============================================================
# RESULT DISPLAY
# ============================================================

def print_result(
    name,
    original,
    test_bits
):

    result = analyze_interleaving(
        test_bits
    )

    print()
    print("-" * 80)
    print(name)
    print("-" * 80)

    print(
        f"Bits:                  "
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
        f"Transition rate:       "
        f"{result['transition_percentage']:.2f}%"
    )

    print(
        f"Mean run length:       "
        f"{result['mean_run_length']:.3f}"
    )

    print(
        f"Max run length:        "
        f"{result['max_run_length']}"
    )

    print(
        f"Strongest lag:         "
        f"{result['strongest_lag']}"
    )

    print(
        f"Correlation:           "
        f"{result['strongest_correlation']:.4f}"
    )

    print(
        f"Periodicity score:     "
        f"{result['periodicity_score']:.4f}"
    )

    print(
        f"Block size:            "
        f"{result['block_size']}"
    )

    print(
        f"Block variation:       "
        f"{result['block_balance_variation']:.4f}"
    )

    print(
        f"INTERLEAVING EVIDENCE: "
        f"{result['interleaving_evidence']}"
    )

    print(
        f"CONFIDENCE:            "
        f"{result['confidence']:.2f}%"
    )

    print(
        f"Length preserved:      "
        f"{len(original) == len(test_bits)}"
    )


# ============================================================
# TEST
# ============================================================

print("=" * 80)
print("SPECTRA — STRUCTURED INTERLEAVING TEST")
print("=" * 80)

reference = make_bits(
    N_BITS
)

print()
print("Reference pattern:")
print(
    "0000000000000000"
    "1111111111111111"
    "0000000000000000"
    "1111111111111111"
)


tests = [
    (
        "NO INTERLEAVING",
        reference
    ),

    (
        "BLOCK INTERLEAVING",
        block_interleave(
            reference,
            block_size=32
        )
    ),

    (
        "MATRIX INTERLEAVING",
        matrix_interleave(
            reference,
            rows=32
        )
    ),

    (
        "STRIDE INTERLEAVING",
        stride_interleave(
            reference,
            stride=17
        )
    ),
]


for name, bits in tests:

    print_result(
        name,
        reference,
        bits
    )


print()
print("=" * 80)
print("STRUCTURED INTERLEAVING TEST COMPLETE")
print("=" * 80)