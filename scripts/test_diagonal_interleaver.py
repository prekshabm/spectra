import numpy as np
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from backend.analysis.interleavers import (
    diagonal_interleave,
    diagonal_deinterleave,
)


print("=" * 70)
print("SPECTRA DIAGONAL INTERLEAVER TEST")
print("=" * 70)


rng = np.random.default_rng(42)


# ------------------------------------------------------------
# Test several matrix configurations
# ------------------------------------------------------------

for rows, cols in (
    (2, 128),
    (4, 128),
    (8, 128),
    (8, 256),
    (16, 128),
):

    original = rng.integers(
        0,
        2,
        rows * cols,
        dtype=np.uint8,
    )

    interleaved = diagonal_interleave(
        original,
        rows=rows,
    )

    recovered = diagonal_deinterleave(
        interleaved,
        rows=rows,
    )

    errors = int(
        np.count_nonzero(
            original != recovered
        )
    )

    print(
        f"Rows={rows:2d} | "
        f"Cols={cols:3d} | "
        f"Bits={len(original):4d} | "
        f"Errors={errors:2d} | "
        f"{'PASS' if errors == 0 else 'FAIL'}"
    )


# ------------------------------------------------------------
# Small deterministic example
# ------------------------------------------------------------

print()
print("-" * 70)
print("DETERMINISTIC EXAMPLE")
print("-" * 70)

original = np.array(
    [
        0, 1, 1, 0,
        1, 0, 0, 1,
        1, 1, 0, 1,
        0, 0, 1, 0,
    ],
    dtype=np.uint8,
)

interleaved = diagonal_interleave(
    original,
    rows=4,
)

recovered = diagonal_deinterleave(
    interleaved,
    rows=4,
)

print(
    "Original bits    : "
    f"{''.join(original.astype(str))}"
)

print(
    "Interleaved bits : "
    f"{''.join(interleaved.astype(str))}"
)

print(
    "Recovered bits   : "
    f"{''.join(recovered.astype(str))}"
)

errors = int(
    np.count_nonzero(
        original != recovered
    )
)

print()
print(f"Bit errors       : {errors}")
print(
    "Exact recovery   : "
    f"{'YES' if errors == 0 else 'NO'}"
)


print()
print("=" * 70)
print("DIAGONAL INTERLEAVER TEST COMPLETE")
print("=" * 70)