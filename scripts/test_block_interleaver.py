from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.analysis.interleavers import (
    block_interleave,
    block_deinterleave,
)


print("=" * 70)
print("SPECTRA BLOCK INTERLEAVER TEST")
print("=" * 70)


# ------------------------------------------------------------
# Known bitstream
# ------------------------------------------------------------

original = np.array(
    [
        0, 1, 1, 0,
        1, 0, 0, 1,
        1, 1, 0, 1,
        0, 0, 1, 0,
    ],
    dtype=np.uint8,
)


rows = 4


print()
print(f"Original bits       : {''.join(original.astype(str))}")
print(f"Block rows          : {rows}")


# ------------------------------------------------------------
# INTERLEAVE
# ------------------------------------------------------------

interleaved = block_interleave(
    original,
    rows=rows,
)

print()
print(
    f"Interleaved bits    : "
    f"{''.join(interleaved.astype(str))}"
)


# ------------------------------------------------------------
# DE-INTERLEAVE
# ------------------------------------------------------------

recovered = block_deinterleave(
    interleaved,
    rows=rows,
)

print(
    f"Recovered bits      : "
    f"{''.join(recovered.astype(str))}"
)


# ------------------------------------------------------------
# VERIFY
# ------------------------------------------------------------

errors = int(
    np.count_nonzero(
        original != recovered
    )
)

print()
print(
    f"Bit errors          : {errors}"
)

print(
    f"Exact recovery      : "
    f"{'YES' if errors == 0 else 'NO'}"
)


# ------------------------------------------------------------
# MULTIPLE BLOCK SIZES
# ------------------------------------------------------------

print()
print("-" * 70)
print("MULTIPLE BLOCK-SIZE TEST")
print("-" * 70)

rng = np.random.default_rng(42)

for rows in (2, 4, 8, 16):

    n = rows * 128

    original = rng.integers(
        0,
        2,
        n,
        dtype=np.uint8,
    )

    interleaved = block_interleave(
        original,
        rows=rows,
    )

    recovered = block_deinterleave(
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
        f"Bits={n:4d} | "
        f"Errors={errors:2d} | "
        f"{'PASS' if errors == 0 else 'FAIL'}"
    )


print()
print("=" * 70)
print("BLOCK INTERLEAVER TEST COMPLETE")
print("=" * 70)