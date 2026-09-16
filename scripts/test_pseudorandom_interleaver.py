import numpy as np
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from backend.analysis.interleavers import (
    pseudo_random_interleave,
    pseudo_random_deinterleave,
)


print("=" * 70)
print("SPECTRA PSEUDO-RANDOM INTERLEAVER TEST")
print("=" * 70)


rng = np.random.default_rng(42)


for length, seed in (
    (128, 1),
    (256, 42),
    (512, 1234),
    (1024, 9999),
    (2048, 2026),
):

    original = rng.integers(
        0,
        2,
        length,
        dtype=np.uint8,
    )

    interleaved = pseudo_random_interleave(
        original,
        seed=seed,
    )

    recovered = pseudo_random_deinterleave(
        interleaved,
        seed=seed,
    )

    errors = int(
        np.count_nonzero(
            original != recovered
        )
    )

    changed_positions = int(
        np.count_nonzero(
            original != interleaved
        )
    )

    print(
        f"Length={length:4d} | "
        f"Seed={seed:4d} | "
        f"Changed={changed_positions:4d} | "
        f"Errors={errors:2d} | "
        f"{'PASS' if errors == 0 else 'FAIL'}"
    )


# ------------------------------------------------------------
# Deterministic example
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

seed = 42

interleaved = pseudo_random_interleave(
    original,
    seed=seed,
)

recovered = pseudo_random_deinterleave(
    interleaved,
    seed=seed,
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
print(
    f"Bit errors       : {errors}"
)

print(
    "Exact recovery   : "
    f"{'YES' if errors == 0 else 'NO'}"
)


print()
print("=" * 70)
print("PSEUDO-RANDOM INTERLEAVER TEST COMPLETE")
print("=" * 70)