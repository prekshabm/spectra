import numpy as np
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from backend.analysis.interleavers import (
    convolutional_interleave,
    convolutional_deinterleave,
)


print("=" * 70)
print("SPECTRA CONVOLUTIONAL INTERLEAVER TEST")
print("=" * 70)


rng = np.random.default_rng(42)


for branches, delay in (
    (2, 1),
    (3, 2),
    (4, 3),
    (5, 2),
):

    original = rng.integers(
        0,
        2,
        256,
        dtype=np.uint8,
    )

    interleaved = convolutional_interleave(
        original,
        branches=branches,
        delay=delay,
    )

    recovered = convolutional_deinterleave(
        interleaved,
        branches=branches,
        delay=delay,
    )

    errors = int(
        np.count_nonzero(
            original != recovered
        )
    )

    print(
        f"Branches={branches:2d} | "
        f"Delay={delay:2d} | "
        f"Input={len(original):4d} | "
        f"Interleaved={len(interleaved):4d} | "
        f"Recovered={len(recovered):4d} | "
        f"Errors={errors:3d} | "
        f"{'PASS' if errors == 0 else 'FAIL'}"
    )


print()
print("=" * 70)
print("CONVOLUTIONAL INTERLEAVER TEST COMPLETE")
print("=" * 70)