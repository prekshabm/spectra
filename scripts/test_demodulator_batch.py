from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.signal_io.loader import load_signal
from backend.demod.demodulator import demodulate


TESTS = [
    ("BPSK", "data/bpsk"),
    ("QPSK", "data/qpsk"),
    ("2-FSK", "data/2fsk"),
    ("4-FSK", "data/4fsk"),
    ("16-QAM", "data/16qam"),
]


print("=" * 80)
print("SPECTRA DEMODULATOR BATCH TEST")
print("=" * 80)

total = 0
passed = 0


for modulation, folder in TESTS:

    print()
    print("=" * 80)
    print(modulation)
    print("=" * 80)

    files = sorted(
        Path(folder).glob("*.iq")
    )[:5]

    for path in files:

        total += 1

        try:

            signal, fs, metadata = load_signal(
                path.name,
                path.read_bytes(),
                1_000_000,
                dtype="float32",
                iq_format="IQ",
            )

            result = demodulate(
                signal,
                modulation,
                fs,
            )

            bits = result["bits_recovered"]

            if bits > 0:
                passed += 1
                status = "OK"
            else:
                status = "NO BITS"

            print(
                f"{status:8} "
                f"{path.name:40} "
                f"samples={len(signal):7} "
                f"sps={result['samples_per_symbol']:7.2f} "
                f"bits={bits:6}"
            )

        except Exception as e:

            print(
                f"ERROR    "
                f"{path.name:40} "
                f"{type(e).__name__}: {e}"
            )


print()
print("=" * 80)
print("BATCH TEST COMPLETE")
print("=" * 80)

print(
    f"Passed: {passed}/{total}"
)