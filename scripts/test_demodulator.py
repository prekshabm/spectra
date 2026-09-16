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


print("=" * 70)
print("SPECTRA BITSTREAM DEMODULATOR TEST")
print("=" * 70)


for modulation, folder in TESTS:

    print()
    print("-" * 70)
    print(f"Testing: {modulation}")

    files = list(
        Path(folder).glob("*.iq")
    )

    if not files:
        print("ERROR: No IQ files found.")
        continue

    path = files[0]

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

        print(f"File:              {path.name}")
        print(f"Samples:            {len(signal):,}")
        print(
            f"Samples/symbol:    "
            f"{result['samples_per_symbol']:.2f}"
        )
        print(
            f"Symbols recovered: "
            f"{result['symbols_recovered']:,}"
        )
        print(
            f"Bits recovered:    "
            f"{result['bits_recovered']:,}"
        )
        print(
            f"0 percentage:      "
            f"{result['zero_percentage']:.2f}%"
        )
        print(
            f"1 percentage:      "
            f"{result['one_percentage']:.2f}%"
        )
        print(
            "Bitstream:         "
            f"{result['bitstream'][:100]}"
        )

        print("STATUS:             OK")

    except Exception as e:

        print(
            f"STATUS:             ERROR"
        )

        print(
            f"ERROR:              {type(e).__name__}: {e}"
        )


print()
print("=" * 70)
print("DEMODULATOR TEST COMPLETE")
print("=" * 70)