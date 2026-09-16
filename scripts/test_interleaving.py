from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.signal_io.loader import load_signal
from backend.demod.demodulator import demodulate
from backend.analysis.interleaving import analyze_interleaving


TESTS = [
    ("BPSK", "data/bpsk"),
    ("QPSK", "data/qpsk"),
    ("2-FSK", "data/2fsk"),
    ("4-FSK", "data/4fsk"),
    ("16-QAM", "data/16qam"),
]


print("=" * 80)
print("SPECTRA — DEMODULATOR → INTERLEAVING ANALYSIS TEST")
print("=" * 80)


for modulation, folder in TESTS:

    print()
    print("-" * 80)
    print(f"MODULATION: {modulation}")
    print("-" * 80)

    files = sorted(
        Path(folder).glob("*.iq")
    )

    if not files:
        print("ERROR: No IQ files found.")
        continue

    # Use the first file for each modulation.
    path = files[0]

    try:

        signal, fs, metadata = load_signal(
            path.name,
            path.read_bytes(),
            1_000_000,
            dtype="float32",
            iq_format="IQ",
        )

        demod = demodulate(
            signal,
            modulation,
            fs,
        )

        analysis = analyze_interleaving(
            demod["bitstream"]
        )

        print(f"File:                  {path.name}")
        print(f"Bits recovered:        {demod['bits_recovered']:,}")
        print(
            f"0 percentage:          "
            f"{demod['zero_percentage']:.2f}%"
        )
        print(
            f"1 percentage:          "
            f"{demod['one_percentage']:.2f}%"
        )

        print()
        print("INTERLEAVING ANALYSIS")
        print(
            f"Bits analyzed:         "
            f"{analysis['bits_analyzed']:,}"
        )
        print(
            f"Transition rate:       "
            f"{analysis['transition_percentage']:.2f}%"
        )
        print(
            f"Mean run length:      "
            f"{analysis['mean_run_length']:.3f}"
        )
        print(
            f"Max run length:       "
            f"{analysis['max_run_length']}"
        )
        print(
            f"Strongest lag:        "
            f"{analysis['strongest_lag']}"
        )
        print(
            f"Correlation:          "
            f"{analysis['strongest_correlation']:.4f}"
        )
        print(
            f"Periodicity score:    "
            f"{analysis['periodicity_score']:.4f}"
        )
        print(
            f"Block size:           "
            f"{analysis['block_size']}"
        )
        print(
            f"Block variation:      "
            f"{analysis['block_balance_variation']:.4f}"
        )

        print()
        print(
            f"INTERLEAVING EVIDENCE: "
            f"{analysis['interleaving_evidence']}"
        )
        print(
            f"CONFIDENCE:            "
            f"{analysis['confidence']:.2f}%"
        )

        print("STATUS: OK")

    except Exception as e:

        print("STATUS: ERROR")
        print(
            f"{type(e).__name__}: {e}"
        )


print()
print("=" * 80)
print("INTERLEAVING TEST COMPLETE")
print("=" * 80)