import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.demod.demodulator import demodulate


# ============================================================
# SPECTRA — KNOWN-GROUND-TRUTH BER TEST
# ============================================================

FS = 1_000_000
SPS = 8
N_SYMBOLS = 1024

SNR_VALUES = [0, 1, 2, 3, 4, 5]

RNG = np.random.default_rng(42)


# ============================================================
# BIT / SYMBOL GENERATORS
# ============================================================

def bits_to_bpsk(bits):
    """
    BPSK:
        0 -> -1
        1 -> +1
    """

    symbols = (
        2 * bits.astype(np.float64)
        - 1
    )

    return symbols.astype(
        np.complex128
    )


def bits_to_qpsk(bits):
    """
    QPSK transmitter matching demodulator Gray mapping:

        00 -> +I +Q
        01 -> -I +Q
        11 -> -I -Q
        10 -> +I -Q
    """

    usable = (len(bits) // 2) * 2
    bits = bits[:usable]

    b = bits.reshape(-1, 2)

    symbols = np.zeros(
        len(b),
        dtype=np.complex128
    )

    for k, pair in enumerate(b):

        if pair[0] == 0 and pair[1] == 0:
            symbols[k] = 1 + 1j

        elif pair[0] == 0 and pair[1] == 1:
            symbols[k] = -1 + 1j

        elif pair[0] == 1 and pair[1] == 1:
            symbols[k] = -1 - 1j

        else:
            symbols[k] = 1 - 1j

    return symbols / np.sqrt(2)


def bits_to_16qam(bits):
    """
    16-QAM transmitter matching the demodulator Gray mapping.

    Per-axis Gray mapping:

        00 -> -3
        01 -> -1
        11 -> +1
        10 -> +3
    """

    usable = (len(bits) // 4) * 4
    bits = bits[:usable]

    b = bits.reshape(-1, 4)

    gray_to_level = {
        (0, 0): -3,
        (0, 1): -1,
        (1, 1):  1,
        (1, 0):  3,
    }

    symbols = []

    for row in b:

        i_bits = (
            int(row[0]),
            int(row[1])
        )

        q_bits = (
            int(row[2]),
            int(row[3])
        )

        I = gray_to_level[i_bits]
        Q = gray_to_level[q_bits]

        symbols.append(
            (I + 1j * Q) / np.sqrt(10)
        )

    return np.asarray(
        symbols,
        dtype=np.complex128
    )


def bits_to_fsk(bits, states):
    """
    FSK mapping.

    For 2-FSK:
        1 bit / symbol

    For 4-FSK:
        2 bits / symbol

    Returns both symbols and the transmitted state indices.
    """

    bits_per_symbol = int(
        np.log2(states)
    )

    usable = (
        len(bits) // bits_per_symbol
    ) * bits_per_symbol

    bits = bits[:usable]

    b = bits.reshape(
        -1,
        bits_per_symbol
    )

    state = np.zeros(
        len(b),
        dtype=np.int64
    )

    if states == 2:

        # 2-FSK:
        # 0 -> 0
        # 1 -> 1

        state = b[:, 0].astype(np.int64)

    else:

        # 4-FSK Gray mapping:
        #
        # 00 -> state 0
        # 01 -> state 1
        # 11 -> state 2
        # 10 -> state 3

        state = np.zeros(
            len(b),
            dtype=np.int64
        )

        for i in range(
            len(b)
        ):

            pair = (
                int(b[i, 0]),
                int(b[i, 1])
            )

            gray_to_state = {
                (0, 0): 0,
                (0, 1): 1,
                (1, 1): 2,
                (1, 0): 3,
            }

            state[i] = gray_to_state[pair]

    tones = (
        np.arange(states)
        -
        (states - 1) / 2
    )

    # Use a deliberately comfortable frequency separation.
    spacing = 0.04 * FS

    frequency = (
        tones[state]
        * spacing
    )

    frequency = np.repeat(
        frequency,
        SPS
    )

    phase = np.cumsum(
        2
        * np.pi
        * frequency
        / FS
    )

    signal = np.exp(
        1j * phase
    )

    return signal, state


# ============================================================
# SYMBOL UPSAMPLING
# ============================================================

def upsample(symbols):

    return np.repeat(
        symbols,
        SPS
    )


# ============================================================
# CHANNEL
# ============================================================

def add_awgn(signal, snr_db, rng):

    power = np.mean(
        np.abs(signal) ** 2
    )

    noise_power = (
        power
        /
        (10 ** (snr_db / 10))
    )

    noise = np.sqrt(
        noise_power / 2
    ) * (
        rng.normal(
            size=len(signal)
        )
        +
        1j * rng.normal(
            size=len(signal)
        )
    )

    return signal + noise


# ============================================================
# BER
# ============================================================

def calculate_ber(
    transmitted,
    recovered
):

    n = min(
        len(transmitted),
        len(recovered)
    )

    if n == 0:

        return {
            "errors": None,
            "compared": 0,
            "ber": None,
            "accuracy": None,
        }

    a = np.asarray(
        transmitted[:n],
        dtype=np.uint8
    )

    b = np.asarray(
        recovered[:n],
        dtype=np.uint8
    )

    errors = int(
        np.sum(a != b)
    )

    ber = (
        errors / n
    )

    accuracy = (
        1.0 - ber
    )

    return {
        "errors": errors,
        "compared": n,
        "ber": ber,
        "accuracy": accuracy,
    }


# ============================================================
# SINGLE MODULATION TEST
# ============================================================

def test_modulation(
    modulation,
    snr_db
):

    # --------------------------------------------------------
    # Generate deterministic ground-truth bits
    # --------------------------------------------------------

    if modulation == "BPSK":

        bits_per_symbol = 1

    elif modulation == "QPSK":

        bits_per_symbol = 2

    elif modulation == "2-FSK":

        bits_per_symbol = 1

    elif modulation == "4-FSK":

        bits_per_symbol = 2

    elif modulation == "16-QAM":

        bits_per_symbol = 4

    else:

        raise ValueError(
            f"Unsupported modulation: {modulation}"
        )

    tx_bits = RNG.integers(
        0,
        2,
        size=N_SYMBOLS * bits_per_symbol,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Modulate
    # --------------------------------------------------------

    if modulation == "BPSK":

        symbols = bits_to_bpsk(
            tx_bits
        )

        signal = upsample(
            symbols
        )

    elif modulation == "QPSK":

        symbols = bits_to_qpsk(
            tx_bits
        )

        signal = upsample(
            symbols
        )

    elif modulation == "16-QAM":

        symbols = bits_to_16qam(
            tx_bits
        )

        signal = upsample(
            symbols
        )

    elif modulation == "2-FSK":

        signal, _ = bits_to_fsk(
            tx_bits,
            2
        )

    elif modulation == "4-FSK":

        signal, _ = bits_to_fsk(
            tx_bits,
            4
        )

    # --------------------------------------------------------
    # AWGN
    # --------------------------------------------------------

    rx_signal = add_awgn(
        signal,
        snr_db,
        RNG
    )

    # --------------------------------------------------------
    # Actual SPECTRA demodulator
    # --------------------------------------------------------

    result = demodulate(
        rx_signal,
        modulation,
        FS
    )

    recovered = np.asarray(
        [int(b) for b in result["bitstream"]],
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    ber = calculate_ber(
        tx_bits,
        recovered
    )

    return {
        "transmitted": len(tx_bits),
        "recovered": len(recovered),
        **ber,
    }


# ============================================================
# MAIN
# ============================================================

MODULATIONS = [
    "BPSK",
    "QPSK",
    "2-FSK",
    "4-FSK",
    "16-QAM",
]


print("=" * 90)
print("SPECTRA — DEMODULATOR BER / GROUND-TRUTH TEST")
print("=" * 90)

print()
print(
    f"Sample rate:       {FS:,} Hz"
)

print(
    f"Samples/symbol:    {SPS}"
)

print(
    f"Symbols/test:      {N_SYMBOLS:,}"
)

print(
    f"SNR range:         {SNR_VALUES[0]}–{SNR_VALUES[-1]} dB"
)


for modulation in MODULATIONS:

    print()
    print("=" * 90)
    print(modulation)
    print("=" * 90)

    for snr in SNR_VALUES:

        try:

            result = test_modulation(
                modulation,
                snr
            )

            if result["ber"] is None:

                print(
                    f"SNR {snr:2d} dB | "
                    f"TX={result['transmitted']:5d} | "
                    f"RX={result['recovered']:5d} | "
                    f"NO COMPARISON"
                )

                continue

            print(
                f"SNR {snr:2d} dB | "
                f"TX={result['transmitted']:5d} | "
                f"RX={result['recovered']:5d} | "
                f"errors={result['errors']:5d} | "
                f"BER={result['ber']:.5f} | "
                f"accuracy={result['accuracy'] * 100:6.2f}%"
            )

        except Exception as e:

            print(
                f"SNR {snr:2d} dB | "
                f"ERROR: "
                f"{type(e).__name__}: {e}"
            )


print()
print("=" * 90)
print("GROUND-TRUTH BER TEST COMPLETE")
print("=" * 90)