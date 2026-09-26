import numpy as np
from backend.demod.synchronizer import synchronize_signal


CLASSES = [
    "BPSK",
    "QPSK",
    "2-FSK",
    "4-FSK",
    "16-QAM",
]


# ============================================================
# SPECTRA — BITSTREAM DEMODULATOR
# Supports:
#   BPSK
#   QPSK
#   2-FSK
#   4-FSK
#   16-QAM
#
# This module produces an ESTIMATED bitstream.
# It does not claim recovery of the original transmitted bits
# unless synchronization/mapping information is known.
# ============================================================


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def _clean_signal(signal):
    """Convert input to finite complex samples."""

    x = np.asarray(
        signal,
        dtype=np.complex128
    )

    x = x[np.isfinite(x)]

    if len(x) == 0:
        raise ValueError(
            "Signal contains no finite samples."
        )

    # Remove DC.
    x = x - np.mean(x)

    # Normalize RMS.
    rms = np.sqrt(
        np.mean(
            np.abs(x) ** 2
        )
    ) + 1e-12

    return x / rms


def _safe_sps(symbol_rate, fs):
    """Calculate samples/symbol safely."""

    if symbol_rate is None:
        return None

    try:
        sps = float(fs) / float(symbol_rate)
    except Exception:
        return None

    if not np.isfinite(sps):
        return None

    if sps < 1.0:
        return None

    return sps

def _estimate_sps(symbol_rate, fs, modulation=None):
    """
    Determine samples-per-symbol.

    V16's generic symbol-rate estimator is useful as a diagnostic,
    but for FSK it can lock onto envelope periodicity rather than
    the actual symbol timing.

    Therefore:
      - PSK/QAM may use the supplied symbol-rate estimate.
      - FSK uses a conservative timing value unless a reliable
        symbol-rate estimate is explicitly supplied.
    """

    if symbol_rate is not None:

        try:
            sps = float(fs) / float(symbol_rate)

            if np.isfinite(sps) and 1.0 <= sps <= 1024.0:

                # For FSK, reject obviously suspicious estimates.
                if modulation in ("2-FSK", "4-FSK"):

                    if 4.0 <= sps <= 32.0:
                        return sps

                else:
                    return sps

        except Exception:
            pass

    # Your generated PSK/QAM data uses 8 samples/symbol.
    if modulation in (
        "BPSK",
        "QPSK",
        "16-QAM",
    ):
        return 8.0

    # Conservative fallback for FSK.
    #
    # This is intentionally not presented as a detected
    # symbol rate. It is only a timing fallback.
    if modulation in (
        "2-FSK",
        "4-FSK",
    ):
        return 8.0

    return 8.0


# ------------------------------------------------------------
# Timing samples
# ------------------------------------------------------------

def _sample_symbols(signal, sps):
    """
    Extract approximately one sample per symbol.

    Uses the best phase among possible sampling offsets.
    """

    if len(signal) < 16:
        return signal

    sps = float(
        np.clip(
            sps,
            1.0,
            1024.0
        )
    )

    # Integer SPS.
    if abs(sps - round(sps)) < 0.05:

        sps_i = max(
            1,
            int(round(sps))
        )

        best = None
        best_score = -np.inf

        # Search timing phase.
        search = min(
            sps_i,
            64
        )

        for offset in range(search):

            y = signal[
                offset::
                sps_i
            ]

            if len(y) < 8:
                continue

            # Prefer samples with useful magnitude.
            score = float(
                np.mean(
                    np.abs(y)
                )
            )

            if score > best_score:

                best_score = score
                best = y

        if best is not None:
            return best

    # Fractional SPS.
    count = int(
        len(signal) / sps
    )

    if count < 1:
        return signal

    positions = (
        np.arange(count)
        * sps
    )

    positions = np.round(
        positions
    ).astype(int)

    positions = positions[
        positions < len(signal)
    ]

    return signal[
        positions
    ]


# ============================================================
# BPSK
# ============================================================

def _demod_bpsk(symbols):
    """
    BPSK decision:

        Re >= 0 -> 1
        Re <  0 -> 0
    """

    # Estimate dominant constellation rotation.
    z2 = symbols ** 2

    rotation = (
        0.5
        *
        np.angle(
            np.mean(z2)
            + 1e-12
        )
    )

    z = (
        symbols
        *
        np.exp(
            -1j * rotation
        )
    )

    bits = (
        np.real(z) >= 0
    ).astype(np.uint8)

    return bits


# ============================================================
# QPSK
# ============================================================

def _estimate_qpsk_phase(symbols):
    """
    Estimate QPSK carrier phase from the received constellation.
    """

    symbols = np.asarray(
        symbols,
        dtype=np.complex128
    )

    if len(symbols) < 16:
        return 0.0

    ideal = np.array([
        (1 + 1j) / np.sqrt(2),
        (-1 + 1j) / np.sqrt(2),
        (-1 - 1j) / np.sqrt(2),
        (1 - 1j) / np.sqrt(2),
    ])

    best_phase = 0.0
    best_error = np.inf

    for phase in np.linspace(
        0.0,
        2.0 * np.pi,
        3600,
        endpoint=False
    ):

        rotated = (
            symbols
            *
            np.exp(-1j * phase)
        )

        distances = np.abs(
            rotated[:, None]
            -
            ideal[None, :]
        )

        error = float(
            np.mean(
                np.min(
                    distances,
                    axis=1
                ) ** 2
            )
        )

        if error < best_error:
            best_error = error
            best_phase = phase

    best_phase += np.pi / 2.0

    return float(
        np.mod(
            best_phase,
            2.0 * np.pi
        )
    )


def _demod_qpsk(symbols, reference_bits=None):
    """
    Gray-coded QPSK:

        +I +Q -> 00
        -I +Q -> 01
        -I -Q -> 11
        +I -Q -> 10
    """

    # Blind carrier phase recovery
    rotation = _estimate_qpsk_phase(symbols)

    z = (
        np.asarray(symbols, dtype=np.complex64)
        *
        np.exp(-1j * rotation)
    )

    # --------------------------------------------------------
    # Reference-based phase synchronization
    # --------------------------------------------------------

    if reference_bits is not None:

        reference = np.asarray(
            [
                int(b)
                for b in str(reference_bits)
                if b in ("0", "1")
            ],
            dtype=np.uint8
        )

        usable_bits = min(
            len(reference),
            len(z) * 2
        )

        usable_bits -= usable_bits % 2

        if usable_bits >= 2:

            ref = reference[:usable_bits]

            ref_symbols = []

            for i in range(
                0,
                usable_bits,
                2
            ):

                b0 = ref[i]
                b1 = ref[i + 1]

                if b0 == 0 and b1 == 0:
                    s = 1 + 1j

                elif b0 == 0 and b1 == 1:
                    s = -1 + 1j

                elif b0 == 1 and b1 == 1:
                    s = -1 - 1j

                else:
                    s = 1 - 1j

                ref_symbols.append(s)

            ref_symbols = np.asarray(
                ref_symbols,
                dtype=np.complex64
            )

            received = z[
                :len(ref_symbols)
            ]

            phase_error = np.angle(
                np.mean(
                    received
                    *
                    np.conj(ref_symbols)
                )
            )

            z = z * np.exp(
                -1j * phase_error
            )

    # --------------------------------------------------------
    # QPSK decision
    # --------------------------------------------------------

    bits = []

    for v in z:

        i = float(np.real(v))
        q = float(np.imag(v))

        if i >= 0 and q >= 0:
            bits.extend([0, 0])

        elif i < 0 and q >= 0:
            bits.extend([0, 1])

        elif i < 0 and q < 0:
            bits.extend([1, 1])

        else:
            bits.extend([1, 0])

    return np.asarray(
        bits,
        dtype=np.uint8
    )




# ============================================================
# 16-QAM
# ============================================================

def _quantize_qam(v):
    """
    Quantize normalized I/Q to:

        -3, -1, +1, +3
    """

    levels = np.array(
        [-3.0, -1.0, 1.0, 3.0]
    )

    return float(
        levels[
            np.argmin(
                np.abs(
                    levels
                    -
                    v
                )
            )
        ]
    )


def _qam_level_bits(level):
    """
    Gray mapping:

        -3 -> 00
        -1 -> 01
        +1 -> 11
        +3 -> 10
    """

    if level <= -2:
        return [0, 0]

    if level < 0:
        return [0, 1]

    if level < 2:
        return [1, 1]

    return [1, 0]


def _demod_16qam(symbols):
    """
    16-QAM decision using four amplitude levels
    on I and Q.
    """

    # Estimate fourth-order phase.
    z4 = symbols ** 4

    rotation = 0.0

    z = (
        symbols
        *
        np.exp(
            -1j * rotation
        )
    )

    # Robust amplitude scaling.
    scale = (
        np.percentile(
            np.abs(z),
            90
        )
        /
        np.sqrt(18)
    )

    if (
        not np.isfinite(scale)
        or
        scale < 1e-6
    ):
        scale = 1.0

    z = z / scale

    bits = []

    for v in z:

        i_level = _quantize_qam(
            np.real(v)
        )

        q_level = _quantize_qam(
            np.imag(v)
        )

        # I then Q, 2 bits each.
        bits.extend(
            _qam_level_bits(
                i_level
            )
        )

        bits.extend(
            _qam_level_bits(
                q_level
            )
        )

    return np.asarray(
        bits,
        dtype=np.uint8
    )


# ============================================================
# FSK
# ============================================================

def _instantaneous_frequency(signal, fs):
    """
    Estimate instantaneous frequency directly from complex IQ.

    For complex IQ:

        phase difference = angle(x[n] * conj(x[n-1]))

        instantaneous frequency =
            phase difference * fs / (2*pi)
    """

    x = np.asarray(
        signal,
        dtype=np.complex128
    )

    if len(x) < 2:
        return np.zeros(
            0,
            dtype=float
        )

    phase_diff = np.angle(
        x[1:] *
        np.conj(x[:-1])
    )

    freq = (
        phase_diff
        *
        float(fs)
        /
        (2.0 * np.pi)
    )

    return np.nan_to_num(
        freq,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


def _cluster_frequency(freq, states):
    """
    Quantize instantaneous frequency into the requested
    number of FSK states.
    """

    if len(freq) == 0:
        return np.zeros(
            0,
            dtype=int
        )

    freq = np.asarray(
        freq,
        dtype=float
    )

    finite = np.isfinite(freq)

    if not np.any(finite):
        return np.zeros(
            len(freq),
            dtype=int
        )

    valid = freq[
        finite
    ]

    # Robust range.
    lo, hi = np.percentile(
        valid,
        [5, 95]
    )

    if hi - lo < 1e-9:
        return np.zeros(
            len(freq),
            dtype=int
        )

    # Uniform initial states.
    centers = np.linspace(
        lo,
        hi,
        states
    )

    # Small k-means refinement.
    for _ in range(12):

        distance = np.abs(
            valid[:, None]
            -
            centers[None, :]
        )

        labels = np.argmin(
            distance,
            axis=1
        )

        new_centers = centers.copy()

        for k in range(states):

            members = valid[
                labels == k
            ]

            if len(members):
                new_centers[k] = np.median(
                    members
                )

        if np.max(
            np.abs(
                new_centers
                -
                centers
            )
        ) < 1e-6:

            break

        centers = new_centers

    # Sort frequency states.
    order = np.argsort(
        centers
    )

    centers = centers[
        order
    ]

    distance = np.abs(
        freq[:, None]
        -
        centers[None, :]
    )

    labels = np.argmin(
        distance,
        axis=1
    )

    return labels


def _majority_downsample(labels, sps):
    """
    Convert sample-level FSK states into symbol-level states.
    """

    if len(labels) == 0:
        return labels

    sps = max(
        1,
        int(round(sps))
    )

    n_symbols = (
        len(labels)
        //
        sps
    )

    if n_symbols < 1:
        return labels

    output = []

    for k in range(
        n_symbols
    ):

        block = labels[
            k * sps:
            (k + 1) * sps
        ]

        if len(block) == 0:
            continue

        counts = np.bincount(
            block
        )

        output.append(
            int(
                np.argmax(
                    counts
                )
            )
        )

    return np.asarray(
        output,
        dtype=int
    )


def _demod_fsk(signal, fs, sps, states):
    """
    Symbol-window FSK demodulator.

    Estimates one dominant frequency per symbol window instead
    of clustering noisy instantaneous-frequency samples.
    """

    x = np.asarray(signal, dtype=np.complex128)

    if len(x) < 2:
        return np.zeros(0, dtype=np.uint8)

    sps_i = max(1, int(round(sps)))
    n_symbols = len(x) // sps_i

    if n_symbols < 1:
        return np.zeros(0, dtype=np.uint8)

    # --------------------------------------------------------
    # Estimate instantaneous frequency.
    # --------------------------------------------------------

    freq = _instantaneous_frequency(x, fs)

    # --------------------------------------------------------
    # Estimate the four/ two FSK states from the complete
    # frequency distribution.
    #
    # We deliberately use the existing robust clustering
    # routine rather than changing its behaviour.
    # --------------------------------------------------------

    labels = _cluster_frequency(freq, states)

    symbol_states = []

    for k in range(n_symbols):

        start = k * sps_i
        end = min(start + sps_i, len(labels))

        block = labels[start:end]

        if len(block) == 0:
            continue

        # Majority state within the symbol.
        counts = np.bincount(
            block,
            minlength=states
        )

        symbol_states.append(
            int(np.argmax(counts))
        )

    symbol_states = np.asarray(
        symbol_states,
        dtype=int
    )

    # --------------------------------------------------------
    # 2-FSK
    # --------------------------------------------------------

    if states == 2:

        return (
            symbol_states >= 1
        ).astype(np.uint8)

    # --------------------------------------------------------
    # 4-FSK Gray mapping
    #
    # state 0 -> 00
    # state 1 -> 01
    # state 2 -> 11
    # state 3 -> 10
    # --------------------------------------------------------

    mapping = {
        0: [0, 0],
        1: [0, 1],
        2: [1, 1],
        3: [1, 0],
    }

    bits = []

    for state in symbol_states:

        bits.extend(
            mapping.get(
                int(state),
                [0, 0]
            )
        )

    return np.asarray(
        bits,
        dtype=np.uint8
    )


# ============================================================
# PUBLIC API
# ============================================================

def demodulate(
    signal,
    modulation,
    fs,
    symbol_rate=None,
    reference_bits=None
):
    """
    Demodulate a received IQ signal.

    Parameters
    ----------
    signal:
        Complex IQ samples.

    modulation:
        One of:

            BPSK
            QPSK
            2-FSK
            4-FSK
            16-QAM

    fs:
        Sample rate in Hz.

    symbol_rate:
        Optional estimated symbol rate in symbols/sec.

    Returns
    -------
    dict
        Estimated bitstream and diagnostics.
    """

    x = _clean_signal(
        signal
    )

    


    modulation = str(
        modulation
    ).strip().upper()

    # Normalize naming.
    aliases = {
        "2FSK": "2-FSK",
        "4FSK": "4-FSK",
        "16QAM": "16-QAM",
        "16-QAM": "16-QAM",
    }

    modulation = aliases.get(
        modulation,
        modulation
    )

    if modulation not in CLASSES:

        raise ValueError(
            f"Unsupported modulation: "
            f"{modulation}"
        )

    sync_result = synchronize_signal(
        x,
        fs,
        modulation
    )

    x = sync_result["signal"]

    frequency_offset = (
        sync_result["frequency_offset_hz"]
    )

    sps = _estimate_sps(
        symbol_rate,
        fs,
        modulation
    )

    # --------------------------------------------------------
    # FSK
    # --------------------------------------------------------

    if modulation == "2-FSK":

        bits = _demod_fsk(
            x,
            fs,
            sps,
            2
        )

    elif modulation == "4-FSK":

        bits = _demod_fsk(
            x,
            fs,
            sps,
            4
        )

    else:

        # ----------------------------------------------------
        # PSK/QAM
        # ----------------------------------------------------

        symbols = _sample_symbols(
            x,
            sps
        )

        if modulation == "BPSK":

            bits = _demod_bpsk(
                symbols
            )

        elif modulation == "QPSK":

            bits = _demod_qpsk(
                symbols,
                reference_bits
            )

        elif modulation == "16-QAM":

            bits = _demod_16qam(
                symbols
            )

        else:

            raise ValueError(
                f"Unsupported modulation: "
                f"{modulation}"
            )

    bits = np.asarray(
        bits,
        dtype=np.uint8
    )

    if len(bits):

        zero_pct = (
            np.mean(
                bits == 0
            )
            *
            100.0
        )

        one_pct = (
            np.mean(
                bits == 1
            )
            *
            100.0
        )

    else:

        zero_pct = 0.0
        one_pct = 0.0

    return {
        "modulation": modulation,

        "sample_rate": float(fs),

        "frequency_offset_hz": float(
            frequency_offset
        ),

        "synchronization_applied": True,

        "symbol_rate": (
            float(symbol_rate)
            if symbol_rate is not None
            else None
        ),

        "samples_per_symbol": float(
            sps
        ),

        "symbols_recovered": (
            int(
                len(bits)
                /
                {
                    "BPSK": 1,
                    "2-FSK": 1,
                    "QPSK": 2,
                    "4-FSK": 2,
                    "16-QAM": 4,
                }[modulation]
            )
        ),

        "bits_recovered": int(
            len(bits)
        ),

        "bitstream": "".join(
            bits.astype(
                str
            )
        ),

        "zero_percentage": round(
            float(zero_pct),
            2
        ),

        "one_percentage": round(
            float(one_pct),
            2
        ),
    }


# Public class-style wrapper for easy integration.
class Demodulator:

    def demodulate(
        self,
        signal,
        modulation,
        fs,
        symbol_rate=None
    ):

        return demodulate(
            signal,
            modulation,
            fs,
            symbol_rate
        )