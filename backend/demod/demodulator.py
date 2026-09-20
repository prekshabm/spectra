import numpy as np



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


def _estimate_frequency_offset(
    signal,
    fs,
    modulation
):
    """
    Estimate common carrier/frequency offset from complex IQ.

    Most adjacent samples belong to the same symbol when the
    signal is oversampled. Their phase increment therefore
    contains mainly the carrier frequency offset, while symbol
    transitions form smaller secondary clusters.

    A histogram of phase increments is used to find the dominant
    cluster and obtain a robust CFO estimate.
    """

    if modulation not in (
        "BPSK",
        "QPSK",
        "16-QAM",
    ):
        return 0.0

    if (
        len(signal) < 32
        or
        not np.isfinite(fs)
        or
        fs <= 0
    ):
        return 0.0

    x = np.asarray(
        signal,
        dtype=np.complex128
    )

    phase_steps = np.angle(
        x[1:]
        *
        np.conj(
            x[:-1]
        )
    )

    magnitudes = (
        np.abs(x[1:])
        *
        np.abs(x[:-1])
    )

    valid = (
        np.isfinite(phase_steps)
        &
        np.isfinite(magnitudes)
        &
        (magnitudes > 1e-6)
    )

    phase_steps = phase_steps[valid]
    magnitudes = magnitudes[valid]

    if len(phase_steps) < 32:
        return 0.0

    # Find the dominant phase-increment cluster.
    bins = 512

    histogram, edges = np.histogram(
        phase_steps,
        bins=bins,
        range=(-np.pi, np.pi),
        weights=magnitudes
    )

    peak = int(
        np.argmax(histogram)
    )

    center = (
        edges[peak]
        +
        edges[peak + 1]
    ) / 2.0

    bin_width = (
        edges[1]
        -
        edges[0]
    )

    # Keep samples close to the dominant cluster.
    # This rejects symbol-transition phase jumps.
    wrapped_distance = np.angle(
        np.exp(
            1j
            *
            (
                phase_steps
                -
                center
            )
        )
    )

    keep = (
        np.abs(
            wrapped_distance
        )
        <=
        2.5 * bin_width
    )

    selected = phase_steps[keep]
    selected_weights = magnitudes[keep]

    if len(selected) < 16:
        selected = phase_steps
        selected_weights = magnitudes

    # Circular weighted mean gives a precise phase increment.
    vector = np.sum(
        selected_weights
        *
        np.exp(
            1j * selected
        )
    )

    if abs(vector) < 1e-12:
        return 0.0

    phase_increment = float(
        np.angle(vector)
    )

    frequency_offset = (
        phase_increment
        *
        float(fs)
        /
        (2.0 * np.pi)
    )

    # Principal Nyquist interval.
    half_fs = (
        float(fs)
        / 2.0
    )

    frequency_offset = (
        (
            frequency_offset
            +
            half_fs
        )
        %
        float(fs)
    ) - half_fs

    return float(
        frequency_offset
    )



def _correct_frequency_offset(
    signal,
    fs,
    frequency_offset
):
    """
    Remove a constant frequency offset from complex IQ.
    """

    if (
        len(signal) == 0
        or
        not np.isfinite(frequency_offset)
        or
        not np.isfinite(fs)
        or
        fs <= 0
    ):
        return signal

    n = np.arange(
        len(signal),
        dtype=float
    )

    correction = np.exp(
        -1j
        *
        2.0
        *
        np.pi
        *
        frequency_offset
        *
        n
        /
        float(fs)
    )

    return (
        signal
        *
        correction
    )


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


# ------------------------------------------------------------
# Symbol timing
# ------------------------------------------------------------

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

def _demod_qpsk(symbols):
    """
    Gray-coded QPSK:

        +I +Q -> 00
        -I +Q -> 01
        -I -Q -> 11
        +I -Q -> 10
    """

    # Fourth-power carrier/phase estimate.
    z4 = symbols ** 4

    rotation = 0.0

    z = (
        symbols
        *
        np.exp(
            -1j * rotation
        )
    )

    bits = []

    for v in z:

        i = float(
            np.real(v)
        )

        q = float(
            np.imag(v)
        )

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


def _demod_16qam(
    symbols,
    reference_bits=None
):
    """
    16-QAM demodulation with blind phase estimation.

    Without a reference, square 16-QAM has an inherent
    90-degree rotational ambiguity.

    When reference_bits are supplied, the four possible
    90-degree rotations are tested and the best matching
    hypothesis is selected.
    """

    # --------------------------------------------------------
    # Blind phase estimation
    # --------------------------------------------------------

    z4 = symbols ** 4
    moment = np.mean(z4)

    if (
        np.isfinite(np.real(moment))
        and
        np.isfinite(np.imag(moment))
        and
        abs(moment) > 1e-8
    ):
        rotation = (
            np.angle(moment)
            - np.pi
        ) / 4.0
    else:
        rotation = 0.0

    while rotation <= -np.pi / 4.0:
        rotation += np.pi / 2.0

    while rotation > np.pi / 4.0:
        rotation -= np.pi / 2.0

    # Apply blind phase correction.
    z = (
        symbols
        *
        np.exp(
            -1j * rotation
        )
    )

    # --------------------------------------------------------
    # Robust amplitude scaling
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Demodulate one phase hypothesis
    # --------------------------------------------------------

    def demod_with_extra_rotation(
        extra_rotation
    ):

        zz = (
            z
            *
            np.exp(
                -1j * extra_rotation
            )
        )

        bits = []

        for v in zz:

            i_level = _quantize_qam(
                np.real(v)
            )

            q_level = _quantize_qam(
                np.imag(v)
            )

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

    # Blind result.
    bits = demod_with_extra_rotation(
        0.0
    )

    phase_info = {
        "phase_rotation_deg": round(
            float(
                np.rad2deg(rotation)
            ),
            4
        ),
        "phase_ambiguity_deg": 90.0,
        "phase_ambiguous": True,
        "phase_method": (
            "16-QAM fourth-order moment"
        ),
        "phase_reference_used": False,
        "phase_reference_score": None,
        "phase_ambiguity_resolved": False,
    }

    # --------------------------------------------------------
    # Convert optional reference bits
    # --------------------------------------------------------

    reference = None

    if reference_bits is not None:

        if isinstance(
            reference_bits,
            str
        ):

            clean = "".join(
                ch
                for ch in reference_bits
                if ch in "01"
            )

            if clean:

                reference = np.fromiter(
                    (
                        int(ch)
                        for ch in clean
                    ),
                    dtype=np.uint8
                )

        else:

            reference = np.asarray(
                reference_bits,
                dtype=np.uint8
            ).reshape(-1)

            reference = reference[
                (reference == 0)
                |
                (reference == 1)
            ]

    # --------------------------------------------------------
    # Resolve 90-degree ambiguity using reference
    # --------------------------------------------------------

    if (
        reference is not None
        and
        reference.size > 0
        and
        bits.size > 0
    ):

        candidates = []

        for k in range(4):

            extra_rotation = (
                k * np.pi / 2.0
            )

            candidate_bits = (
                demod_with_extra_rotation(
                    extra_rotation
                )
            )

            n = min(
                reference.size,
                candidate_bits.size
            )

            if n <= 0:
                continue

            score = float(
                np.mean(
                    candidate_bits[:n]
                    ==
                    reference[:n]
                )
            )

            candidates.append(
                (
                    score,
                    k,
                    candidate_bits
                )
            )

        if candidates:

            candidates.sort(
                key=lambda item: item[0],
                reverse=True
            )

            best_score, best_k, best_bits = (
                candidates[0]
            )

            second_score = (
                candidates[1][0]
                if len(candidates) > 1
                else 0.0
            )

            phase_info[
                "phase_reference_used"
            ] = True

            phase_info[
                "phase_reference_score"
            ] = round(
                best_score,
                4
            )

            # Require both a strong match and a clear
            # separation from the next hypothesis.
            if (
                best_score >= 0.80
                and
                (best_score - second_score) >= 0.15
            ):

                bits = best_bits

                total_rotation = (
                    rotation
                    +
                    best_k * np.pi / 2.0
                )

                total_deg = float(
                    np.rad2deg(
                        total_rotation
                    )
                )

                while total_deg <= -180.0:
                    total_deg += 360.0

                while total_deg > 180.0:
                    total_deg -= 360.0

                phase_info[
                    "phase_rotation_deg"
                ] = round(
                    total_deg,
                    4
                )

                phase_info[
                    "phase_ambiguous"
                ] = False

                phase_info[
                    "phase_ambiguity_resolved"
                ] = True

                phase_info[
                    "phase_method"
                ] = (
                    "16-QAM fourth-order moment "
                    "+ reference-bit resolution"
                )

    return (
        np.asarray(
            bits,
            dtype=np.uint8
        ),
        phase_info
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

    sps = _estimate_sps(
        symbol_rate,
        fs,
        modulation
    )

    # --------------------------------------------------------
    # Frequency-offset correction
    # --------------------------------------------------------

    frequency_offset_hz = 0.0

    if modulation in (
        "BPSK",
        "QPSK",
        "16-QAM",
    ):

        frequency_offset_hz = (
            _estimate_frequency_offset(
                x,
                fs,
                modulation
            )
        )

        x = _correct_frequency_offset(
            x,
            fs,
            frequency_offset_hz
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
                symbols
            )

        elif modulation == "16-QAM":

            bits, qam_phase_info = _demod_16qam(
                symbols,
                reference_bits=reference_bits
            )

        else:

            raise ValueError(
                f"Unsupported modulation: "
                f"{modulation}"
            )

    if modulation != "16-QAM":
        qam_phase_info = {}

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

        "phase_rotation_deg": (
            qam_phase_info.get(
                "phase_rotation_deg"
            )
            if modulation == "16-QAM"
            else None
        ),

        "phase_ambiguity_deg": (
            qam_phase_info.get(
                "phase_ambiguity_deg"
            )
            if modulation == "16-QAM"
            else None
        ),

        "phase_ambiguous": (
            qam_phase_info.get(
                "phase_ambiguous"
            )
            if modulation == "16-QAM"
            else False
        ),

        "phase_method": (
            qam_phase_info.get(
                "phase_method"
            )
            if modulation == "16-QAM"
            else None
        ),

        "phase_reference_used": (
            qam_phase_info.get(
                "phase_reference_used"
            )
            if modulation == "16-QAM"
            else False
        ),

        "phase_reference_score": (
            qam_phase_info.get(
                "phase_reference_score"
            )
            if modulation == "16-QAM"
            else None
        ),

        "phase_ambiguity_resolved": (
            qam_phase_info.get(
                "phase_ambiguity_resolved"
            )
            if modulation == "16-QAM"
            else False
        ),

        "sample_rate": float(fs),

        "frequency_offset_hz": round(
            float(
                frequency_offset_hz
            ),
            2
        ),

        "frequency_offset_corrected": (
            modulation in (
                "BPSK",
                "QPSK",
                "16-QAM",
            )
        ),

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