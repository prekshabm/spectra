import numpy as np


# ============================================================
# CONVOLUTIONAL FEC
#
# Standard rate-1/2, constraint-length-3 code
# Generator polynomials:
#     G1 = 111 (octal 7)
#     G2 = 101 (octal 5)
#
# This module provides:
#     - convolutional_encode()
#     - viterbi_decode()
# ============================================================


DEFAULT_G1 = 0b111
DEFAULT_G2 = 0b101
DEFAULT_CONSTRAINT_LENGTH = 3


# ============================================================
# INPUT VALIDATION
# ============================================================

def _clean_bits(bits):
    """
    Convert input into a uint8 binary array.
    """

    if isinstance(bits, str):
        bits = [
            int(b)
            for b in bits
            if b in ("0", "1")
        ]

    bits = np.asarray(
        bits,
        dtype=np.uint8
    )

    if len(bits):
        if np.any(
            (bits != 0)
            &
            (bits != 1)
        ):
            raise ValueError(
                "Input bits must contain only 0 and 1."
            )

    return bits


def _parity(value):
    """
    Return XOR parity of an integer.
    """

    return int(
        value.bit_count() & 1
    )


# ============================================================
# CONVOLUTIONAL ENCODER
# ============================================================

def convolutional_encode(
    bits,
    g1=DEFAULT_G1,
    g2=DEFAULT_G2,
    constraint_length=DEFAULT_CONSTRAINT_LENGTH,
    terminate=True,
):
    """
    Encode binary bits using a rate-1/2 convolutional code.

    Parameters
    ----------
    bits:
        Binary input sequence.

    g1, g2:
        Generator polynomials represented as bit masks.

        Default:
            g1 = 111 binary = octal 7
            g2 = 101 binary = octal 5

    constraint_length:
        Constraint length K.

    terminate:
        When True, append K-1 zero bits to return the encoder
        to the all-zero state.

    Returns
    -------
    np.ndarray
        Encoded bits, two output bits per input bit.
    """

    bits = _clean_bits(bits)

    k = int(constraint_length)

    if k < 2:
        raise ValueError(
            "constraint_length must be at least 2."
        )

    g1 = int(g1)
    g2 = int(g2)

    if g1 <= 0 or g2 <= 0:
        raise ValueError(
            "Generator polynomials must be positive."
        )

    if len(bits) == 0:
        return bits.copy()

    state = 0
    mask = (1 << k) - 1

    sequence = bits.tolist()

    if terminate:
        sequence.extend(
            [0] * (k - 1)
        )

    encoded = []

    for bit in sequence:

        bit = int(bit)

        # Shift current input into the newest position.
        state = (
            (state << 1)
            | bit
        ) & mask

        out1 = _parity(
            state & g1
        )

        out2 = _parity(
            state & g2
        )

        encoded.extend(
            [out1, out2]
        )

    return np.asarray(
        encoded,
        dtype=np.uint8
    )


# ============================================================
# TRELLIS
# ============================================================

def _build_trellis(
    g1,
    g2,
    constraint_length,
):
    """
    Build next-state/output tables.
    """

    k = int(constraint_length)

    n_states = 1 << (k - 1)
    mask = n_states - 1

    next_state = np.zeros(
        (n_states, 2),
        dtype=np.int32
    )

    output_bits = np.zeros(
        (n_states, 2, 2),
        dtype=np.uint8
    )

    full_mask = (1 << k) - 1

    for state in range(n_states):

        for bit in (0, 1):

            full_state = (
                (state << 1)
                | bit
            ) & full_mask

            ns = full_state & mask

            o1 = _parity(
                full_state & int(g1)
            )

            o2 = _parity(
                full_state & int(g2)
            )

            next_state[
                state,
                bit
            ] = ns

            output_bits[
                state,
                bit
            ] = (
                o1,
                o2,
            )

    return (
        next_state,
        output_bits,
    )


# ============================================================
# VITERBI DECODER
# ============================================================

def viterbi_decode(
    received,
    g1=DEFAULT_G1,
    g2=DEFAULT_G2,
    constraint_length=DEFAULT_CONSTRAINT_LENGTH,
    terminated=True,
):
    """
    Hard-decision Viterbi decoding for the convolutional code.

    Parameters
    ----------
    received:
        Encoded/corrupted binary sequence.

    terminated:
        True when the encoder was terminated back to state zero.

    Returns
    -------
    np.ndarray
        Recovered information bits.
    """

    received = _clean_bits(
        received
    )

    if len(received) == 0:
        return received.copy()

    if len(received) % 2 != 0:
        raise ValueError(
            "Rate-1/2 convolutional code requires an even "
            "number of received bits."
        )

    k = int(constraint_length)

    if k < 2:
        raise ValueError(
            "constraint_length must be at least 2."
        )

    (
        next_state,
        output_bits,
    ) = _build_trellis(
        g1,
        g2,
        k,
    )

    n_states = next_state.shape[0]

    n_steps = len(received) // 2

    # --------------------------------------------------------
    # Path metrics
    # --------------------------------------------------------

    INF = 10**9

    metrics = np.full(
        n_states,
        INF,
        dtype=np.int64,
    )

    metrics[0] = 0

    # For traceback:
    # previous state and input bit for every step/state.
    predecessor = np.full(
        (n_steps, n_states),
        -1,
        dtype=np.int32,
    )

    survivor_bit = np.zeros(
        (n_steps, n_states),
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Forward recursion
    # --------------------------------------------------------

    for step in range(n_steps):

        r0 = int(
            received[2 * step]
        )

        r1 = int(
            received[2 * step + 1]
        )

        new_metrics = np.full(
            n_states,
            INF,
            dtype=np.int64,
        )

        for state in range(n_states):

            current_metric = metrics[
                state
            ]

            if current_metric >= INF:
                continue

            for bit in (0, 1):

                ns = int(
                    next_state[
                        state,
                        bit
                    ]
                )

                expected = output_bits[
                    state,
                    bit
                ]

                # Hamming branch metric.
                branch_metric = (
                    int(expected[0]) != r0
                ) + (
                    int(expected[1]) != r1
                )

                candidate = (
                    current_metric
                    + branch_metric
                )

                if candidate < new_metrics[ns]:

                    new_metrics[ns] = candidate

                    predecessor[
                        step,
                        ns
                    ] = state

                    survivor_bit[
                        step,
                        ns
                    ] = bit

        metrics = new_metrics

    # --------------------------------------------------------
    # Choose final state
    # --------------------------------------------------------

    if terminated:
        final_state = 0
    else:
        final_state = int(
            np.argmin(metrics)
        )

    if metrics[final_state] >= INF:
        raise RuntimeError(
            "Viterbi decoder could not find a valid path."
        )

    # --------------------------------------------------------
    # Traceback
    # --------------------------------------------------------

    decoded = np.zeros(
        n_steps,
        dtype=np.uint8,
    )

    state = final_state

    for step in range(
        n_steps - 1,
        -1,
        -1,
    ):

        prev = predecessor[
            step,
            state
        ]

        if prev < 0:
            raise RuntimeError(
                "Viterbi traceback failed."
            )

        decoded[
            step
        ] = survivor_bit[
            step,
            state
        ]

        state = prev

    # Remove termination bits.
    if terminated:
        if len(decoded) >= k - 1:
            decoded = decoded[
                :-(k - 1)
            ]

    return decoded.astype(
        np.uint8
    )


# ============================================================
# BIT ERROR UTILITY
# ============================================================

def bit_errors(a, b):
    """
    Count bit errors between two equal-length sequences.
    """

    a = _clean_bits(a)
    b = _clean_bits(b)

    if len(a) != len(b):
        raise ValueError(
            "Sequences must have equal length."
        )

    return int(
        np.count_nonzero(
            a != b
        )
    )


def bits_to_string(bits):
    """
    Convert bits to a string.
    """

    bits = _clean_bits(bits)

    return "".join(
        bits.astype(str)
    )


# ============================================================
# CLASS-STYLE WRAPPER
# ============================================================

class ConvolutionalFEC:

    def __init__(
        self,
        g1=DEFAULT_G1,
        g2=DEFAULT_G2,
        constraint_length=DEFAULT_CONSTRAINT_LENGTH,
    ):
        self.g1 = int(g1)
        self.g2 = int(g2)
        self.constraint_length = int(
            constraint_length
        )

    def encode(
        self,
        bits,
        terminate=True,
    ):
        return convolutional_encode(
            bits,
            self.g1,
            self.g2,
            self.constraint_length,
            terminate,
        )

    def decode(
        self,
        received,
        terminated=True,
    ):
        return viterbi_decode(
            received,
            self.g1,
            self.g2,
            self.constraint_length,
            terminated,
        )