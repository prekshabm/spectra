import numpy as np


# ============================================================
# SIMPLE LDPC FEC
#
# Regular (3,6) LDPC-style code
#
# K = 12 information bits
# N = 24 coded bits
#
# H = [A | I]
#
# The decoder uses syndrome-based single-bit correction.
# ============================================================


DEFAULT_K = 12
DEFAULT_N = 24


# ============================================================
# PARITY CHECK MATRIX
# ============================================================

def _build_H():

    A = np.array(
        [
            [1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
            [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        ],
        dtype=np.uint8,
    )

    identity = np.eye(
        DEFAULT_K,
        dtype=np.uint8,
    )

    return np.concatenate(
        [A, identity],
        axis=1,
    )


H = _build_H()


# ============================================================
# INPUT VALIDATION
# ============================================================

def _clean_bits(bits):

    bits = np.asarray(
        bits,
        dtype=np.uint8,
    ).flatten()

    if len(bits):
        if np.any(
            (bits != 0) &
            (bits != 1)
        ):
            raise ValueError(
                "Input must contain only 0 and 1."
            )

    return bits


# ============================================================
# ENCODER
# ============================================================

def ldpc_encode(bits):

    bits = _clean_bits(bits)

    if len(bits) % DEFAULT_K != 0:
        raise ValueError(
            f"Input length must be a multiple of "
            f"{DEFAULT_K} bits."
        )

    encoded_blocks = []

    A = H[:, :DEFAULT_K]

    for start in range(
        0,
        len(bits),
        DEFAULT_K,
    ):

        message = bits[
            start:start + DEFAULT_K
        ]

        # GF(2) parity calculation.
        parity = (
            A @ message
        ) % 2

        codeword = np.concatenate(
            [
                message,
                parity.astype(np.uint8),
            ]
        )

        encoded_blocks.append(
            codeword
        )

    return np.concatenate(
        encoded_blocks
    ).astype(np.uint8)


# ============================================================
# SYNDROME
# ============================================================

def ldpc_syndrome(bits):

    bits = _clean_bits(bits)

    if len(bits) % DEFAULT_N != 0:
        raise ValueError(
            f"Input length must be a multiple of "
            f"{DEFAULT_N} bits."
        )

    all_syndromes = []

    for start in range(
        0,
        len(bits),
        DEFAULT_N,
    ):

        codeword = bits[
            start:start + DEFAULT_N
        ]

        syndrome = (
            H @ codeword
        ) % 2

        all_syndromes.append(
            syndrome.astype(np.uint8)
        )

    return np.concatenate(
        all_syndromes
    )


# ============================================================
# SYNDROME -> INTEGER
# ============================================================

def _syndrome_key(syndrome):

    key = 0

    for i, bit in enumerate(syndrome):

        if int(bit):
            key |= (
                1 << i
            )

    return key


# ============================================================
# BUILD SINGLE-BIT ERROR TABLE
# ============================================================

def _build_single_error_table():

    table = {}

    for bit_position in range(
        DEFAULT_N
    ):

        syndrome = H[
            :,
            bit_position
        ]

        key = _syndrome_key(
            syndrome
        )

        # A non-zero syndrome identifies
        # the corresponding bit position.
        if key != 0:
            table[key] = bit_position

    return table


SINGLE_ERROR_TABLE = (
    _build_single_error_table()
)


# ============================================================
# DECODE ONE CODEWORD
# ============================================================

def _decode_codeword(
    received,
):

    bits = received.copy()

    syndrome = (
        H @ bits
    ) % 2

    # Already valid.
    if not np.any(syndrome):

        return (
            bits,
            True,
            0,
        )

    # --------------------------------------------------------
    # Single-bit correction
    # --------------------------------------------------------

    key = _syndrome_key(
        syndrome
    )

    bit_position = (
        SINGLE_ERROR_TABLE.get(
            key
        )
    )

    if bit_position is not None:

        bits[
            bit_position
        ] ^= 1

        corrected_syndrome = (
            H @ bits
        ) % 2

        success = not np.any(
            corrected_syndrome
        )

        return (
            bits,
            success,
            1,
        )

    # --------------------------------------------------------
    # Fallback: simple iterative bit flipping
    # --------------------------------------------------------

    for iteration in range(
        1,
        21,
    ):

        syndrome = (
            H @ bits
        ) % 2

        if not np.any(
            syndrome
        ):

            return (
                bits,
                True,
                iteration,
            )

        scores = (
            H.T @ syndrome
        )

        max_score = int(
            np.max(scores)
        )

        if max_score == 0:
            break

        # Flip strongest candidate.
        position = int(
            np.argmax(scores)
        )

        bits[position] ^= 1

    syndrome = (
        H @ bits
    ) % 2

    success = not np.any(
        syndrome
    )

    return (
        bits,
        success,
        20,
    )


# ============================================================
# DECODER
# ============================================================

def ldpc_decode(
    received,
    max_iterations=20,
):

    received = _clean_bits(
        received
    )

    if len(received) % DEFAULT_N != 0:
        raise ValueError(
            f"Received length must be a multiple of "
            f"{DEFAULT_N} bits."
        )

    messages = []

    for start in range(
        0,
        len(received),
        DEFAULT_N,
    ):

        codeword = received[
            start:start + DEFAULT_N
        ].copy()

        corrected, success, _ = (
            _decode_codeword(
                codeword
            )
        )

        if not success:

            raise RuntimeError(
                "LDPC decoder could not correct "
                "the received codeword."
            )

        # First 12 bits = original information.
        message = corrected[
            :DEFAULT_K
        ]

        messages.append(
            message
        )

    return np.concatenate(
        messages
    ).astype(np.uint8)


# ============================================================
# ERROR INJECTION
# ============================================================

def introduce_bit_errors(
    bits,
    positions,
):

    bits = _clean_bits(
        bits
    ).copy()

    positions = np.asarray(
        positions,
        dtype=np.int64,
    )

    if np.any(
        positions < 0
    ):
        raise IndexError(
            "Bit position cannot be negative."
        )

    if np.any(
        positions >= len(bits)
    ):
        raise IndexError(
            "Bit position is outside the sequence."
        )

    bits[
        positions
    ] ^= 1

    return bits


# ============================================================
# BIT ERROR COUNT
# ============================================================

def bit_errors(
    original,
    recovered,
):

    original = _clean_bits(
        original
    )

    recovered = _clean_bits(
        recovered
    )

    if len(original) != len(recovered):
        raise ValueError(
            "Sequences must have equal length."
        )

    return int(
        np.count_nonzero(
            original != recovered
        )
    )


# ============================================================
# CLASS WRAPPER
# ============================================================

class LDPCCoder:

    def __init__(
        self,
        max_iterations=20,
    ):

        self.max_iterations = int(
            max_iterations
        )

    def encode(self, bits):

        return ldpc_encode(
            bits
        )

    def decode(self, received):

        return ldpc_decode(
            received,
            max_iterations=self.max_iterations,
        )

    def syndrome(self, bits):

        return ldpc_syndrome(
            bits
        )