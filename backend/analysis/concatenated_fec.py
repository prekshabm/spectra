import numpy as np

from backend.analysis.reed_solomon_fec import (
    reed_solomon_encode,
    reed_solomon_decode,
)

from backend.analysis.convolutional_fec import (
    convolutional_encode,
    viterbi_decode,
)


DEFAULT_NSYM = 10


# ============================================================
# BYTE <-> BIT CONVERSION
# ============================================================

def bytes_to_bits(data):
    """
    Convert uint8 bytes to a flat bit array.
    MSB first.
    """
    data = np.asarray(data, dtype=np.uint8)

    return np.unpackbits(data)


def bits_to_bytes(bits):
    """
    Convert a bit array back to uint8 bytes.
    """
    bits = np.asarray(bits, dtype=np.uint8)

    if len(bits) % 8 != 0:
        raise ValueError(
            "Number of bits must be divisible by 8."
        )

    return np.packbits(bits)


# ============================================================
# CONCATENATED ENCODER
#
# Outer code  = Reed-Solomon
# Inner code  = Convolutional
# ============================================================

def concatenated_encode(
    data,
    nsym=DEFAULT_NSYM,
):
    """
    Encode data using concatenated FEC:

        Data
          ↓
        Reed-Solomon
          ↓
        Convolutional
          ↓
        Encoded bits
    """

    # Outer FEC
    rs_encoded = reed_solomon_encode(
        data,
        nsym=nsym,
    )

    # Convert RS bytes to bits.
    rs_bits = bytes_to_bits(
        rs_encoded
    )

    # Inner FEC.
    conv_encoded = convolutional_encode(
        rs_bits,
        terminate=True,
    )

    return conv_encoded


# ============================================================
# CONCATENATED DECODER
#
# Convolutional Viterbi first
# Then Reed-Solomon
# ============================================================

def concatenated_decode(
    received_bits,
    nsym=DEFAULT_NSYM,
):
    """
    Decode concatenated FEC:

        Received bits
              ↓
        Viterbi decoder
              ↓
        Reed-Solomon decoder
              ↓
        Original data
    """

    received_bits = np.asarray(
        received_bits,
        dtype=np.uint8,
    )

    # Inner decoder.
    conv_decoded_bits = viterbi_decode(
        received_bits,
        terminated=True,
    )

    # Convert decoded bits to RS bytes.
    rs_received = bits_to_bytes(
        conv_decoded_bits
    )

    # Outer decoder.
    recovered = reed_solomon_decode(
        rs_received,
        nsym=nsym,
    )

    return recovered


# ============================================================
# CONTROLLED BIT ERROR INJECTION
# ============================================================

def introduce_bit_errors(
    bits,
    positions,
):
    """
    Flip selected bit positions.
    """

    bits = np.asarray(
        bits,
        dtype=np.uint8,
    ).copy()

    positions = np.asarray(
        positions,
        dtype=np.int64,
    )

    if np.any(positions < 0):
        raise IndexError(
            "Bit position cannot be negative."
        )

    if np.any(positions >= len(bits)):
        raise IndexError(
            "Bit position is outside the sequence."
        )

    bits[positions] ^= 1

    return bits


# ============================================================
# BYTE ERROR COUNT
# ============================================================

def byte_errors(
    original,
    recovered,
):
    """
    Count differing byte positions.
    """

    original = np.asarray(
        original,
        dtype=np.uint8,
    )

    recovered = np.asarray(
        recovered,
        dtype=np.uint8,
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

class ConcatenatedFEC:

    def __init__(
        self,
        nsym=DEFAULT_NSYM,
    ):
        self.nsym = int(nsym)

    def encode(self, data):
        return concatenated_encode(
            data,
            nsym=self.nsym,
        )

    def decode(self, received_bits):
        return concatenated_decode(
            received_bits,
            nsym=self.nsym,
        )